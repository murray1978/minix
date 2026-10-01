
/*
 * minix_llama.c
 *
 * A small FP32 Llama 2 inference engine for MINIX 3.3.0, compatible with
 * the version-0 checkpoint and tokenizer formats used by karpathy/llama2.c.
 *
 * Design goals for this MINIX port:
 *   - ordinary unprivileged process
 *   - buffered regular-file I/O (no mmap dependency)
 *   - checked size arithmetic for 32-bit size_t
 *   - strict checkpoint/tokenizer validation
 *   - deterministic greedy and seeded top-p generation
 *   - optional binary logit traces for host/MINIX comparison
 *
 * The transformer architecture and file formats are based on llama2.c:
 *   https://github.com/karpathy/llama2.c
 *
 * llama2.c is MIT licensed; see THIRD_PARTY_NOTICE.md.
 */

#include <ctype.h>
#include <errno.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TRACE_MAGIC "MLTR"
#define TRACE_VERSION 1U
#define MAX_TOKEN_BYTES (1024U * 1024U)
#define MAX_TOKENIZER_BYTES (64U * 1024U * 1024U)

#if defined(__clang__) || defined(__GNUC__)
#define NORETURN __attribute__((__noreturn__))
#else
#define NORETURN
#endif

static void die(const char *message) NORETURN;
static void die_errno(const char *operation, const char *path) NORETURN;
static void usage(const char *program) NORETURN;
static int read_exact(FILE *file, void *buffer, size_t bytes);
static void *checked_calloc(size_t count, size_t size, const char *what);

typedef struct {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
    int shared_weights;
} Config;

typedef struct {
    float *token_embedding_table;
    float *rms_att_weight;
    float *wq;
    float *wk;
    float *wv;
    float *wo;
    float *rms_ffn_weight;
    float *w1;
    float *w2;
    float *w3;
    float *rms_final_weight;
    float *wcls;
} TransformerWeights;

typedef struct {
    float *x;
    float *xb;
    float *xb2;
    float *hb;
    float *hb2;
    float *q;
    float *att;
    float *logits;
    float *key_cache;
    float *value_cache;
} RunState;

typedef struct {
    Config config;
    TransformerWeights weights;
    RunState state;
    float *weight_data;
    size_t weight_count;
    size_t checkpoint_bytes;
    size_t run_state_bytes;
} Transformer;

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char **vocab;
    float *vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512];
    size_t allocated_bytes;
} Tokenizer;

typedef struct {
    float prob;
    int index;
} ProbIndex;

typedef struct {
    int vocab_size;
    ProbIndex *probindex;
    float temperature;
    float topp;
    uint64_t rng_state;
} Sampler;

typedef struct {
    FILE *file;
    uint32_t vocab_size;
    uint32_t entries;
} TraceWriter;

typedef struct {
    FILE *file;
    int enabled;
    int max_positions;
} ForwardDiag;

typedef struct {
    char magic[4];
    uint32_t version;
    uint32_t rank;
    uint32_t dim;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t hidden_dim;
    uint32_t n_layers;
    uint32_t vocab_size;
    uint32_t seq_len;
    uint32_t reserved0;
    uint64_t checkpoint_bytes;
    uint8_t checkpoint_sha256[32];
    uint32_t layout_version;
    uint32_t reserved1;
    uint64_t a_params;
    uint64_t b_params;
    uint64_t a_bytes;
    uint64_t b_bytes;
    uint64_t params_count;
    uint64_t data_bytes;
} AdapterFileHeader;

typedef struct {
    int enabled;
    int rank;
    float scale;
    float *a;
    float *b;
    float *u;
} AdapterRuntime;

static ForwardDiag g_forward_diag;
static AdapterRuntime g_adapter_runtime;

static uint64_t checksum_f32(const float *values, size_t count);
static void diag_log_vector(const char *label, int position, int layer,
    const float *values, size_t count);
static void diag_log_top_logits(int position, const float *logits, int vocab_size);
int minix_llama_set_diagnostic_output(const char *path, int max_positions);
void minix_llama_clear_diagnostic_output(void);

static uint64_t checksum_f32(const float *values, size_t count)
{
    size_t i;
    uint64_t hash = UINT64_C(1469598103934665603);

    for (i = 0; i < count; i++) {
        union {
            float f;
            uint32_t u;
        } bits;

        bits.f = values[i];
        hash ^= (uint64_t)bits.u;
        hash *= UINT64_C(1099511628211);
    }

    return hash;
}

static void diag_log_vector(const char *label, int position, int layer,
    const float *values, size_t count)
{
    uint64_t hash;

    if (!g_forward_diag.enabled || g_forward_diag.file == NULL)
        return;
    if (position >= g_forward_diag.max_positions)
        return;

    hash = checksum_f32(values, count);
    if (layer >= 0) {
        fprintf(g_forward_diag.file,
            "diag pos=%d layer=%d %s count=%lu checksum=%016llx\n",
            position, layer, label, (unsigned long)count,
            (unsigned long long)hash);
    } else {
        fprintf(g_forward_diag.file,
            "diag pos=%d %s count=%lu checksum=%016llx\n",
            position, label, (unsigned long)count,
            (unsigned long long)hash);
    }
}

static void diag_log_top_logits(int position, const float *logits, int vocab_size)
{
    int rank;
    int i;
    int ids[10];
    float vals[10];

    if (!g_forward_diag.enabled || g_forward_diag.file == NULL)
        return;
    if (position >= g_forward_diag.max_positions)
        return;

    for (rank = 0; rank < 10; rank++) {
        ids[rank] = -1;
        vals[rank] = -FLT_MAX;
    }

    for (i = 0; i < vocab_size; i++) {
        float value = logits[i];
        for (rank = 0; rank < 10; rank++) {
            if (value > vals[rank]) {
                int j;
                for (j = 9; j > rank; j--) {
                    vals[j] = vals[j - 1];
                    ids[j] = ids[j - 1];
                }
                vals[rank] = value;
                ids[rank] = i;
                break;
            }
        }
    }

    fprintf(g_forward_diag.file, "diag pos=%d top10", position);
    for (rank = 0; rank < 10; rank++) {
        fprintf(g_forward_diag.file, " [%d]=%.9g", ids[rank], vals[rank]);
    }
    fprintf(g_forward_diag.file, "\n");
}

int minix_llama_set_diagnostic_output(const char *path, int max_positions)
{
    FILE *file;

    minix_llama_clear_diagnostic_output();

    if (path == NULL || path[0] == '\0')
        return 0;

    file = fopen(path, "wb");
    if (file == NULL)
        return -1;

    g_forward_diag.file = file;
    g_forward_diag.enabled = 1;
    g_forward_diag.max_positions = (max_positions > 0) ? max_positions : 4;

    fprintf(g_forward_diag.file,
        "diag start max_positions=%d\n", g_forward_diag.max_positions);
    fflush(g_forward_diag.file);
    return 0;
}

void minix_llama_clear_diagnostic_output(void)
{
    if (g_forward_diag.file != NULL)
        fclose(g_forward_diag.file);

    g_forward_diag.file = NULL;
    g_forward_diag.enabled = 0;
    g_forward_diag.max_positions = 0;
}

static void unload_adapter_runtime(void)
{
    free(g_adapter_runtime.a);
    free(g_adapter_runtime.b);
    free(g_adapter_runtime.u);
    memset(&g_adapter_runtime, 0, sizeof(g_adapter_runtime));
}

static int load_adapter_runtime(const char *path,
    const Transformer *transformer, float scale)
{
    FILE *file;
    AdapterFileHeader hdr;
    const Config *cfg = &transformer->config;
    size_t a_count;
    size_t b_count;
    size_t total_count;
    uint64_t expected_data_bytes;
    size_t i;
    const char *error = NULL;

    unload_adapter_runtime();

    if (path == NULL || path[0] == '\0') {
        fprintf(stderr, "ai_model: required adapter path is empty\n");
        return 0;
    }

    file = fopen(path, "rb");
    if (file == NULL) {
        fprintf(stderr, "ai_model: cannot open required adapter '%s': %s\n",
            path, strerror(errno));
        return 0;
    }

    if (!read_exact(file, &hdr, sizeof(hdr))) {
        error = "cannot read complete adapter header";
        goto fail;
    }

    if (memcmp(hdr.magic, "MLAD", 4) != 0) {
        error = "adapter has invalid magic";
        goto fail;
    }
    if (hdr.version != 2U) {
        error = "adapter format version is not 2";
        goto fail;
    }
    if (hdr.layout_version != 1U) {
        error = "adapter layout version is not 1";
        goto fail;
    }
    if (hdr.rank != 8U) {
        error = "adapter rank is not 8";
        goto fail;
    }

    if (cfg->dim != 288 || cfg->vocab_size != 32000 ||
        (int)hdr.dim != cfg->dim ||
        (int)hdr.vocab_size != cfg->vocab_size) {
        error = "adapter dimensions do not match required model configuration";
        goto fail;
    }

    a_count = (size_t)hdr.rank * (size_t)cfg->dim;
    b_count = (size_t)cfg->vocab_size * (size_t)hdr.rank;
    total_count = a_count + b_count;
    expected_data_bytes = (uint64_t)total_count * (uint64_t)sizeof(float);

    if (hdr.a_params != (uint64_t)a_count ||
        hdr.b_params != (uint64_t)b_count ||
        hdr.params_count != (uint64_t)total_count ||
        hdr.a_bytes != (uint64_t)(a_count * sizeof(float)) ||
        hdr.b_bytes != (uint64_t)(b_count * sizeof(float)) ||
        hdr.data_bytes != expected_data_bytes) {
        error = "adapter tensor accounting does not match expected A/B layout";
        goto fail;
    }

    g_adapter_runtime.a = calloc(a_count, sizeof(float));
    g_adapter_runtime.b = calloc(b_count, sizeof(float));
    g_adapter_runtime.u = calloc((size_t)hdr.rank, sizeof(float));
    if (g_adapter_runtime.a == NULL || g_adapter_runtime.b == NULL ||
        g_adapter_runtime.u == NULL) {
        error = "cannot allocate persistent adapter tensors";
        goto fail;
    }

    if (!read_exact(file, g_adapter_runtime.a, a_count * sizeof(float)) ||
        !read_exact(file, g_adapter_runtime.b, b_count * sizeof(float))) {
        error = "adapter payload is truncated";
        goto fail;
    }
    if (fgetc(file) != EOF || ferror(file)) {
        error = "adapter has trailing or unreadable data";
        goto fail;
    }

    if (fclose(file) != 0) {
        file = NULL;
        error = "cannot close adapter file";
        goto fail;
    }
    file = NULL;

    for (i = 0; i < a_count; i++) {
        if (!isfinite(g_adapter_runtime.a[i])) {
            error = "adapter A contains a non-finite value";
            goto fail;
        }
    }
    for (i = 0; i < b_count; i++) {
        if (!isfinite(g_adapter_runtime.b[i])) {
            error = "adapter B contains a non-finite value";
            goto fail;
        }
    }

    g_adapter_runtime.rank = (int)hdr.rank;
    g_adapter_runtime.scale = scale;
    g_adapter_runtime.enabled = 1;
    return 1;

fail:
    if (file != NULL)
        fclose(file);
    unload_adapter_runtime();
    fprintf(stderr, "ai_model: required adapter '%s' rejected: %s\n",
        path, error != NULL ? error : "unknown validation failure");
    return 0;
}

static void apply_adapter_correction(const Transformer *t,
    const float *hidden, float *logits)
{
    const Config *cfg = &t->config;
    const float *a = g_adapter_runtime.a;
    const float *b = g_adapter_runtime.b;
    float *u = g_adapter_runtime.u;
    int rank = g_adapter_runtime.rank;
    int r;
    int i;

    if (!g_adapter_runtime.enabled)
        return;

    for (r = 0; r < rank; r++) {
        const float *row = a + (size_t)r * (size_t)cfg->dim;
        double sum = 0.0;
        for (i = 0; i < cfg->dim; i++)
            sum += (double)row[i] * (double)hidden[i];
        u[r] = (float)sum;
    }

    for (i = 0; i < cfg->vocab_size; i++) {
        const float *row = b + (size_t)i * (size_t)rank;
        double correction = 0.0;
        int j;

        for (j = 0; j < rank; j++)
            correction += (double)row[j] * (double)u[j];

        logits[i] += g_adapter_runtime.scale * (float)correction;
    }
}

static void die(const char *message)
{
    fprintf(stderr, "error: %s\n", message);
    exit(EXIT_FAILURE);
}

static void die_errno(const char *operation, const char *path)
{
    if (path != NULL)
        fprintf(stderr, "error: %s '%s': %s\n", operation, path, strerror(errno));
    else
        fprintf(stderr, "error: %s: %s\n", operation, strerror(errno));
    exit(EXIT_FAILURE);
}

static int checked_add_size(size_t a, size_t b, size_t *out)
{
    if (out == NULL || b > SIZE_MAX - a)
        return 0;
    *out = a + b;
    return 1;
}

static int checked_mul_size(size_t a, size_t b, size_t *out)
{
    if (out == NULL || (a != 0 && b > SIZE_MAX / a))
        return 0;
    *out = a * b;
    return 1;
}

static int checked_add_count(size_t *total, size_t value)
{
    size_t result;
    if (total == NULL || !checked_add_size(*total, value, &result))
        return 0;
    *total = result;
    return 1;
}

static int checked_product2(size_t a, size_t b, size_t *out)
{
    return checked_mul_size(a, b, out);
}

static int checked_product3(size_t a, size_t b, size_t c, size_t *out)
{
    size_t temp;
    return checked_mul_size(a, b, &temp) && checked_mul_size(temp, c, out);
}

static void *checked_calloc(size_t count, size_t size, const char *what)
{
    void *pointer;
    size_t bytes;

    if (!checked_mul_size(count, size, &bytes)) {
        fprintf(stderr, "error: size overflow allocating %s\n", what);
        exit(EXIT_FAILURE);
    }

    pointer = calloc(count, size);
    if (pointer == NULL && bytes != 0) {
        fprintf(stderr, "error: cannot allocate %lu bytes for %s: %s\n",
            (unsigned long)bytes, what, strerror(errno));
        exit(EXIT_FAILURE);
    }

    return pointer;
}

static int host_is_little_endian(void)
{
    uint32_t value = 1U;
    return *((unsigned char *)&value) == 1U;
}

static int read_exact(FILE *file, void *buffer, size_t bytes)
{
    unsigned char *cursor = (unsigned char *)buffer;
    size_t done = 0;

    while (done < bytes) {
        size_t count = fread(cursor + done, 1, bytes - done, file);
        if (count == 0) {
            if (ferror(file))
                return 0;
            return 0;
        }
        done += count;
    }
    return 1;
}

static int write_exact(FILE *file, const void *buffer, size_t bytes)
{
    const unsigned char *cursor = (const unsigned char *)buffer;
    size_t done = 0;

    while (done < bytes) {
        size_t count = fwrite(cursor + done, 1, bytes - done, file);
        if (count == 0)
            return 0;
        done += count;
    }
    return 1;
}

static size_t regular_file_size(FILE *file, const char *path)
{
    off_t end;

    if (fseeko(file, (off_t)0, SEEK_END) != 0)
        die_errno("seeking to end of", path);

    end = ftello(file);
    if (end < 0)
        die_errno("getting size of", path);

    if ((uint64_t)end > (uint64_t)SIZE_MAX)
        die("file is too large for this 32-bit process");

    if (fseeko(file, (off_t)0, SEEK_SET) != 0)
        die_errno("rewinding", path);

    return (size_t)end;
}

static int validate_config(Config *p)
{
    if (p->dim <= 0 || p->hidden_dim <= 0 || p->n_layers <= 0 ||
        p->n_heads <= 0 || p->n_kv_heads <= 0 ||
        p->vocab_size <= 3 || p->seq_len <= 0)
        return 0;

    if (p->dim % p->n_heads != 0)
        return 0;

    if (p->n_heads % p->n_kv_heads != 0)
        return 0;

    if ((p->dim / p->n_heads) % 2 != 0)
        return 0;

    /*
     * Conservative bounds for a 32-bit experimental runtime. These reject
     * corrupt headers before any multiplication or allocation.
     */
    if (p->dim > 65536 || p->hidden_dim > 262144 ||
        p->n_layers > 4096 || p->n_heads > 4096 ||
        p->vocab_size > 1000000 || p->seq_len > 1048576)
        return 0;

    return 1;
}

static int expected_weight_count(const Config *p, size_t *count_out)
{
    size_t total = 0;
    size_t count;
    size_t dim = (size_t)p->dim;
    size_t hidden = (size_t)p->hidden_dim;
    size_t layers = (size_t)p->n_layers;
    size_t vocab = (size_t)p->vocab_size;
    size_t seq = (size_t)p->seq_len;
    size_t head_size = dim / (size_t)p->n_heads;
    size_t kv_dim;

    if (!checked_mul_size(dim, (size_t)p->n_kv_heads, &kv_dim))
        return 0;
    kv_dim /= (size_t)p->n_heads;

    if (!checked_product2(vocab, dim, &count) || !checked_add_count(&total, count))
        return 0; /* token embedding */

    if (!checked_product2(layers, dim, &count) || !checked_add_count(&total, count))
        return 0; /* rms_att */

    if (!checked_product3(layers, dim, dim, &count) || !checked_add_count(&total, count))
        return 0; /* wq */

    if (!checked_product3(layers, dim, kv_dim, &count) || !checked_add_count(&total, count))
        return 0; /* wk */

    if (!checked_product3(layers, dim, kv_dim, &count) || !checked_add_count(&total, count))
        return 0; /* wv */

    if (!checked_product3(layers, dim, dim, &count) || !checked_add_count(&total, count))
        return 0; /* wo */

    if (!checked_product2(layers, dim, &count) || !checked_add_count(&total, count))
        return 0; /* rms_ffn */

    if (!checked_product3(layers, hidden, dim, &count) || !checked_add_count(&total, count))
        return 0; /* w1 */

    if (!checked_product3(layers, dim, hidden, &count) || !checked_add_count(&total, count))
        return 0; /* w2 */

    if (!checked_product3(layers, hidden, dim, &count) || !checked_add_count(&total, count))
        return 0; /* w3 */

    if (!checked_add_count(&total, dim))
        return 0; /* rms_final */

    if (!checked_product2(seq, head_size / 2U, &count) ||
        !checked_add_count(&total, count) ||
        !checked_add_count(&total, count))
        return 0; /* legacy RoPE tables */

    if (!p->shared_weights) {
        if (!checked_product2(vocab, dim, &count) ||
            !checked_add_count(&total, count))
            return 0;
    }

    *count_out = total;
    return 1;
}

static float *take_weights(float **cursor, size_t count)
{
    float *result = *cursor;
    *cursor += count;
    return result;
}

static void map_weights(Transformer *t)
{
    Config *p = &t->config;
    TransformerWeights *w = &t->weights;
    float *cursor = t->weight_data;
    size_t dim = (size_t)p->dim;
    size_t hidden = (size_t)p->hidden_dim;
    size_t layers = (size_t)p->n_layers;
    size_t vocab = (size_t)p->vocab_size;
    size_t seq = (size_t)p->seq_len;
    size_t head_size = dim / (size_t)p->n_heads;
    size_t kv_dim = (dim * (size_t)p->n_kv_heads) / (size_t)p->n_heads;

    w->token_embedding_table = take_weights(&cursor, vocab * dim);
    w->rms_att_weight = take_weights(&cursor, layers * dim);
    w->wq = take_weights(&cursor, layers * dim * dim);
    w->wk = take_weights(&cursor, layers * dim * kv_dim);
    w->wv = take_weights(&cursor, layers * dim * kv_dim);
    w->wo = take_weights(&cursor, layers * dim * dim);
    w->rms_ffn_weight = take_weights(&cursor, layers * dim);
    w->w1 = take_weights(&cursor, layers * hidden * dim);
    w->w2 = take_weights(&cursor, layers * dim * hidden);
    w->w3 = take_weights(&cursor, layers * hidden * dim);
    w->rms_final_weight = take_weights(&cursor, dim);

    cursor += seq * head_size / 2U; /* legacy freq_cis_real */
    cursor += seq * head_size / 2U; /* legacy freq_cis_imag */

    if (p->shared_weights)
        w->wcls = w->token_embedding_table;
    else
        w->wcls = take_weights(&cursor, vocab * dim);

    if ((size_t)(cursor - t->weight_data) != t->weight_count)
        die("internal weight-layout mismatch");
}

static void allocate_run_state(Transformer *t)
{
    Config *p = &t->config;
    RunState *s = &t->state;
    size_t dim = (size_t)p->dim;
    size_t hidden = (size_t)p->hidden_dim;
    size_t layers = (size_t)p->n_layers;
    size_t seq = (size_t)p->seq_len;
    size_t heads = (size_t)p->n_heads;
    size_t vocab = (size_t)p->vocab_size;
    size_t kv_dim;
    size_t cache_count;
    size_t bytes = 0;
    size_t temp;

    if (!checked_mul_size(dim, (size_t)p->n_kv_heads, &kv_dim))
        die("KV dimension overflow");
    kv_dim /= (size_t)p->n_heads;

    if (!checked_product3(layers, seq, kv_dim, &cache_count))
        die("KV cache size overflow");

    s->x = checked_calloc(dim, sizeof(float), "x");
    s->xb = checked_calloc(dim, sizeof(float), "xb");
    s->xb2 = checked_calloc(dim, sizeof(float), "xb2");
    s->hb = checked_calloc(hidden, sizeof(float), "hb");
    s->hb2 = checked_calloc(hidden, sizeof(float), "hb2");
    s->q = checked_calloc(dim, sizeof(float), "q");
    s->att = checked_calloc(heads * seq, sizeof(float), "attention scores");
    s->logits = checked_calloc(vocab, sizeof(float), "logits");
    s->key_cache = checked_calloc(cache_count, sizeof(float), "key cache");
    s->value_cache = checked_calloc(cache_count, sizeof(float), "value cache");

#define ADD_FLOAT_BYTES(n) \
    do { \
        if (!checked_mul_size((n), sizeof(float), &temp) || \
            !checked_add_count(&bytes, temp)) \
            die("run-state byte count overflow"); \
    } while (0)

    ADD_FLOAT_BYTES(dim);
    ADD_FLOAT_BYTES(dim);
    ADD_FLOAT_BYTES(dim);
    ADD_FLOAT_BYTES(hidden);
    ADD_FLOAT_BYTES(hidden);
    ADD_FLOAT_BYTES(dim);
    ADD_FLOAT_BYTES(heads * seq);
    ADD_FLOAT_BYTES(vocab);
    ADD_FLOAT_BYTES(cache_count);
    ADD_FLOAT_BYTES(cache_count);

#undef ADD_FLOAT_BYTES

    t->run_state_bytes = bytes;
}

static void clear_run_state(Transformer *t)
{
    Config *p = &t->config;
    RunState *s = &t->state;
    size_t dim = (size_t)p->dim;
    size_t hidden = (size_t)p->hidden_dim;
    size_t heads = (size_t)p->n_heads;
    size_t seq = (size_t)p->seq_len;
    size_t vocab = (size_t)p->vocab_size;
    size_t kv_dim = (dim * (size_t)p->n_kv_heads) / (size_t)p->n_heads;
    size_t cache_count = (size_t)p->n_layers * seq * kv_dim;

    memset(s->x, 0, dim * sizeof(float));
    memset(s->xb, 0, dim * sizeof(float));
    memset(s->xb2, 0, dim * sizeof(float));
    memset(s->hb, 0, hidden * sizeof(float));
    memset(s->hb2, 0, hidden * sizeof(float));
    memset(s->q, 0, dim * sizeof(float));
    memset(s->att, 0, heads * seq * sizeof(float));
    memset(s->logits, 0, vocab * sizeof(float));
    memset(s->key_cache, 0, cache_count * sizeof(float));
    memset(s->value_cache, 0, cache_count * sizeof(float));
}

static void free_transformer(Transformer *t)
{
    RunState *s = &t->state;

    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
    free(t->weight_data);

    memset(t, 0, sizeof(*t));
}

static void load_transformer(Transformer *t, const char *checkpoint_path)
{
    FILE *file;
    int32_t disk[7];
    int32_t raw_vocab;
    size_t expected_count;
    size_t payload_bytes = 0;
    size_t expected_bytes = 0;
    size_t actual_bytes;

    memset(t, 0, sizeof(*t));

    if (!host_is_little_endian())
        die("version-0 llama2.c checkpoints require a little-endian host");

    file = fopen(checkpoint_path, "rb");
    if (file == NULL)
        die_errno("opening checkpoint", checkpoint_path);

    actual_bytes = regular_file_size(file, checkpoint_path);

    if (actual_bytes < sizeof(disk)) {
        fclose(file);
        die("checkpoint is shorter than the 28-byte header");
    }

    if (!read_exact(file, disk, sizeof(disk))) {
        fclose(file);
        die("cannot read checkpoint header");
    }

    raw_vocab = disk[5];
    if (raw_vocab == INT32_MIN) {
        fclose(file);
        die("invalid signed vocabulary size");
    }

    t->config.dim = disk[0];
    t->config.hidden_dim = disk[1];
    t->config.n_layers = disk[2];
    t->config.n_heads = disk[3];
    t->config.n_kv_heads = disk[4];
    t->config.shared_weights = raw_vocab > 0;
    t->config.vocab_size = raw_vocab < 0 ? -raw_vocab : raw_vocab;
    t->config.seq_len = disk[6];

    if (!validate_config(&t->config)) {
        fclose(file);
        die("invalid or unsupported checkpoint dimensions");
    }

    if (!expected_weight_count(&t->config, &expected_count)) {
        fclose(file);
        die("checkpoint weight-count overflow");
    }

    if (!checked_mul_size(expected_count, sizeof(float), &payload_bytes) ||
        !checked_add_size(sizeof(disk), payload_bytes, &expected_bytes)) {
        fclose(file);
        die("checkpoint byte-size overflow");
    }

    if (actual_bytes != expected_bytes) {
        fprintf(stderr,
            "error: checkpoint length mismatch: expected %lu bytes, found %lu\n",
            (unsigned long)expected_bytes, (unsigned long)actual_bytes);
        fclose(file);
        exit(EXIT_FAILURE);
    }

    t->weight_data = checked_calloc(expected_count, sizeof(float), "checkpoint weights");
    if (!read_exact(file, t->weight_data, payload_bytes)) {
        fclose(file);
        die("cannot read complete checkpoint payload");
    }

    if (fgetc(file) != EOF) {
        fclose(file);
        die("checkpoint contains unexpected trailing data");
    }

    if (fclose(file) != 0)
        die_errno("closing checkpoint", checkpoint_path);

    t->weight_count = expected_count;
    t->checkpoint_bytes = actual_bytes;
    map_weights(t);
    allocate_run_state(t);
}

static int compare_token_index(const void *a, const void *b)
{
    const TokenIndex *left = (const TokenIndex *)a;
    const TokenIndex *right = (const TokenIndex *)b;
    return strcmp(left->str, right->str);
}

static int token_lookup(const char *str, TokenIndex *sorted_vocab, int vocab_size)
{
    TokenIndex key;
    TokenIndex *result;

    key.str = (char *)str;
    key.id = 0;

    result = (TokenIndex *)bsearch(&key, sorted_vocab,
        (size_t)vocab_size, sizeof(TokenIndex), compare_token_index);

    return result != NULL ? result->id : -1;
}

static void prepare_sorted_vocab(Tokenizer *t)
{
    int i;

    if (t->sorted_vocab != NULL)
        return;

    t->sorted_vocab = checked_calloc((size_t)t->vocab_size,
        sizeof(TokenIndex), "sorted vocabulary");

    for (i = 0; i < t->vocab_size; i++) {
        t->sorted_vocab[i].str = t->vocab[i];
        t->sorted_vocab[i].id = i;
    }

    qsort(t->sorted_vocab, (size_t)t->vocab_size,
        sizeof(TokenIndex), compare_token_index);
}

static void free_tokenizer(Tokenizer *t)
{
    int i;

    if (t->vocab != NULL) {
        for (i = 0; i < t->vocab_size; i++)
            free(t->vocab[i]);
    }

    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
    memset(t, 0, sizeof(*t));
}

static void load_tokenizer(Tokenizer *t, const char *path, int vocab_size)
{
    FILE *file;
    int i;
    uint32_t max_length;
    size_t file_bytes;
    size_t accounted = 0;

    memset(t, 0, sizeof(*t));
    t->vocab_size = vocab_size;

    for (i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }

    file = fopen(path, "rb");
    if (file == NULL)
        die_errno("opening tokenizer", path);

    file_bytes = regular_file_size(file, path);
    if (file_bytes > MAX_TOKENIZER_BYTES) {
        fclose(file);
        die("tokenizer file exceeds safety limit");
    }

    if (!read_exact(file, &max_length, sizeof(max_length))) {
        fclose(file);
        die("cannot read tokenizer header");
    }
    accounted += sizeof(max_length);

    if (max_length == 0 || max_length > MAX_TOKEN_BYTES) {
        fclose(file);
        die("invalid tokenizer maximum token length");
    }

    t->max_token_length = max_length;
    t->vocab = checked_calloc((size_t)vocab_size, sizeof(char *), "vocabulary pointers");
    t->vocab_scores = checked_calloc((size_t)vocab_size, sizeof(float), "vocabulary scores");

    for (i = 0; i < vocab_size; i++) {
        float score = 0.0f;
        int32_t signed_length = 0;
        size_t length;
        size_t next_accounted;

        if (!read_exact(file, &score, sizeof(score)) ||
            !read_exact(file, &signed_length, sizeof(signed_length))) {
            fclose(file);
            free_tokenizer(t);
            die("truncated tokenizer entry header");
        }

        if (signed_length < 0) {
            fclose(file);
            free_tokenizer(t);
            die("negative tokenizer piece length");
        }

        length = (size_t)signed_length;
        if (length > (size_t)max_length || length > MAX_TOKEN_BYTES) {
            fclose(file);
            free_tokenizer(t);
            die("tokenizer piece exceeds declared maximum length");
        }

        if (!checked_add_size(accounted, sizeof(score) + sizeof(signed_length),
                &next_accounted) ||
            !checked_add_size(next_accounted, length, &accounted) ||
            accounted > file_bytes) {
            fclose(file);
            free_tokenizer(t);
            die("tokenizer length arithmetic overflow");
        }

        t->vocab[i] = checked_calloc(length + 1U, 1, "tokenizer piece");
        if (length != 0 && !read_exact(file, t->vocab[i], length)) {
            fclose(file);
            free_tokenizer(t);
            die("truncated tokenizer piece");
        }

        t->vocab[i][length] = '\0';
        t->vocab_scores[i] = score;
        t->allocated_bytes += length + 1U;
    }

    if (accounted != file_bytes || fgetc(file) != EOF) {
        fclose(file);
        free_tokenizer(t);
        die("tokenizer length mismatch or trailing data");
    }

    if (fclose(file) != 0) {
        free_tokenizer(t);
        die_errno("closing tokenizer", path);
    }
}

static int hex_value(unsigned char ch)
{
    if (ch >= '0' && ch <= '9')
        return ch - '0';
    if (ch >= 'a' && ch <= 'f')
        return 10 + ch - 'a';
    if (ch >= 'A' && ch <= 'F')
        return 10 + ch - 'A';
    return -1;
}

static char *decode_piece(Tokenizer *t, int previous_token, int token)
{
    char *piece;

    if (token < 0 || token >= t->vocab_size)
        die("decoder received out-of-range token");

    piece = t->vocab[token];

    if (previous_token == 1 && piece[0] == ' ')
        piece++;

    if (strlen(piece) == 6 &&
        piece[0] == '<' && piece[1] == '0' && piece[2] == 'x' &&
        piece[5] == '>') {
        int high = hex_value((unsigned char)piece[3]);
        int low = hex_value((unsigned char)piece[4]);
        if (high >= 0 && low >= 0)
            return (char *)t->byte_pieces + ((high << 4) | low) * 2;
    }

    return piece;
}

static void print_safe_piece(char *piece)
{
    unsigned char value;

    if (piece == NULL || piece[0] == '\0')
        return;

    if (piece[1] == '\0') {
        value = (unsigned char)piece[0];
        if (!(isprint((int)value) || isspace((int)value)))
            return;
    }

    fputs(piece, stdout);
}

static int encode_text(Tokenizer *t, const char *text, int bos, int eos,
    int **tokens_out, int *count_out)
{
    size_t text_length;
    size_t capacity;
    int *tokens;
    int count = 0;
    char *buffer;
    size_t buffer_capacity;
    size_t codepoint_length = 0;
    const unsigned char *cursor;

    if (text == NULL || tokens_out == NULL || count_out == NULL)
        return 0;

    prepare_sorted_vocab(t);

    text_length = strlen(text);
    if (!checked_add_size(text_length, 3U, &capacity) ||
        capacity > (size_t)INT_MAX)
        return 0;

    tokens = checked_calloc(capacity, sizeof(int), "encoded prompt");

    if (!checked_mul_size((size_t)t->max_token_length, 2U, &buffer_capacity) ||
        !checked_add_size(buffer_capacity, 3U, &buffer_capacity)) {
        free(tokens);
        return 0;
    }

    buffer = checked_calloc(buffer_capacity, 1, "token merge buffer");

    if (bos)
        tokens[count++] = 1;

    if (text[0] != '\0') {
        int dummy_prefix = token_lookup(" ", t->sorted_vocab, t->vocab_size);
        if (dummy_prefix < 0) {
            if (t->vocab_size > 35)
                dummy_prefix = 3 + 0x20;
            else {
                free(buffer);
                free(tokens);
                return 0;
            }
        }
        tokens[count++] = dummy_prefix;
    }

    cursor = (const unsigned char *)text;
    while (*cursor != '\0') {
        unsigned char current = *cursor;
        int id;
        size_t i;

        if ((current & 0xC0U) != 0x80U)
            codepoint_length = 0;

        if (codepoint_length >= 4U || codepoint_length + 1U >= buffer_capacity) {
            free(buffer);
            free(tokens);
            return 0;
        }

        buffer[codepoint_length++] = (char)current;
        buffer[codepoint_length] = '\0';

        if ((cursor[1] & 0xC0U) == 0x80U && codepoint_length < 4U) {
            cursor++;
            continue;
        }

        id = token_lookup(buffer, t->sorted_vocab, t->vocab_size);
        if (id >= 0) {
            tokens[count++] = id;
        } else {
            for (i = 0; i < codepoint_length; i++) {
                int byte_token = (unsigned char)buffer[i] + 3;
                if (byte_token >= t->vocab_size) {
                    free(buffer);
                    free(tokens);
                    return 0;
                }
                tokens[count++] = byte_token;
            }
        }

        codepoint_length = 0;
        cursor++;
    }

    for (;;) {
        float best_score = -FLT_MAX;
        int best_id = -1;
        int best_index = -1;
        int i;

        for (i = 0; i < count - 1; i++) {
            const char *left = t->vocab[tokens[i]];
            const char *right = t->vocab[tokens[i + 1]];
            size_t left_length = strlen(left);
            size_t right_length = strlen(right);
            size_t combined;
            int id;

            if (!checked_add_size(left_length, right_length, &combined) ||
                combined + 1U > buffer_capacity)
                continue;

            memcpy(buffer, left, left_length);
            memcpy(buffer + left_length, right, right_length);
            buffer[combined] = '\0';

            id = token_lookup(buffer, t->sorted_vocab, t->vocab_size);
            if (id >= 0 && t->vocab_scores[id] > best_score) {
                best_score = t->vocab_scores[id];
                best_id = id;
                best_index = i;
            }
        }

        if (best_index < 0)
            break;

        tokens[best_index] = best_id;
        for (i = best_index + 1; i < count - 1; i++)
            tokens[i] = tokens[i + 1];
        count--;
    }

    if (eos)
        tokens[count++] = 2;

    free(buffer);
    *tokens_out = tokens;
    *count_out = count;
    return 1;
}

static void rmsnorm(float *output, const float *input,
    const float *weight, int size)
{
    float sum_squares = 0.0f;
    float scale;
    int i;

    for (i = 0; i < size; i++)
        sum_squares += input[i] * input[i];

    sum_squares /= (float)size;
    scale = 1.0f / sqrtf(sum_squares + 1e-5f);

    for (i = 0; i < size; i++)
        output[i] = weight[i] * (scale * input[i]);
}

static void softmax(float *values, int size)
{
    float maximum;
    float sum = 0.0f;
    int i;

    maximum = values[0];
    for (i = 1; i < size; i++) {
        if (values[i] > maximum)
            maximum = values[i];
    }

    for (i = 0; i < size; i++) {
        values[i] = expf(values[i] - maximum);
        sum += values[i];
    }

    if (!(sum > 0.0f) || !isfinite(sum))
        die("softmax produced an invalid sum");

    for (i = 0; i < size; i++)
        values[i] /= sum;
}

static void matmul(float *output, const float *input,
    const float *weights, int input_size, int output_size)
{
    int row;

    for (row = 0; row < output_size; row++) {
        const float *weight_row = weights + (size_t)row * (size_t)input_size;
        float value = 0.0f;
        int column;

        for (column = 0; column < input_size; column++)
            value += weight_row[column] * input[column];

        output[row] = value;
    }
}

static float *forward(Transformer *t, int token, int position)
{
    Config *p = &t->config;
    TransformerWeights *w = &t->weights;
    RunState *s = &t->state;
    float *x = s->x;
    int dim = p->dim;
    int hidden = p->hidden_dim;
    int head_size = dim / p->n_heads;
    int kv_dim = (dim * p->n_kv_heads) / p->n_heads;
    int kv_multiplier = p->n_heads / p->n_kv_heads;
    int layer;

    if (token < 0 || token >= p->vocab_size)
        die("forward received out-of-range token");
    if (position < 0 || position >= p->seq_len)
        die("forward position exceeds context length");

    memcpy(x, w->token_embedding_table + (size_t)token * (size_t)dim,
        (size_t)dim * sizeof(float));

    if (g_forward_diag.enabled && g_forward_diag.file != NULL &&
        position < g_forward_diag.max_positions) {
        fprintf(g_forward_diag.file, "diag pos=%d input_token=%d\n", position, token);
    }
    diag_log_vector("embedding", position, -1, x, (size_t)dim);

    for (layer = 0; layer < p->n_layers; layer++) {
        size_t layer_dim = (size_t)layer * (size_t)dim;
        size_t layer_dim_dim = (size_t)layer * (size_t)dim * (size_t)dim;
        size_t layer_dim_kv = (size_t)layer * (size_t)dim * (size_t)kv_dim;
        size_t cache_layer = (size_t)layer * (size_t)p->seq_len * (size_t)kv_dim;
        float *key = s->key_cache + cache_layer +
            (size_t)position * (size_t)kv_dim;
        float *value = s->value_cache + cache_layer +
            (size_t)position * (size_t)kv_dim;
        int i;
        int head;

        rmsnorm(s->xb, x, w->rms_att_weight + layer_dim, dim);
        if (layer == 0)
            diag_log_vector("layer0_rmsnorm", position, layer, s->xb, (size_t)dim);

        matmul(s->q, s->xb, w->wq + layer_dim_dim, dim, dim);
        matmul(key, s->xb, w->wk + layer_dim_kv, dim, kv_dim);
        matmul(value, s->xb, w->wv + layer_dim_kv, dim, kv_dim);
        if (layer == 0) {
            diag_log_vector("layer0_q_pre_rope", position, layer, s->q, (size_t)dim);
            diag_log_vector("layer0_k_pre_rope", position, layer, key, (size_t)kv_dim);
            diag_log_vector("layer0_v", position, layer, value, (size_t)kv_dim);
        }

        for (i = 0; i < dim; i += 2) {
            int head_dimension = i % head_size;
            float frequency = 1.0f /
                powf(10000.0f, head_dimension / (float)head_size);
            float angle = position * frequency;
            float cosine = cosf(angle);
            float sine = sinf(angle);
            int rotations = i < kv_dim ? 2 : 1;
            int vector_index;

            for (vector_index = 0; vector_index < rotations; vector_index++) {
                float *vector = vector_index == 0 ? s->q : key;
                float first = vector[i];
                float second = vector[i + 1];
                vector[i] = first * cosine - second * sine;
                vector[i + 1] = first * sine + second * cosine;
            }
        }
        if (layer == 0) {
            diag_log_vector("layer0_q_post_rope", position, layer, s->q, (size_t)dim);
            diag_log_vector("layer0_k_post_rope", position, layer, key, (size_t)kv_dim);
        }

        for (head = 0; head < p->n_heads; head++) {
            float *query = s->q + (size_t)head * (size_t)head_size;
            float *attention = s->att +
                (size_t)head * (size_t)p->seq_len;
            float *head_output = s->xb +
                (size_t)head * (size_t)head_size;
            int time_index;

            for (time_index = 0; time_index <= position; time_index++) {
                const float *cached_key = s->key_cache + cache_layer +
                    (size_t)time_index * (size_t)kv_dim +
                    (size_t)(head / kv_multiplier) * (size_t)head_size;
                float score = 0.0f;

                for (i = 0; i < head_size; i++)
                    score += query[i] * cached_key[i];

                attention[time_index] =
                    score / sqrtf((float)head_size);
            }

            softmax(attention, position + 1);
            memset(head_output, 0, (size_t)head_size * sizeof(float));

            for (time_index = 0; time_index <= position; time_index++) {
                const float *cached_value = s->value_cache + cache_layer +
                    (size_t)time_index * (size_t)kv_dim +
                    (size_t)(head / kv_multiplier) * (size_t)head_size;
                float coefficient = attention[time_index];

                for (i = 0; i < head_size; i++)
                    head_output[i] += coefficient * cached_value[i];
            }
        }

        if (layer == 0)
            diag_log_vector("layer0_attention_out", position, layer, s->xb, (size_t)dim);

        matmul(s->xb2, s->xb, w->wo + layer_dim_dim, dim, dim);
        for (i = 0; i < dim; i++)
            x[i] += s->xb2[i];

        rmsnorm(s->xb, x, w->rms_ffn_weight + layer_dim, dim);

        matmul(s->hb, s->xb,
            w->w1 + (size_t)layer * (size_t)hidden * (size_t)dim,
            dim, hidden);
        matmul(s->hb2, s->xb,
            w->w3 + (size_t)layer * (size_t)hidden * (size_t)dim,
            dim, hidden);

        for (i = 0; i < hidden; i++) {
            float value1 = s->hb[i];
            value1 *= 1.0f / (1.0f + expf(-value1));
            s->hb[i] = value1 * s->hb2[i];
        }

        matmul(s->xb, s->hb,
            w->w2 + (size_t)layer * (size_t)dim * (size_t)hidden,
            hidden, dim);

        if (layer == 0)
            diag_log_vector("layer0_ffn_out", position, layer, s->xb, (size_t)dim);

        for (i = 0; i < dim; i++)
            x[i] += s->xb[i];
    }

    rmsnorm(x, x, w->rms_final_weight, dim);
    matmul(s->logits, x, w->wcls, dim, p->vocab_size);
    apply_adapter_correction(t, x, s->logits);

    diag_log_vector("final_logits", position, -1, s->logits,
        (size_t)p->vocab_size);
    diag_log_top_logits(position, s->logits, p->vocab_size);

    return s->logits;
}

static int sample_argmax(const float *values, int count)
{
    int best = 0;
    float best_value = values[0];
    int i;

    for (i = 1; i < count; i++) {
        if (values[i] > best_value) {
            best_value = values[i];
            best = i;
        }
    }

    return best;
}

static int compare_probabilities(const void *a, const void *b)
{
    const ProbIndex *left = (const ProbIndex *)a;
    const ProbIndex *right = (const ProbIndex *)b;

    if (left->prob > right->prob)
        return -1;
    if (left->prob < right->prob)
        return 1;
    return 0;
}

static uint32_t random_u32(uint64_t *state)
{
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (uint32_t)((*state * UINT64_C(0x2545F4914F6CDD1D)) >> 32);
}

static float random_f32(uint64_t *state)
{
    return (random_u32(state) >> 8) / 16777216.0f;
}

static void build_sampler(Sampler *sampler, int vocab_size,
    float temperature, float topp, uint64_t seed)
{
    memset(sampler, 0, sizeof(*sampler));
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = seed == 0 ? UINT64_C(1) : seed;
    sampler->probindex = checked_calloc((size_t)vocab_size,
        sizeof(ProbIndex), "top-p sampling buffer");
}

static void free_sampler(Sampler *sampler)
{
    free(sampler->probindex);
    memset(sampler, 0, sizeof(*sampler));
}

static int sample_token(Sampler *sampler, float *logits)
{
    int i;

    if (sampler->temperature == 0.0f)
        return sample_argmax(logits, sampler->vocab_size);

    for (i = 0; i < sampler->vocab_size; i++)
        logits[i] /= sampler->temperature;

    softmax(logits, sampler->vocab_size);

    if (sampler->topp <= 0.0f || sampler->topp >= 1.0f) {
        float coin = random_f32(&sampler->rng_state);
        float cumulative = 0.0f;

        for (i = 0; i < sampler->vocab_size; i++) {
            cumulative += logits[i];
            if (coin < cumulative)
                return i;
        }
        return sampler->vocab_size - 1;
    } else {
        float cutoff = (1.0f - sampler->topp) /
            (float)(sampler->vocab_size - 1);
        int candidate_count = 0;
        float cumulative = 0.0f;
        int last;
        float coin;
        float target;

        for (i = 0; i < sampler->vocab_size; i++) {
            if (logits[i] >= cutoff) {
                sampler->probindex[candidate_count].index = i;
                sampler->probindex[candidate_count].prob = logits[i];
                candidate_count++;
            }
        }

        if (candidate_count == 0)
            return sample_argmax(logits, sampler->vocab_size);

        qsort(sampler->probindex, (size_t)candidate_count,
            sizeof(ProbIndex), compare_probabilities);

        last = candidate_count - 1;
        for (i = 0; i < candidate_count; i++) {
            cumulative += sampler->probindex[i].prob;
            if (cumulative > sampler->topp) {
                last = i;
                break;
            }
        }

        coin = random_f32(&sampler->rng_state);
        target = coin * cumulative;
        cumulative = 0.0f;

        for (i = 0; i <= last; i++) {
            cumulative += sampler->probindex[i].prob;
            if (target < cumulative)
                return sampler->probindex[i].index;
        }

        return sampler->probindex[last].index;
    }
}

static double monotonic_seconds(void)
{
    struct timespec value;

    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0)
        die_errno("clock_gettime(CLOCK_MONOTONIC)", NULL);

    return (double)value.tv_sec + (double)value.tv_nsec / 1000000000.0;
}

static void trace_open(TraceWriter *trace, const char *path, int vocab_size)
{
    uint32_t version = TRACE_VERSION;
    uint32_t vocab = (uint32_t)vocab_size;
    uint32_t entries = 0;
    uint32_t reserved = 0;

    memset(trace, 0, sizeof(*trace));
    if (path == NULL)
        return;

    trace->file = fopen(path, "wb");
    if (trace->file == NULL)
        die_errno("creating trace", path);

    if (!write_exact(trace->file, TRACE_MAGIC, 4) ||
        !write_exact(trace->file, &version, sizeof(version)) ||
        !write_exact(trace->file, &vocab, sizeof(vocab)) ||
        !write_exact(trace->file, &entries, sizeof(entries)) ||
        !write_exact(trace->file, &reserved, sizeof(reserved)))
        die("cannot write trace header");

    trace->vocab_size = vocab;
}

static void trace_add(TraceWriter *trace, int position,
    int input_token, const float *logits)
{
    uint32_t pos;
    uint32_t token;
    size_t bytes;

    if (trace->file == NULL)
        return;

    pos = (uint32_t)position;
    token = (uint32_t)input_token;
    bytes = (size_t)trace->vocab_size * sizeof(float);

    if (!write_exact(trace->file, &pos, sizeof(pos)) ||
        !write_exact(trace->file, &token, sizeof(token)) ||
        !write_exact(trace->file, logits, bytes))
        die("cannot write trace record");

    trace->entries++;
}

static void trace_close(TraceWriter *trace)
{
    if (trace->file == NULL)
        return;

    if (fseeko(trace->file, (off_t)12, SEEK_SET) != 0 ||
        !write_exact(trace->file, &trace->entries, sizeof(trace->entries)))
        die("cannot finalize trace header");

    if (fclose(trace->file) != 0)
        die_errno("closing trace", NULL);

    trace->file = NULL;
}

static void print_model_info(const Transformer *t, const Tokenizer *tokenizer)
{
    const Config *p = &t->config;
    int head_size = p->dim / p->n_heads;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    size_t weight_bytes = t->weight_count * sizeof(float);
    size_t total = weight_bytes + t->run_state_bytes +
        tokenizer->allocated_bytes;

    printf("Model configuration\n");
    printf("  dim:             %d\n", p->dim);
    printf("  hidden_dim:      %d\n", p->hidden_dim);
    printf("  layers:          %d\n", p->n_layers);
    printf("  attention heads: %d\n", p->n_heads);
    printf("  KV heads:        %d\n", p->n_kv_heads);
    printf("  head size:       %d\n", head_size);
    printf("  KV dimension:    %d\n", kv_dim);
    printf("  vocabulary:      %d\n", p->vocab_size);
    printf("  context length:  %d\n", p->seq_len);
    printf("  shared output:   %s\n", p->shared_weights ? "yes" : "no");
    printf("  weight floats:   %lu\n", (unsigned long)t->weight_count);
    printf("  checkpoint:      %lu bytes\n", (unsigned long)t->checkpoint_bytes);
    printf("  run state:       %lu bytes\n", (unsigned long)t->run_state_bytes);
    printf("  tokenizer text:  %lu bytes\n",
        (unsigned long)tokenizer->allocated_bytes);
    printf("  approximate live allocation: %lu bytes\n",
        (unsigned long)total);
}

static void print_tokenization(Tokenizer *tokenizer,
    const char *prompt, const int *tokens, int token_count)
{
    int i;

    printf("Prompt: %s\n", prompt);
    printf("Token count: %d\n", token_count);
    printf("Token IDs:");

    for (i = 0; i < token_count; i++)
        printf(" %d", tokens[i]);
    printf("\n");

    printf("Decoded pieces:");
    for (i = 1; i < token_count; i++) {
        char *piece = decode_piece(tokenizer, tokens[i - 1], tokens[i]);
        printf(" [%s]", piece);
    }
    printf("\n");
}

static int generate(Transformer *transformer, Tokenizer *tokenizer,
    Sampler *sampler, const char *prompt, int steps,
    const char *trace_path, const char *token_trace_path, int verbose)
{
    int *prompt_tokens = NULL;
    int prompt_count = 0;
    int token;
    int next = 0;
    int position = 0;
    int generated = 0;
    double start = 0.0;
    double end = 0.0;
    TraceWriter trace;
    FILE *token_trace = NULL;

    if (prompt == NULL)
        prompt = "";

    if (!encode_text(tokenizer, prompt, 1, 0,
            &prompt_tokens, &prompt_count))
        die("cannot encode prompt");

    if (prompt_count < 1)
        die("prompt produced no tokens");

    if (prompt_count > transformer->config.seq_len) {
        free(prompt_tokens);
        die("prompt exceeds model context length");
    }

    if (steps <= 0 || steps > transformer->config.seq_len)
        steps = transformer->config.seq_len;

    if (steps < prompt_count)
        steps = prompt_count;

    clear_run_state(transformer);
    trace_open(&trace, trace_path, transformer->config.vocab_size);

    if (token_trace_path != NULL) {
        token_trace = fopen(token_trace_path, "wb");
        if (token_trace == NULL)
            die_errno("creating token trace", token_trace_path);
    }

    if (verbose)
        print_tokenization(tokenizer, prompt, prompt_tokens, prompt_count);

    token = prompt_tokens[0];

    while (position < steps) {
        float *logits = forward(transformer, token, position);
        trace_add(&trace, position, token, logits);

        if (position < prompt_count - 1) {
            next = prompt_tokens[position + 1];
        } else {
            if (generated == 0)
                start = monotonic_seconds();
            next = sample_token(sampler, logits);
            if (token_trace != NULL)
                fprintf(token_trace, "%d\n", next);
            generated++;
        }

        position++;

        /* llama2.c TinyStories models use BOS as the sequence delimiter. */
        if (next == 1)
            break;

        print_safe_piece(decode_piece(tokenizer, token, next));
        fflush(stdout);
        token = next;
    }

    if (generated > 0)
        end = monotonic_seconds();

    printf("\n");
    if (generated > 0 && end > start) {
        fprintf(stderr,
            "generated %d token(s) in %.6f seconds (%.3f token/s)\n",
            generated, end - start, generated / (end - start));
    }

    trace_close(&trace);
    if (token_trace != NULL && fclose(token_trace) != 0)
        die_errno("closing token trace", token_trace_path);
    free(prompt_tokens);
    return generated;
}

static unsigned long parse_unsigned_long(const char *text, const char *name)
{
    char *end = NULL;
    unsigned long value;

    errno = 0;
    value = strtoul(text, &end, 0);
    if (errno != 0 || end == text || *end != '\0') {
        fprintf(stderr, "error: invalid %s: %s\n", name, text);
        exit(EXIT_FAILURE);
    }
    return value;
}

static double parse_double_value(const char *text, const char *name)
{
    char *end = NULL;
    double value;

    errno = 0;
    value = strtod(text, &end);
    if (errno != 0 || end == text || *end != '\0') {
        fprintf(stderr, "error: invalid %s: %s\n", name, text);
        exit(EXIT_FAILURE);
    }
    return value;
}

static void usage(const char *program)
{
    fprintf(stderr,
        "Usage: %s <checkpoint.bin> [options]\n"
        "\n"
        "Options:\n"
        "  -z <tokenizer.bin> tokenizer path (default tokenizer.bin)\n"
        "  -i <prompt>        input prompt (default empty)\n"
        "  -n <steps>         total context steps (default 256, 0=max)\n"
        "  -t <temperature>   0 for greedy, otherwise >0 (default 1.0)\n"
        "  -p <top-p>         nucleus probability in [0,1] (default 0.9)\n"
        "  -s <seed>          deterministic RNG seed (default 1)\n"
        "  -r <trace.bin>     write all per-step logits to a binary trace\n"
        "  -T <tokens.txt>    write generated token IDs (one per line)\n"
        "  -a <adapter.bin>   apply adapter correction during inference\n"
        "  -y <scale>         adapter correction scale (default 1.0)\n"
        "  -D <diag.txt>      write compact forward diagnostics\n"
        "  -X <positions>     diagnostics for first N positions (default 4)\n"
        "  -c                 check checkpoint/tokenizer and exit\n"
        "  -v                 verbose model and tokenizer information\n"
        "  -h                 show this help\n"
        "\n"
        "Examples:\n"
        "  %s stories15M.bin -z tokenizer.bin -t 0 -n 128 \\\n"
        "      -i \"Once upon a time\"\n"
        "  %s stories15M.bin -z tokenizer.bin -t 0 -n 64 \\\n"
        "      -i \"Once upon a time\" -r minix.trace\n"
        "  %s stories15M.bin -z tokenizer.bin -t 0 -n 64 \\\n"
        "      -i \"Once upon a time\" -a stories15M.adapter.bin -y 1.0\n",
        program, program, program, program);
    exit(EXIT_FAILURE);
}

int main(int argc, char **argv)
{
    const char *checkpoint_path;
    const char *tokenizer_path = "tokenizer.bin";
    const char *prompt = "";
    const char *trace_path = NULL;
    const char *token_trace_path = NULL;
    const char *adapter_path = NULL;
    int steps = 256;
    float temperature = 1.0f;
    float topp = 0.9f;
    float adapter_scale = 1.0f;
    uint64_t seed = UINT64_C(1);
    int check_only = 0;
    int verbose = 0;
    const char *diag_path = NULL;
    int diag_positions = 4;
    int index;
    Transformer transformer;
    Tokenizer tokenizer;
    Sampler sampler;
    int *check_tokens = NULL;
    int check_count = 0;

    if (argc < 2)
        usage(argv[0]);

    checkpoint_path = argv[1];

    for (index = 2; index < argc; index++) {
        const char *option = argv[index];

        if (strcmp(option, "-h") == 0) {
            usage(argv[0]);
        } else if (strcmp(option, "-c") == 0) {
            check_only = 1;
        } else if (strcmp(option, "-v") == 0) {
            verbose = 1;
        } else {
            const char *value;

            if (index + 1 >= argc)
                usage(argv[0]);
            value = argv[++index];

            if (strcmp(option, "-z") == 0) {
                tokenizer_path = value;
            } else if (strcmp(option, "-i") == 0) {
                prompt = value;
            } else if (strcmp(option, "-r") == 0) {
                trace_path = value;
            } else if (strcmp(option, "-T") == 0) {
                token_trace_path = value;
            } else if (strcmp(option, "-a") == 0) {
                adapter_path = value;
            } else if (strcmp(option, "-n") == 0) {
                unsigned long parsed = parse_unsigned_long(value, "step count");
                if (parsed > (unsigned long)INT_MAX)
                    die("step count is too large");
                steps = (int)parsed;
            } else if (strcmp(option, "-s") == 0) {
                seed = (uint64_t)parse_unsigned_long(value, "seed");
                if (seed == 0)
                    seed = UINT64_C(1);
            } else if (strcmp(option, "-t") == 0) {
                double parsed = parse_double_value(value, "temperature");
                if (parsed < 0.0 || parsed > FLT_MAX)
                    die("temperature is outside supported range");
                temperature = (float)parsed;
            } else if (strcmp(option, "-p") == 0) {
                double parsed = parse_double_value(value, "top-p");
                if (parsed < 0.0 || parsed > 1.0)
                    die("top-p must be in [0,1]");
                topp = (float)parsed;
            } else if (strcmp(option, "-y") == 0) {
                double parsed = parse_double_value(value, "adapter scale");
                if (!isfinite(parsed) || parsed < -FLT_MAX || parsed > FLT_MAX)
                    die("adapter scale is outside supported range");
                adapter_scale = (float)parsed;
            } else if (strcmp(option, "-D") == 0) {
                diag_path = value;
            } else if (strcmp(option, "-X") == 0) {
                unsigned long parsed = parse_unsigned_long(value,
                    "diagnostic positions");
                if (parsed > (unsigned long)INT_MAX)
                    die("diagnostic positions are too large");
                diag_positions = (int)parsed;
            } else {
                usage(argv[0]);
            }
        }
    }

    if (diag_path != NULL) {
        if (minix_llama_set_diagnostic_output(diag_path, diag_positions) != 0)
            die_errno("opening diagnostic output", diag_path);
    }

    load_transformer(&transformer, checkpoint_path);
    load_tokenizer(&tokenizer, tokenizer_path,
        transformer.config.vocab_size);
    if (adapter_path != NULL &&
        !load_adapter_runtime(adapter_path, &transformer, adapter_scale))
        die("adapter initialization failed");

    if (verbose || check_only)
        print_model_info(&transformer, &tokenizer);

    if (check_only) {
        if (!encode_text(&tokenizer, prompt, 1, 0,
                &check_tokens, &check_count))
            die("tokenizer check failed");
        print_tokenization(&tokenizer, prompt, check_tokens, check_count);
        free(check_tokens);
        free_tokenizer(&tokenizer);
        free_transformer(&transformer);
        unload_adapter_runtime();
        printf("checkpoint and tokenizer validation: PASS\n");
        return EXIT_SUCCESS;
    }

    build_sampler(&sampler, transformer.config.vocab_size,
        temperature, topp, seed);

    generate(&transformer, &tokenizer, &sampler,
        prompt, steps, trace_path, token_trace_path, verbose);

    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    unload_adapter_runtime();
    minix_llama_clear_diagnostic_output();
    return EXIT_SUCCESS;
}
