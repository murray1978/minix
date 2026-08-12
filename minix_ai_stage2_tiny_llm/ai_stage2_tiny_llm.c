#define _POSIX_C_SOURCE 200112L
#define _NETBSD_SOURCE 1

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <time.h>

#define TEST_PASS 0
#define TEST_FAIL 1
#define TEST_INFO 2

#define MODEL_MAGIC UINT32_C(0x4d584c31)
#define REF_MAGIC UINT32_C(0x4d585231)
#define FORMAT_VERSION UINT32_C(1)
#define ENDIAN_TAG UINT32_C(0x01020304)
#define HEADER_WORDS 16U
#define HEADER_BYTES (HEADER_WORDS * sizeof(uint32_t))
#define FNV_OFFSET UINT32_C(2166136261)
#define FNV_PRIME UINT32_C(16777619)

#define DEFAULT_MODEL_PATH "tiny_minix_llm.bin"
#define DEFAULT_REFERENCE_PATH "tiny_minix_llm.ref"
#define DEFAULT_ITERATIONS 100U
#define DEFAULT_BENCH_TOKENS 4096U
#define MAX_REASONABLE_DIMENSION 65536U

#define ARRAY_LEN(a) (sizeof(a) / sizeof((a)[0]))

struct test_context {
    const char *model_path;
    const char *reference_path;
    unsigned int iterations;
    unsigned int bench_tokens;
    int verbose;
};

struct test_case {
    int number;
    const char *name;
    int (*run)(const struct test_context *ctx);
};

struct model_config {
    uint32_t vocab_size;
    uint32_t context_length;
    uint32_t d_model;
    uint32_t n_heads;
    uint32_t n_layers;
    uint32_t d_ff;
    uint32_t tie_embeddings;
    uint32_t weight_count;
    uint32_t weight_checksum;
};

struct layer_weights {
    float *rms_att;
    float *wq;
    float *wk;
    float *wv;
    float *wo;
    float *rms_ffn;
    float *w1;
    float *w3;
    float *w2;
};

struct tiny_model {
    struct model_config config;
    float *weights;
    float *token_embedding;
    struct layer_weights *layers;
    float *final_rms;
};

struct run_state {
    float *x;
    float *xb;
    float *xb2;
    float *q;
    float *k;
    float *v;
    float *attention_output;
    float *attention_scores;
    float *hidden1;
    float *hidden3;
    float *logits;
    float *key_cache;
    float *value_cache;
};

struct reference_trace {
    uint32_t vocab_size;
    uint32_t prompt_length;
    uint32_t generation_length;
    uint32_t step_count;
    uint32_t payload_checksum;
    uint32_t *prompt;
    uint32_t *generated;
    float *logits;
};

union allocation_header {
    struct {
        size_t bytes;
    } meta;
    long double align_long_double;
    void *align_pointer;
};

static int g_passed;
static int g_failed;
static int g_info;
static size_t g_live_allocations;
static size_t g_live_bytes;
static size_t g_peak_bytes;

static void print_rule(void)
{
    puts("------------------------------------------------------------");
}

static void report_result(int number, const char *name, int result)
{
    const char *label;

    if (result == TEST_PASS) {
        label = "PASS";
        g_passed++;
    } else if (result == TEST_INFO) {
        label = "INFO";
        g_info++;
    } else {
        label = "FAIL";
        g_failed++;
    }

    printf("[%s] Test %d: %s\n", label, number, name);
}

static int checked_multiply_size(size_t left, size_t right, size_t *result)
{
    if (result == NULL)
        return -1;

    if (left != 0U && right > ((size_t)-1) / left)
        return -1;

    *result = left * right;
    return 0;
}

static int checked_add_size(size_t left, size_t right, size_t *result)
{
    if (result == NULL)
        return -1;

    if (right > ((size_t)-1) - left)
        return -1;

    *result = left + right;
    return 0;
}

static void *tracked_alloc(size_t count, size_t element_size, int clear)
{
    union allocation_header *header;
    size_t payload_bytes;
    size_t total_bytes;

    if (checked_multiply_size(count, element_size, &payload_bytes) != 0 ||
        checked_add_size(sizeof(*header), payload_bytes, &total_bytes) != 0) {
        errno = ENOMEM;
        return NULL;
    }

    header = (union allocation_header *)malloc(total_bytes);
    if (header == NULL)
        return NULL;

    header->meta.bytes = payload_bytes;
    g_live_allocations++;
    g_live_bytes += payload_bytes;
    if (g_live_bytes > g_peak_bytes)
        g_peak_bytes = g_live_bytes;

    if (clear && payload_bytes != 0U)
        memset((void *)(header + 1), 0, payload_bytes);

    return (void *)(header + 1);
}

static void tracked_free(void *pointer)
{
    union allocation_header *header;

    if (pointer == NULL)
        return;

    header = ((union allocation_header *)pointer) - 1;
    if (g_live_allocations > 0U)
        g_live_allocations--;
    if (g_live_bytes >= header->meta.bytes)
        g_live_bytes -= header->meta.bytes;
    else
        g_live_bytes = 0U;

    free(header);
}

static uint32_t fnv1a_update(uint32_t hash, const void *data, size_t size)
{
    const unsigned char *bytes;
    size_t index;

    bytes = (const unsigned char *)data;
    for (index = 0U; index < size; index++) {
        hash ^= (uint32_t)bytes[index];
        hash *= FNV_PRIME;
    }

    return hash;
}

static int read_exact(FILE *file, void *buffer, size_t bytes)
{
    unsigned char *output;
    size_t done;

    output = (unsigned char *)buffer;
    done = 0U;
    while (done < bytes) {
        size_t received;

        received = fread(output + done, 1U, bytes - done, file);
        if (received == 0U)
            return -1;
        done += received;
    }

    return 0;
}

static int file_length(FILE *file, off_t *length)
{
    off_t current;
    off_t end;

    if (file == NULL || length == NULL)
        return -1;

    current = ftello(file);
    if (current < 0)
        return -1;
    if (fseeko(file, 0, SEEK_END) != 0)
        return -1;
    end = ftello(file);
    if (end < 0)
        return -1;
    if (fseeko(file, current, SEEK_SET) != 0)
        return -1;

    *length = end;
    return 0;
}

static int calculate_expected_weight_count(const struct model_config *config,
    size_t *result)
{
    size_t total;
    size_t value;
    size_t per_layer;

    if (config == NULL || result == NULL)
        return -1;

    total = 0U;

    if (checked_multiply_size((size_t)config->vocab_size,
        (size_t)config->d_model, &value) != 0 ||
        checked_add_size(total, value, &total) != 0)
        return -1;

    per_layer = 0U;
    if (checked_add_size(per_layer, (size_t)config->d_model,
        &per_layer) != 0)
        return -1;

    if (checked_multiply_size((size_t)config->d_model,
        (size_t)config->d_model, &value) != 0)
        return -1;
    if (checked_multiply_size(value, 4U, &value) != 0 ||
        checked_add_size(per_layer, value, &per_layer) != 0)
        return -1;

    if (checked_add_size(per_layer, (size_t)config->d_model,
        &per_layer) != 0)
        return -1;

    if (checked_multiply_size((size_t)config->d_ff,
        (size_t)config->d_model, &value) != 0)
        return -1;
    if (checked_multiply_size(value, 2U, &value) != 0 ||
        checked_add_size(per_layer, value, &per_layer) != 0)
        return -1;

    if (checked_multiply_size((size_t)config->d_model,
        (size_t)config->d_ff, &value) != 0 ||
        checked_add_size(per_layer, value, &per_layer) != 0)
        return -1;

    if (checked_multiply_size(per_layer, (size_t)config->n_layers,
        &value) != 0 || checked_add_size(total, value, &total) != 0)
        return -1;

    if (checked_add_size(total, (size_t)config->d_model, &total) != 0)
        return -1;

    *result = total;
    return 0;
}

static int validate_config(const struct model_config *config,
    char *error, size_t error_size)
{
    size_t expected_count;

    if (config == NULL)
        return -1;

    if (config->vocab_size == 0U ||
        config->context_length == 0U ||
        config->d_model == 0U ||
        config->n_heads == 0U ||
        config->n_layers == 0U ||
        config->d_ff == 0U) {
        snprintf(error, error_size, "zero model dimension");
        return -1;
    }

    if (config->vocab_size > MAX_REASONABLE_DIMENSION ||
        config->context_length > MAX_REASONABLE_DIMENSION ||
        config->d_model > MAX_REASONABLE_DIMENSION ||
        config->n_heads > MAX_REASONABLE_DIMENSION ||
        config->n_layers > MAX_REASONABLE_DIMENSION ||
        config->d_ff > MAX_REASONABLE_DIMENSION) {
        snprintf(error, error_size, "unreasonable model dimension");
        return -1;
    }

    if (config->d_model % config->n_heads != 0U) {
        snprintf(error, error_size, "d_model is not divisible by n_heads");
        return -1;
    }

    if (((config->d_model / config->n_heads) & 1U) != 0U) {
        snprintf(error, error_size, "head size must be even for RoPE");
        return -1;
    }

    if (config->tie_embeddings != 1U) {
        snprintf(error, error_size, "only tied embeddings are supported");
        return -1;
    }

    if (calculate_expected_weight_count(config, &expected_count) != 0 ||
        expected_count > UINT32_MAX ||
        (uint32_t)expected_count != config->weight_count) {
        snprintf(error, error_size, "weight count mismatch");
        return -1;
    }

    return 0;
}

static void unload_model(struct tiny_model *model)
{
    if (model == NULL)
        return;

    tracked_free(model->layers);
    tracked_free(model->weights);
    memset(model, 0, sizeof(*model));
}

static int map_model_weights(struct tiny_model *model)
{
    float *cursor;
    uint32_t layer_index;
    size_t square;
    size_t ffn_input;
    size_t ffn_output;

    if (model == NULL || model->weights == NULL)
        return -1;

    square = (size_t)model->config.d_model * model->config.d_model;
    ffn_input = (size_t)model->config.d_ff * model->config.d_model;
    ffn_output = (size_t)model->config.d_model * model->config.d_ff;

    model->layers = (struct layer_weights *)tracked_alloc(
        model->config.n_layers, sizeof(*model->layers), 1);
    if (model->layers == NULL)
        return -1;

    cursor = model->weights;
    model->token_embedding = cursor;
    cursor += (size_t)model->config.vocab_size * model->config.d_model;

    for (layer_index = 0U; layer_index < model->config.n_layers;
        layer_index++) {
        struct layer_weights *layer;

        layer = &model->layers[layer_index];
        layer->rms_att = cursor;
        cursor += model->config.d_model;
        layer->wq = cursor;
        cursor += square;
        layer->wk = cursor;
        cursor += square;
        layer->wv = cursor;
        cursor += square;
        layer->wo = cursor;
        cursor += square;
        layer->rms_ffn = cursor;
        cursor += model->config.d_model;
        layer->w1 = cursor;
        cursor += ffn_input;
        layer->w3 = cursor;
        cursor += ffn_input;
        layer->w2 = cursor;
        cursor += ffn_output;
    }

    model->final_rms = cursor;
    cursor += model->config.d_model;

    if ((size_t)(cursor - model->weights) != model->config.weight_count)
        return -1;

    return 0;
}

static int load_model(const char *path, struct tiny_model *model,
    char *error, size_t error_size)
{
    FILE *file;
    uint32_t header[HEADER_WORDS];
    off_t length;
    size_t payload_bytes;
    size_t expected_bytes;
    uint32_t checksum;

    if (path == NULL || model == NULL)
        return -1;

    memset(model, 0, sizeof(*model));
    file = fopen(path, "rb");
    if (file == NULL) {
        snprintf(error, error_size, "open failed: %s", strerror(errno));
        return -1;
    }

    if (file_length(file, &length) != 0 || fseeko(file, 0, SEEK_SET) != 0) {
        snprintf(error, error_size, "could not determine file length");
        fclose(file);
        return -1;
    }

    if (read_exact(file, header, sizeof(header)) != 0) {
        snprintf(error, error_size, "truncated header");
        fclose(file);
        return -1;
    }

    if (header[0] != MODEL_MAGIC || header[1] != FORMAT_VERSION ||
        header[2] != ENDIAN_TAG) {
        snprintf(error, error_size, "invalid magic, version, or endian tag");
        fclose(file);
        return -1;
    }

    model->config.vocab_size = header[3];
    model->config.context_length = header[4];
    model->config.d_model = header[5];
    model->config.n_heads = header[6];
    model->config.n_layers = header[7];
    model->config.d_ff = header[8];
    model->config.tie_embeddings = header[9];
    model->config.weight_count = header[10];
    model->config.weight_checksum = header[11];

    if (validate_config(&model->config, error, error_size) != 0) {
        fclose(file);
        return -1;
    }

    if (checked_multiply_size(model->config.weight_count, sizeof(float),
        &payload_bytes) != 0 ||
        checked_add_size(HEADER_BYTES, payload_bytes, &expected_bytes) != 0 ||
        (off_t)expected_bytes != length) {
        snprintf(error, error_size, "checkpoint length mismatch");
        fclose(file);
        return -1;
    }

    model->weights = (float *)tracked_alloc(model->config.weight_count,
        sizeof(float), 0);
    if (model->weights == NULL) {
        snprintf(error, error_size, "weight allocation failed");
        fclose(file);
        return -1;
    }

    if (read_exact(file, model->weights, payload_bytes) != 0) {
        snprintf(error, error_size, "truncated weight payload");
        fclose(file);
        unload_model(model);
        return -1;
    }
    fclose(file);

    checksum = fnv1a_update(FNV_OFFSET, model->weights, payload_bytes);
    if (checksum != model->config.weight_checksum) {
        snprintf(error, error_size, "weight checksum mismatch");
        unload_model(model);
        return -1;
    }

    if (map_model_weights(model) != 0) {
        snprintf(error, error_size, "internal weight mapping failure");
        unload_model(model);
        return -1;
    }

    return 0;
}

static void unload_reference(struct reference_trace *reference)
{
    if (reference == NULL)
        return;

    tracked_free(reference->logits);
    tracked_free(reference->generated);
    tracked_free(reference->prompt);
    memset(reference, 0, sizeof(*reference));
}

static int load_reference(const char *path, struct reference_trace *reference,
    char *error, size_t error_size)
{
    FILE *file;
    uint32_t header[HEADER_WORDS];
    off_t length;
    size_t prompt_bytes;
    size_t generated_bytes;
    size_t logit_count;
    size_t logits_bytes;
    size_t payload_bytes;
    size_t expected_bytes;
    uint32_t checksum;

    memset(reference, 0, sizeof(*reference));
    file = fopen(path, "rb");
    if (file == NULL) {
        snprintf(error, error_size, "open failed: %s", strerror(errno));
        return -1;
    }

    if (file_length(file, &length) != 0 || fseeko(file, 0, SEEK_SET) != 0 ||
        read_exact(file, header, sizeof(header)) != 0) {
        snprintf(error, error_size, "invalid or truncated reference header");
        fclose(file);
        return -1;
    }

    if (header[0] != REF_MAGIC || header[1] != FORMAT_VERSION) {
        snprintf(error, error_size, "invalid reference magic or version");
        fclose(file);
        return -1;
    }

    reference->vocab_size = header[2];
    reference->prompt_length = header[3];
    reference->generation_length = header[4];
    reference->step_count = header[5];
    reference->payload_checksum = header[6];

    if (reference->vocab_size == 0U || reference->prompt_length == 0U ||
        reference->step_count != reference->prompt_length +
            reference->generation_length) {
        snprintf(error, error_size, "invalid reference dimensions");
        fclose(file);
        return -1;
    }

    if (checked_multiply_size(reference->prompt_length, sizeof(uint32_t),
        &prompt_bytes) != 0 ||
        checked_multiply_size(reference->generation_length, sizeof(uint32_t),
        &generated_bytes) != 0 ||
        checked_multiply_size(reference->step_count, reference->vocab_size,
        &logit_count) != 0 ||
        checked_multiply_size(logit_count, sizeof(float), &logits_bytes) != 0 ||
        checked_add_size(prompt_bytes, generated_bytes, &payload_bytes) != 0 ||
        checked_add_size(payload_bytes, logits_bytes, &payload_bytes) != 0 ||
        checked_add_size(HEADER_BYTES, payload_bytes, &expected_bytes) != 0 ||
        (off_t)expected_bytes != length) {
        snprintf(error, error_size, "reference length overflow or mismatch");
        fclose(file);
        return -1;
    }

    reference->prompt = (uint32_t *)tracked_alloc(reference->prompt_length,
        sizeof(uint32_t), 0);
    reference->generated = (uint32_t *)tracked_alloc(
        reference->generation_length, sizeof(uint32_t), 0);
    reference->logits = (float *)tracked_alloc(logit_count, sizeof(float), 0);

    if (reference->prompt == NULL ||
        (reference->generation_length != 0U && reference->generated == NULL) ||
        reference->logits == NULL) {
        snprintf(error, error_size, "reference allocation failed");
        fclose(file);
        unload_reference(reference);
        return -1;
    }

    if (read_exact(file, reference->prompt, prompt_bytes) != 0 ||
        read_exact(file, reference->generated, generated_bytes) != 0 ||
        read_exact(file, reference->logits, logits_bytes) != 0) {
        snprintf(error, error_size, "truncated reference payload");
        fclose(file);
        unload_reference(reference);
        return -1;
    }
    fclose(file);

    checksum = FNV_OFFSET;
    checksum = fnv1a_update(checksum, reference->prompt, prompt_bytes);
    checksum = fnv1a_update(checksum, reference->generated, generated_bytes);
    checksum = fnv1a_update(checksum, reference->logits, logits_bytes);
    if (checksum != reference->payload_checksum) {
        snprintf(error, error_size, "reference checksum mismatch");
        unload_reference(reference);
        return -1;
    }

    return 0;
}

static void free_run_state(struct run_state *state)
{
    if (state == NULL)
        return;

    tracked_free(state->value_cache);
    tracked_free(state->key_cache);
    tracked_free(state->logits);
    tracked_free(state->hidden3);
    tracked_free(state->hidden1);
    tracked_free(state->attention_scores);
    tracked_free(state->attention_output);
    tracked_free(state->v);
    tracked_free(state->k);
    tracked_free(state->q);
    tracked_free(state->xb2);
    tracked_free(state->xb);
    tracked_free(state->x);
    memset(state, 0, sizeof(*state));
}

static int allocate_run_state(const struct tiny_model *model,
    struct run_state *state)
{
    size_t cache_count;

    memset(state, 0, sizeof(*state));
    if (checked_multiply_size(model->config.n_layers,
        model->config.context_length, &cache_count) != 0 ||
        checked_multiply_size(cache_count, model->config.d_model,
        &cache_count) != 0)
        return -1;

    state->x = (float *)tracked_alloc(model->config.d_model, sizeof(float), 0);
    state->xb = (float *)tracked_alloc(model->config.d_model, sizeof(float), 0);
    state->xb2 = (float *)tracked_alloc(model->config.d_model, sizeof(float), 0);
    state->q = (float *)tracked_alloc(model->config.d_model, sizeof(float), 0);
    state->k = (float *)tracked_alloc(model->config.d_model, sizeof(float), 0);
    state->v = (float *)tracked_alloc(model->config.d_model, sizeof(float), 0);
    state->attention_output = (float *)tracked_alloc(model->config.d_model,
        sizeof(float), 0);
    state->attention_scores = (float *)tracked_alloc(
        model->config.context_length, sizeof(float), 0);
    state->hidden1 = (float *)tracked_alloc(model->config.d_ff,
        sizeof(float), 0);
    state->hidden3 = (float *)tracked_alloc(model->config.d_ff,
        sizeof(float), 0);
    state->logits = (float *)tracked_alloc(model->config.vocab_size,
        sizeof(float), 0);
    state->key_cache = (float *)tracked_alloc(cache_count, sizeof(float), 1);
    state->value_cache = (float *)tracked_alloc(cache_count, sizeof(float), 1);

    if (state->x == NULL || state->xb == NULL || state->xb2 == NULL ||
        state->q == NULL || state->k == NULL || state->v == NULL ||
        state->attention_output == NULL || state->attention_scores == NULL ||
        state->hidden1 == NULL || state->hidden3 == NULL ||
        state->logits == NULL || state->key_cache == NULL ||
        state->value_cache == NULL) {
        free_run_state(state);
        return -1;
    }

    return 0;
}

static void reset_run_state(const struct tiny_model *model,
    struct run_state *state)
{
    size_t cache_count;

    cache_count = (size_t)model->config.n_layers *
        model->config.context_length * model->config.d_model;
    memset(state->key_cache, 0, cache_count * sizeof(float));
    memset(state->value_cache, 0, cache_count * sizeof(float));
}

static void matvec(float *output, const float *weight, const float *input,
    uint32_t output_count, uint32_t input_count)
{
    uint32_t row;
    uint32_t column;

    for (row = 0U; row < output_count; row++) {
        float total;

        total = 0.0f;
        for (column = 0U; column < input_count; column++)
            total += weight[(size_t)row * input_count + column] * input[column];
        output[row] = total;
    }
}

static void rmsnorm(float *output, const float *input, const float *weight,
    uint32_t count)
{
    float mean_square;
    float scale;
    uint32_t index;

    mean_square = 0.0f;
    for (index = 0U; index < count; index++)
        mean_square += input[index] * input[index];
    mean_square /= (float)count;
    scale = 1.0f / sqrtf(mean_square + 1.0e-5f);

    for (index = 0U; index < count; index++)
        output[index] = weight[index] * input[index] * scale;
}

static int stable_softmax(float *values, uint32_t count)
{
    float maximum;
    float sum;
    uint32_t index;

    if (values == NULL || count == 0U)
        return -1;

    maximum = values[0];
    for (index = 1U; index < count; index++) {
        if (values[index] > maximum)
            maximum = values[index];
    }

    sum = 0.0f;
    for (index = 0U; index < count; index++) {
        values[index] = expf(values[index] - maximum);
        sum += values[index];
    }

    if (!(sum > 0.0f) || !isfinite(sum))
        return -1;

    for (index = 0U; index < count; index++)
        values[index] /= sum;

    return 0;
}

static float silu(float value)
{
    return value / (1.0f + expf(-value));
}

static void apply_rope(float *q, float *k, uint32_t position,
    uint32_t n_heads, uint32_t head_size)
{
    uint32_t head;

    for (head = 0U; head < n_heads; head++) {
        uint32_t base;
        uint32_t index;

        base = head * head_size;
        for (index = 0U; index < head_size; index += 2U) {
            float frequency;
            float angle;
            float cosine;
            float sine;
            float q0;
            float q1;
            float k0;
            float k1;

            frequency = powf(10000.0f,
                -(float)index / (float)head_size);
            angle = (float)position * frequency;
            cosine = cosf(angle);
            sine = sinf(angle);
            q0 = q[base + index];
            q1 = q[base + index + 1U];
            k0 = k[base + index];
            k1 = k[base + index + 1U];
            q[base + index] = q0 * cosine - q1 * sine;
            q[base + index + 1U] = q0 * sine + q1 * cosine;
            k[base + index] = k0 * cosine - k1 * sine;
            k[base + index + 1U] = k0 * sine + k1 * cosine;
        }
    }
}

static float *forward_token(const struct tiny_model *model,
    struct run_state *state, uint32_t token, uint32_t position)
{
    uint32_t d_model;
    uint32_t head_size;
    uint32_t layer_index;
    uint32_t index;

    if (token >= model->config.vocab_size ||
        position >= model->config.context_length)
        return NULL;

    d_model = model->config.d_model;
    head_size = d_model / model->config.n_heads;
    memcpy(state->x,
        model->token_embedding + (size_t)token * d_model,
        d_model * sizeof(float));

    for (layer_index = 0U; layer_index < model->config.n_layers;
        layer_index++) {
        const struct layer_weights *layer;
        size_t cache_base;
        uint32_t head;

        layer = &model->layers[layer_index];
        rmsnorm(state->xb, state->x, layer->rms_att, d_model);
        matvec(state->q, layer->wq, state->xb, d_model, d_model);
        matvec(state->k, layer->wk, state->xb, d_model, d_model);
        matvec(state->v, layer->wv, state->xb, d_model, d_model);
        apply_rope(state->q, state->k, position,
            model->config.n_heads, head_size);

        cache_base = ((size_t)layer_index * model->config.context_length +
            position) * d_model;
        memcpy(state->key_cache + cache_base, state->k,
            d_model * sizeof(float));
        memcpy(state->value_cache + cache_base, state->v,
            d_model * sizeof(float));
        memset(state->attention_output, 0, d_model * sizeof(float));

        for (head = 0U; head < model->config.n_heads; head++) {
            uint32_t head_base;
            uint32_t timestep;

            head_base = head * head_size;
            for (timestep = 0U; timestep <= position; timestep++) {
                float total;
                size_t key_base;
                uint32_t element;

                total = 0.0f;
                key_base = ((size_t)layer_index *
                    model->config.context_length + timestep) * d_model;
                for (element = 0U; element < head_size; element++) {
                    total += state->q[head_base + element] *
                        state->key_cache[key_base + head_base + element];
                }
                state->attention_scores[timestep] = total /
                    sqrtf((float)head_size);
            }

            if (stable_softmax(state->attention_scores,
                position + 1U) != 0)
                return NULL;

            for (index = 0U; index < head_size; index++) {
                float total;

                total = 0.0f;
                for (timestep = 0U; timestep <= position; timestep++) {
                    size_t value_base;

                    value_base = ((size_t)layer_index *
                        model->config.context_length + timestep) * d_model;
                    total += state->attention_scores[timestep] *
                        state->value_cache[value_base + head_base + index];
                }
                state->attention_output[head_base + index] = total;
            }
        }

        matvec(state->xb2, layer->wo, state->attention_output,
            d_model, d_model);
        for (index = 0U; index < d_model; index++)
            state->x[index] += state->xb2[index];

        rmsnorm(state->xb, state->x, layer->rms_ffn, d_model);
        matvec(state->hidden1, layer->w1, state->xb,
            model->config.d_ff, d_model);
        matvec(state->hidden3, layer->w3, state->xb,
            model->config.d_ff, d_model);
        for (index = 0U; index < model->config.d_ff; index++)
            state->hidden1[index] = silu(state->hidden1[index]) *
                state->hidden3[index];
        matvec(state->xb2, layer->w2, state->hidden1,
            d_model, model->config.d_ff);
        for (index = 0U; index < d_model; index++)
            state->x[index] += state->xb2[index];
    }

    rmsnorm(state->xb, state->x, model->final_rms, d_model);
    for (index = 0U; index < model->config.vocab_size; index++) {
        float total;
        uint32_t element;
        const float *embedding;

        total = 0.0f;
        embedding = model->token_embedding + (size_t)index * d_model;
        for (element = 0U; element < d_model; element++)
            total += embedding[element] * state->xb[element];
        state->logits[index] = total;
    }

    return state->logits;
}

static uint32_t argmax(const float *values, uint32_t count)
{
    uint32_t best;
    uint32_t index;

    best = 0U;
    for (index = 1U; index < count; index++) {
        if (values[index] > values[best])
            best = index;
    }
    return best;
}

static float maximum_absolute_error(const float *left, const float *right,
    size_t count)
{
    float maximum;
    size_t index;

    maximum = 0.0f;
    for (index = 0U; index < count; index++) {
        float difference;

        difference = fabsf(left[index] - right[index]);
        if (difference > maximum)
            maximum = difference;
    }
    return maximum;
}

static int load_model_and_reference(const struct test_context *ctx,
    struct tiny_model *model, struct reference_trace *reference,
    char *error, size_t error_size)
{
    if (load_model(ctx->model_path, model, error, error_size) != 0)
        return -1;
    if (load_reference(ctx->reference_path, reference, error,
        error_size) != 0) {
        unload_model(model);
        return -1;
    }

    if (reference->vocab_size != model->config.vocab_size ||
        reference->step_count > model->config.context_length) {
        snprintf(error, error_size,
            "reference dimensions do not match checkpoint");
        unload_reference(reference);
        unload_model(model);
        return -1;
    }

    return 0;
}

static int run_teacher_forced(const struct tiny_model *model,
    struct run_state *state, const struct reference_trace *reference,
    float *maximum_error)
{
    uint32_t step;
    float max_error;

    reset_run_state(model, state);
    max_error = 0.0f;
    for (step = 0U; step < reference->step_count; step++) {
        uint32_t token;
        float *logits;
        float error;

        if (step < reference->prompt_length)
            token = reference->prompt[step];
        else
            token = reference->generated[step - reference->prompt_length];

        logits = forward_token(model, state, token, step);
        if (logits == NULL)
            return -1;
        error = maximum_absolute_error(logits,
            reference->logits + (size_t)step * reference->vocab_size,
            reference->vocab_size);
        if (error > max_error)
            max_error = error;
    }

    *maximum_error = max_error;
    return 0;
}

static int test_checkpoint_validation(const struct test_context *ctx)
{
    struct tiny_model model;
    char error[160];

    puts("\nTest 21: checkpoint header, size and checksum validation");
    print_rule();

    if (load_model(ctx->model_path, &model, error, sizeof(error)) != 0) {
        printf("load failed: %s\n", error);
        return TEST_FAIL;
    }

    printf("vocab=%u context=%u d_model=%u heads=%u layers=%u d_ff=%u\n",
        model.config.vocab_size, model.config.context_length,
        model.config.d_model, model.config.n_heads,
        model.config.n_layers, model.config.d_ff);
    printf("parameters=%u FP32 bytes=%lu checksum=0x%08lx\n",
        model.config.weight_count,
        (unsigned long)((size_t)model.config.weight_count * sizeof(float)),
        (unsigned long)model.config.weight_checksum);
    printf("tracked live bytes while loaded=%lu\n",
        (unsigned long)g_live_bytes);

    unload_model(&model);
    if (g_live_allocations != 0U || g_live_bytes != 0U) {
        printf("allocation accounting did not return to zero\n");
        return TEST_FAIL;
    }
    return TEST_PASS;
}

static int test_reference_logits(const struct test_context *ctx)
{
    struct tiny_model model;
    struct reference_trace reference;
    struct run_state state;
    char error[160];
    float max_error;
    int result;

    puts("\nTest 22: full multi-layer forward pass against reference logits");
    print_rule();
    result = TEST_FAIL;

    if (load_model_and_reference(ctx, &model, &reference,
        error, sizeof(error)) != 0) {
        printf("load failed: %s\n", error);
        return TEST_FAIL;
    }
    if (allocate_run_state(&model, &state) != 0) {
        puts("run-state allocation failed");
        unload_reference(&reference);
        unload_model(&model);
        return TEST_FAIL;
    }

    if (run_teacher_forced(&model, &state, &reference, &max_error) == 0) {
        printf("compared %u steps x %u logits\n",
            reference.step_count, reference.vocab_size);
        printf("maximum absolute error: %.9g\n", (double)max_error);
        if (max_error <= 2.0e-5f)
            result = TEST_PASS;
    }

    free_run_state(&state);
    unload_reference(&reference);
    unload_model(&model);
    return result;
}

static int test_kv_cache_replay(const struct test_context *ctx)
{
    struct tiny_model model;
    struct reference_trace reference;
    struct run_state state;
    float *first_logits;
    char error[160];
    float first_error;
    float second_error;
    float replay_error;
    int result;

    puts("\nTest 23: KV-cache reset and deterministic replay");
    print_rule();
    result = TEST_FAIL;
    first_logits = NULL;

    if (load_model_and_reference(ctx, &model, &reference,
        error, sizeof(error)) != 0) {
        printf("load failed: %s\n", error);
        return TEST_FAIL;
    }
    if (allocate_run_state(&model, &state) != 0) {
        puts("run-state allocation failed");
        unload_reference(&reference);
        unload_model(&model);
        return TEST_FAIL;
    }

    first_logits = (float *)tracked_alloc(model.config.vocab_size,
        sizeof(float), 0);
    if (first_logits != NULL &&
        run_teacher_forced(&model, &state, &reference, &first_error) == 0) {
        memcpy(first_logits, state.logits,
            model.config.vocab_size * sizeof(float));
        if (run_teacher_forced(&model, &state, &reference,
            &second_error) == 0) {
            replay_error = maximum_absolute_error(first_logits,
                state.logits, model.config.vocab_size);
            printf("first reference error=%.9g second=%.9g replay=%.9g\n",
                (double)first_error, (double)second_error,
                (double)replay_error);
            if (first_error <= 2.0e-5f && second_error <= 2.0e-5f &&
                replay_error == 0.0f)
                result = TEST_PASS;
        }
    }

    tracked_free(first_logits);
    free_run_state(&state);
    unload_reference(&reference);
    unload_model(&model);
    return result;
}

static void print_token_sequence(const char *label, const uint32_t *tokens,
    uint32_t count)
{
    uint32_t index;

    printf("%s ids:", label);
    for (index = 0U; index < count; index++)
        printf(" %u", tokens[index]);
    putchar('\n');

    printf("%s bytes: ", label);
    for (index = 0U; index < count; index++) {
        unsigned int token;

        token = tokens[index];
        if (token >= 32U && token < 127U)
            putchar((int)token);
        else
            printf("\\x%02x", token & 0xffU);
    }
    putchar('\n');
}

static int generate_tokens(const struct tiny_model *model,
    struct run_state *state, const struct reference_trace *reference,
    uint32_t *output)
{
    uint32_t position;
    uint32_t index;
    float *logits;

    reset_run_state(model, state);
    logits = NULL;
    position = 0U;
    for (index = 0U; index < reference->prompt_length; index++) {
        logits = forward_token(model, state, reference->prompt[index],
            position++);
        if (logits == NULL)
            return -1;
    }

    for (index = 0U; index < reference->generation_length; index++) {
        uint32_t token;

        token = argmax(logits, model->config.vocab_size);
        output[index] = token;
        logits = forward_token(model, state, token, position++);
        if (logits == NULL)
            return -1;
    }

    return 0;
}

static int test_autoregressive_generation(const struct test_context *ctx)
{
    struct tiny_model model;
    struct reference_trace reference;
    struct run_state state;
    uint32_t *generated;
    char error[160];
    int result;

    puts("\nTest 24: deterministic autoregressive byte-token generation");
    print_rule();
    result = TEST_FAIL;
    generated = NULL;

    if (load_model_and_reference(ctx, &model, &reference,
        error, sizeof(error)) != 0) {
        printf("load failed: %s\n", error);
        return TEST_FAIL;
    }
    if (allocate_run_state(&model, &state) != 0) {
        puts("run-state allocation failed");
        unload_reference(&reference);
        unload_model(&model);
        return TEST_FAIL;
    }

    generated = (uint32_t *)tracked_alloc(reference.generation_length,
        sizeof(uint32_t), 0);
    if (generated != NULL &&
        generate_tokens(&model, &state, &reference, generated) == 0) {
        print_token_sequence("prompt", reference.prompt,
            reference.prompt_length);
        print_token_sequence("generated", generated,
            reference.generation_length);
        if (memcmp(generated, reference.generated,
            reference.generation_length * sizeof(uint32_t)) == 0)
            result = TEST_PASS;
        else {
            print_token_sequence("expected", reference.generated,
                reference.generation_length);
        }
    }

    tracked_free(generated);
    free_run_state(&state);
    unload_reference(&reference);
    unload_model(&model);
    return result;
}

static int copy_prefix(const char *source_path, const char *target_path,
    size_t bytes, int corrupt_magic, int corrupt_weight_count)
{
    FILE *source;
    FILE *target;
    unsigned char buffer[256];
    size_t remaining;
    uint32_t header[HEADER_WORDS];

    source = fopen(source_path, "rb");
    if (source == NULL)
        return -1;
    target = fopen(target_path, "wb");
    if (target == NULL) {
        fclose(source);
        return -1;
    }

    if (corrupt_magic || corrupt_weight_count) {
        if (read_exact(source, header, sizeof(header)) != 0) {
            fclose(target);
            fclose(source);
            return -1;
        }
        if (corrupt_magic)
            header[0] ^= UINT32_C(0x000000ff);
        if (corrupt_weight_count)
            header[10] += 1U;
        if (fwrite(header, 1U, sizeof(header), target) != sizeof(header)) {
            fclose(target);
            fclose(source);
            return -1;
        }
        if (bytes <= sizeof(header)) {
            fclose(target);
            fclose(source);
            return 0;
        }
        remaining = bytes - sizeof(header);
    } else {
        remaining = bytes;
    }

    while (remaining > 0U) {
        size_t request;
        size_t received;

        request = remaining < sizeof(buffer) ? remaining : sizeof(buffer);
        received = fread(buffer, 1U, request, source);
        if (received == 0U)
            break;
        if (fwrite(buffer, 1U, received, target) != received) {
            fclose(target);
            fclose(source);
            return -1;
        }
        remaining -= received;
    }

    fclose(target);
    fclose(source);
    return remaining == 0U ? 0 : -1;
}

static int expect_model_rejection(const char *path, const char *label)
{
    struct tiny_model model;
    char error[160];

    if (load_model(path, &model, error, sizeof(error)) == 0) {
        printf("%s unexpectedly loaded\n", label);
        unload_model(&model);
        return -1;
    }

    printf("%s rejected: %s\n", label, error);
    return 0;
}

static int test_malformed_checkpoint(const struct test_context *ctx)
{
    const char *bad_magic;
    const char *bad_count;
    const char *truncated;
    int result;

    puts("\nTest 25: malformed and truncated checkpoint rejection");
    print_rule();

    bad_magic = "stage2-bad-magic.bin";
    bad_count = "stage2-bad-count.bin";
    truncated = "stage2-truncated.bin";
    result = TEST_FAIL;

    remove(bad_magic);
    remove(bad_count);
    remove(truncated);

    if (copy_prefix(ctx->model_path, bad_magic, HEADER_BYTES, 1, 0) != 0 ||
        copy_prefix(ctx->model_path, bad_count, HEADER_BYTES, 0, 1) != 0 ||
        copy_prefix(ctx->model_path, truncated, HEADER_BYTES + 32U,
            0, 0) != 0) {
        puts("could not create malformed checkpoint fixtures");
        goto cleanup;
    }

    if (expect_model_rejection(bad_magic, "bad magic") == 0 &&
        expect_model_rejection(bad_count, "bad weight count") == 0 &&
        expect_model_rejection(truncated, "truncated payload") == 0)
        result = TEST_PASS;

cleanup:
    remove(bad_magic);
    remove(bad_count);
    remove(truncated);
    return result;
}

static int test_repeated_load_generate_unload(const struct test_context *ctx)
{
    unsigned int iteration;
    size_t starting_allocations;
    size_t starting_bytes;
    int result;

    puts("\nTest 26: repeated load, generate and unload stability");
    print_rule();

    starting_allocations = g_live_allocations;
    starting_bytes = g_live_bytes;
    result = TEST_PASS;

    for (iteration = 0U; iteration < ctx->iterations; iteration++) {
        struct tiny_model model;
        struct reference_trace reference;
        struct run_state state;
        uint32_t *generated;
        char error[160];

        generated = NULL;
        if (load_model_and_reference(ctx, &model, &reference,
            error, sizeof(error)) != 0) {
            printf("iteration %u load failed: %s\n", iteration, error);
            result = TEST_FAIL;
            break;
        }
        if (allocate_run_state(&model, &state) != 0) {
            printf("iteration %u state allocation failed\n", iteration);
            unload_reference(&reference);
            unload_model(&model);
            result = TEST_FAIL;
            break;
        }
        generated = (uint32_t *)tracked_alloc(reference.generation_length,
            sizeof(uint32_t), 0);
        if (generated == NULL ||
            generate_tokens(&model, &state, &reference, generated) != 0 ||
            memcmp(generated, reference.generated,
                reference.generation_length * sizeof(uint32_t)) != 0) {
            printf("iteration %u generation mismatch\n", iteration);
            result = TEST_FAIL;
        }

        tracked_free(generated);
        free_run_state(&state);
        unload_reference(&reference);
        unload_model(&model);

        if (g_live_allocations != starting_allocations ||
            g_live_bytes != starting_bytes) {
            printf("iteration %u leaked: allocations=%lu bytes=%lu\n",
                iteration, (unsigned long)g_live_allocations,
                (unsigned long)g_live_bytes);
            result = TEST_FAIL;
        }
        if (result != TEST_PASS)
            break;
    }

    printf("completed %u/%u cycles; live allocations=%lu live bytes=%lu\n",
        iteration, ctx->iterations, (unsigned long)g_live_allocations,
        (unsigned long)g_live_bytes);
    printf("tracked peak bytes during process: %lu\n",
        (unsigned long)g_peak_bytes);
    return result;
}

static double elapsed_seconds(const struct timespec *start,
    const struct timespec *end)
{
    return (double)(end->tv_sec - start->tv_sec) +
        (double)(end->tv_nsec - start->tv_nsec) / 1000000000.0;
}

static int test_token_benchmark(const struct test_context *ctx)
{
    struct tiny_model model;
    struct run_state state;
    struct timespec start;
    struct timespec end;
    char error[160];
    uint32_t token;
    uint32_t position;
    unsigned int completed;
    double seconds;
    float sink;
    int result;

    puts("\nTest 27: end-to-end token throughput benchmark");
    print_rule();
    result = TEST_FAIL;

    if (load_model(ctx->model_path, &model, error, sizeof(error)) != 0) {
        printf("load failed: %s\n", error);
        return TEST_FAIL;
    }
    if (allocate_run_state(&model, &state) != 0) {
        puts("run-state allocation failed");
        unload_model(&model);
        return TEST_FAIL;
    }

    reset_run_state(&model, &state);
    token = (uint32_t)'M';
    position = 0U;
    sink = 0.0f;
    if (clock_gettime(CLOCK_MONOTONIC, &start) != 0) {
        perror("clock_gettime");
        goto cleanup;
    }

    for (completed = 0U; completed < ctx->bench_tokens; completed++) {
        float *logits;

        if (position == model.config.context_length) {
            reset_run_state(&model, &state);
            position = 0U;
        }
        logits = forward_token(&model, &state, token, position++);
        if (logits == NULL)
            goto cleanup;
        token = argmax(logits, model.config.vocab_size);
        sink += logits[token];
    }

    if (clock_gettime(CLOCK_MONOTONIC, &end) != 0) {
        perror("clock_gettime");
        goto cleanup;
    }

    seconds = elapsed_seconds(&start, &end);
    printf("%u tokens in %.6f seconds", completed, seconds);
    if (seconds > 0.0)
        printf(" (%.3f tokens/second)", (double)completed / seconds);
    putchar('\n');
    printf("benchmark sink: %.9g final token=%u\n",
        (double)sink, token);

    if (completed == ctx->bench_tokens && isfinite(sink))
        result = TEST_PASS;

cleanup:
    free_run_state(&state);
    unload_model(&model);
    return result;
}

static const struct test_case g_tests[] = {
    { 21, "checkpoint header, size and checksum validation",
        test_checkpoint_validation },
    { 22, "full multi-layer forward pass against reference logits",
        test_reference_logits },
    { 23, "KV-cache reset and deterministic replay",
        test_kv_cache_replay },
    { 24, "deterministic autoregressive byte-token generation",
        test_autoregressive_generation },
    { 25, "malformed and truncated checkpoint rejection",
        test_malformed_checkpoint },
    { 26, "repeated load, generate and unload stability",
        test_repeated_load_generate_unload },
    { 27, "end-to-end token throughput benchmark",
        test_token_benchmark }
};

static void print_usage(const char *program)
{
    printf("Usage: %s [options]\n", program);
    puts("  --model PATH          checkpoint file");
    puts("  --reference PATH      reference trace file");
    puts("  --test NUMBER         run one test only");
    puts("  --iterations NUMBER   test 26 cycles (default 100)");
    puts("  --bench-tokens NUMBER test 27 token count (default 4096)");
    puts("  --verbose             print selected configuration");
    puts("  --list                list tests");
    puts("  --help                show this help");
}

static int parse_unsigned(const char *text, unsigned int *value)
{
    char *end;
    unsigned long parsed;

    errno = 0;
    parsed = strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed > UINT32_MAX)
        return -1;
    *value = (unsigned int)parsed;
    return 0;
}

int main(int argc, char **argv)
{
    struct test_context context;
    int selected_test;
    int list_only;
    int argument;
    size_t index;
    int found;

    context.model_path = DEFAULT_MODEL_PATH;
    context.reference_path = DEFAULT_REFERENCE_PATH;
    context.iterations = DEFAULT_ITERATIONS;
    context.bench_tokens = DEFAULT_BENCH_TOKENS;
    context.verbose = 0;
    selected_test = 0;
    list_only = 0;

    for (argument = 1; argument < argc; argument++) {
        if (strcmp(argv[argument], "--model") == 0 && argument + 1 < argc)
            context.model_path = argv[++argument];
        else if (strcmp(argv[argument], "--reference") == 0 &&
            argument + 1 < argc)
            context.reference_path = argv[++argument];
        else if (strcmp(argv[argument], "--test") == 0 &&
            argument + 1 < argc) {
            unsigned int parsed;
            if (parse_unsigned(argv[++argument], &parsed) != 0) {
                fprintf(stderr, "invalid test number\n");
                return 2;
            }
            selected_test = (int)parsed;
        } else if (strcmp(argv[argument], "--iterations") == 0 &&
            argument + 1 < argc) {
            if (parse_unsigned(argv[++argument], &context.iterations) != 0 ||
                context.iterations == 0U) {
                fprintf(stderr, "invalid iteration count\n");
                return 2;
            }
        } else if (strcmp(argv[argument], "--bench-tokens") == 0 &&
            argument + 1 < argc) {
            if (parse_unsigned(argv[++argument], &context.bench_tokens) != 0 ||
                context.bench_tokens == 0U) {
                fprintf(stderr, "invalid benchmark token count\n");
                return 2;
            }
        } else if (strcmp(argv[argument], "--verbose") == 0)
            context.verbose = 1;
        else if (strcmp(argv[argument], "--list") == 0)
            list_only = 1;
        else if (strcmp(argv[argument], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "unknown or incomplete option: %s\n",
                argv[argument]);
            print_usage(argv[0]);
            return 2;
        }
    }

    if (list_only) {
        for (index = 0U; index < ARRAY_LEN(g_tests); index++)
            printf("%d: %s\n", g_tests[index].number, g_tests[index].name);
        return 0;
    }

    puts("MINIX AI Stage 2 tiny autoregressive LLM probe");
    if (context.verbose) {
        printf("model=%s reference=%s iterations=%u bench_tokens=%u selected_test=%d\n",
            context.model_path, context.reference_path, context.iterations,
            context.bench_tokens, selected_test);
    }
    print_rule();

    found = selected_test == 0;
    for (index = 0U; index < ARRAY_LEN(g_tests); index++) {
        int result;

        if (selected_test != 0 && g_tests[index].number != selected_test)
            continue;
        found = 1;
        result = g_tests[index].run(&context);
        report_result(g_tests[index].number, g_tests[index].name, result);
    }

    if (!found) {
        fprintf(stderr, "test %d was not found\n", selected_test);
        return 2;
    }

    print_rule();
    printf("Summary: %d passed, %d failed, %d informational\n",
        g_passed, g_failed, g_info);

    if (g_live_allocations != 0U || g_live_bytes != 0U) {
        printf("WARNING: final tracked allocations=%lu bytes=%lu\n",
            (unsigned long)g_live_allocations,
            (unsigned long)g_live_bytes);
        return 1;
    }

    return g_failed == 0 ? 0 : 1;
}
