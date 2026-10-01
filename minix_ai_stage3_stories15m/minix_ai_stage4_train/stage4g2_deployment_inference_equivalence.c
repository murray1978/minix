#include "stage4_adapter_train.h"

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int minix_llama_embedded_main(int argc, char **argv);

#define main minix_llama_embedded_main
#include "../minix_llama.c"
#undef main

#define G2_BASE_MODEL "/usr/src/minix_ai_stage3_stories15m/stories15M.bin"
#define G2_TOKENIZER "/usr/src/minix_ai_stage3_stories15m/tokenizer.bin"
#define G2_DATASET "stage4e-approved.records"
#define G2_ADAPTER "stage4g1-selected-step20.adapter.bin"
#define G2_G1_REPORT "stage4g1-selected-step20-export-report.txt"
#define G2_REPORT "stage4g2-deployment-inference-equivalence-report.txt"
#define G2_BASE_SHA "cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a"
#define G2_TOKENIZER_SHA "50a52ef822ee9e83de5ce9d0be0a025a773d019437f58b5ff9dcafb063ece361"
#define G2_DATASET_SHA "71744cea4de049e2e192cb2a4494c8a4a68e769b3a7f3b8ed4880433e4583b84"
#define G2_ADAPTER_SHA "a248d7a6f988684404e86e5b0cf6b298eefc54ffc5fe41f016e4c8add3683f5e"
#define G2_G1_REPORT_SHA "37874c6356dbd2189793d0795afc68d75340f06ec2f26f30cb3c4a5ad9a45eb4"
#define G2_DIM 288
#define G2_VOCAB 32000
#define G2_RANK 8
#define G2_SCALE 1.0f
#define G2_TRAIN_REFERENCE "7.026346899236"
#define G2_VALIDATION_REFERENCE "8.328697455428"

#define G2_MAX_LINE 8192
#define G2_MAX_FIELD 256
#define G2_MAX_TEXT 4096

enum {
    G2_SECTION_OUTSIDE = 0,
    G2_SECTION_HEADER,
    G2_SECTION_PROMPT,
    G2_SECTION_TARGET
};

typedef struct {
    char id[G2_MAX_FIELD];
    char task[G2_MAX_FIELD];
    char source[G2_MAX_FIELD];
    char split[G2_MAX_FIELD];
    double weight;
    char prompt[G2_MAX_TEXT];
    char target[G2_MAX_TEXT];
    char canonical_prompt[G2_MAX_TEXT * 2];
    char canonical_sequence[G2_MAX_TEXT * 2];
    int *training_tokens;
    int training_count;
    int first_target_index;
    int target_count;
} g2_record_t;

typedef struct {
    g2_record_t *items;
    size_t count;
    size_t capacity;
} g2_dataset_t;

typedef struct {
    uint64_t records;
    uint64_t target_rows;
    uint64_t top1_correct;
    double total_loss;
} g2_split_stats_t;

static void g2_trim_line_end(char *line)
{
    size_t length = strlen(line);

    while (length > 0U && (line[length - 1U] == '\n' ||
        line[length - 1U] == '\r'))
        line[--length] = '\0';
}

static const char *g2_skip_space(const char *text)
{
    while (*text != '\0' && isspace((unsigned char)*text))
        text++;
    return text;
}

static void g2_trim_space(char *text)
{
    char *start = text;
    char *end;

    while (*start != '\0' && isspace((unsigned char)*start))
        start++;
    if (start != text)
        memmove(text, start, strlen(start) + 1U);
    end = text + strlen(text);
    while (end > text && isspace((unsigned char)end[-1]))
        end--;
    *end = '\0';
}

static int g2_append_prompt_line(char *destination, size_t capacity,
    const char *line)
{
    size_t used = strlen(destination);
    size_t length = strlen(line);

    if (used + length + 2U > capacity)
        return 0;
    memcpy(destination + used, line, length);
    destination[used + length] = '\n';
    destination[used + length + 1U] = '\0';
    return 1;
}

static void g2_dataset_free(g2_dataset_t *dataset)
{
    size_t i;

    for (i = 0; i < dataset->count; i++)
        free(dataset->items[i].training_tokens);
    free(dataset->items);
    memset(dataset, 0, sizeof(*dataset));
}

static int g2_dataset_push(g2_dataset_t *dataset, const g2_record_t *record)
{
    g2_record_t *items;

    if (dataset->count == dataset->capacity) {
        size_t capacity = dataset->capacity == 0U ? 16U : dataset->capacity * 2U;
        items = realloc(dataset->items, capacity * sizeof(*items));
        if (items == NULL)
            return 0;
        dataset->items = items;
        dataset->capacity = capacity;
    }
    dataset->items[dataset->count++] = *record;
    return 1;
}

static int g2_set_metadata(g2_record_t *record, const char *key,
    const char *value)
{
    char *end = NULL;
    double weight;

    if (strcmp(key, "id") == 0) {
        if (record->id[0] != '\0' || value[0] == '\0')
            return 0;
        strncpy(record->id, value, sizeof(record->id) - 1U);
    } else if (strcmp(key, "task") == 0) {
        if (record->task[0] != '\0' || value[0] == '\0')
            return 0;
        strncpy(record->task, value, sizeof(record->task) - 1U);
    } else if (strcmp(key, "source") == 0) {
        if (record->source[0] != '\0' || value[0] == '\0')
            return 0;
        strncpy(record->source, value, sizeof(record->source) - 1U);
    } else if (strcmp(key, "split") == 0) {
        if (record->split[0] != '\0' ||
            (strcmp(value, "train") != 0 && strcmp(value, "validation") != 0 &&
             strcmp(value, "test") != 0 && strcmp(value, "regression") != 0))
            return 0;
        strncpy(record->split, value, sizeof(record->split) - 1U);
    } else if (strcmp(key, "weight") == 0) {
        if (record->weight > 0.0)
            return 0;
        errno = 0;
        weight = strtod(value, &end);
        if (errno != 0 || end == value || *end != '\0' ||
            !(weight > 0.0) || !isfinite(weight))
            return 0;
        record->weight = weight;
    } else if (strcmp(key, "source_unit") != 0) {
        return 0;
    }
    return 1;
}

static int g2_load_dataset(const char *path, g2_dataset_t *dataset)
{
    FILE *file;
    char line[G2_MAX_LINE];
    g2_record_t current;
    int section = G2_SECTION_OUTSIDE;
    int target_lines = 0;

    memset(dataset, 0, sizeof(*dataset));
    memset(&current, 0, sizeof(current));
    file = fopen(path, "rb");
    if (file == NULL)
        return 0;
    while (fgets(line, sizeof(line), file) != NULL) {
        const char *start;

        if (strchr(line, '\n') == NULL && !feof(file))
            goto fail;
        g2_trim_line_end(line);
        start = g2_skip_space(line);
        if (section == G2_SECTION_OUTSIDE) {
            if (*start == '\0' || *start == '#')
                continue;
            if (strcmp(start, "%%BEGIN") != 0)
                goto fail;
            memset(&current, 0, sizeof(current));
            target_lines = 0;
            section = G2_SECTION_HEADER;
        } else if (section == G2_SECTION_HEADER) {
            const char *equals;
            char key[G2_MAX_FIELD];
            char value[G2_MAX_TEXT];
            size_t key_length;

            if (strcmp(start, "%%PROMPT") == 0) {
                section = G2_SECTION_PROMPT;
                continue;
            }
            if (*start == '\0')
                continue;
            equals = strchr(start, '=');
            if (equals == NULL)
                goto fail;
            key_length = (size_t)(equals - start);
            if (key_length == 0U || key_length >= sizeof(key) ||
                strlen(equals + 1) >= sizeof(value))
                goto fail;
            memcpy(key, start, key_length);
            key[key_length] = '\0';
            strcpy(value, equals + 1);
            g2_trim_space(key);
            g2_trim_space(value);
            if (!g2_set_metadata(&current, key, value))
                goto fail;
        } else if (section == G2_SECTION_PROMPT) {
            if (strcmp(start, "%%TARGET") == 0) {
                section = G2_SECTION_TARGET;
                continue;
            }
            if (!g2_append_prompt_line(current.prompt,
                sizeof(current.prompt), start))
                goto fail;
        } else {
            if (strcmp(start, "%%END") == 0) {
                g2_trim_space(current.prompt);
                g2_trim_space(current.target);
                if (current.id[0] == '\0' || current.task[0] == '\0' ||
                    current.source[0] == '\0' || current.split[0] == '\0' ||
                    !(current.weight > 0.0) || current.prompt[0] == '\0' ||
                    current.target[0] == '\0' || target_lines != 1 ||
                    !g2_dataset_push(dataset, &current))
                    goto fail;
                memset(&current, 0, sizeof(current));
                target_lines = 0;
                section = G2_SECTION_OUTSIDE;
                continue;
            }
            if (*start == '\0')
                continue;
            target_lines++;
            if (target_lines == 1)
                strncpy(current.target, start, sizeof(current.target) - 1U);
        }
    }
    if (ferror(file) || section != G2_SECTION_OUTSIDE || dataset->count == 0U)
        goto fail;
    if (fclose(file) != 0)
        goto fail_closed;
    return 1;

fail:
    fclose(file);
fail_closed:
    g2_dataset_free(dataset);
    return 0;
}

static int g2_prepare_record_tokens(Tokenizer *tokenizer,
    const Transformer *transformer, g2_record_t *record)
{
    int *prompt_tokens = NULL;
    int prompt_count = 0;

    if (snprintf(record->canonical_prompt, sizeof(record->canonical_prompt),
        "User: %s\nAssistant:", record->prompt) >=
        (int)sizeof(record->canonical_prompt) ||
        snprintf(record->canonical_sequence, sizeof(record->canonical_sequence),
            "User: %s\nAssistant: %s\n", record->prompt, record->target) >=
        (int)sizeof(record->canonical_sequence))
        return 0;
    if (!encode_text(tokenizer, record->canonical_prompt, 1, 0,
        &prompt_tokens, &prompt_count) ||
        !encode_text(tokenizer, record->canonical_sequence, 1, 0,
        &record->training_tokens, &record->training_count)) {
        free(prompt_tokens);
        free(record->training_tokens);
        record->training_tokens = NULL;
        return 0;
    }
    free(prompt_tokens);
    if (record->training_count > transformer->config.seq_len) {
        free(record->training_tokens);
        record->training_tokens = NULL;
        return 0;
    }
    record->first_target_index = prompt_count;
    if (record->first_target_index <= 0 ||
        record->first_target_index >= record->training_count) {
        free(record->training_tokens);
        record->training_tokens = NULL;
        return 0;
    }
    record->target_count = record->training_count - record->first_target_index;
    return record->target_count > 0;
}

static int g2_loss_from_logits(const float *logits, int vocab, int target,
    double *loss_out)
{
    double maximum, sum = 0.0;
    int i;

    if (target < 0 || target >= vocab || !isfinite(logits[0]))
        return 0;
    maximum = logits[0];
    for (i = 1; i < vocab; i++) {
        if (!isfinite(logits[i]))
            return 0;
        if (logits[i] > maximum)
            maximum = logits[i];
    }
    for (i = 0; i < vocab; i++)
        sum += exp((double)logits[i] - maximum);
    if (!(sum > 0.0) || !isfinite(sum))
        return 0;
    *loss_out = -((double)logits[target] - maximum - log(sum));
    return isfinite(*loss_out);
}

static int g2_argmax_token(const float *logits, int vocab)
{
    int i, best = 0;
    float best_value = logits[0];

    for (i = 1; i < vocab; i++) {
        if (logits[i] > best_value) {
            best_value = logits[i];
            best = i;
        }
    }
    return best;
}

static int g2_hash_matches(const char *path, const char *expected, char actual[65])
{
    return stage4_hash_file_sha256_hex(path, actual) &&
        strcmp(actual, expected) == 0;
}

static int g2_allocate_deployment_adapter(stage4_adapter_t *ad)
{
    memset(ad, 0, sizeof(*ad));
    ad->dim = G2_DIM;
    ad->vocab = G2_VOCAB;
    ad->rank = G2_RANK;
    ad->scale = G2_SCALE;
    ad->a_count = (size_t)G2_RANK * (size_t)G2_DIM;
    ad->b_count = (size_t)G2_VOCAB * (size_t)G2_RANK;
    ad->a = calloc(ad->a_count, sizeof(float));
    ad->b = calloc(ad->b_count, sizeof(float));
    ad->u = calloc((size_t)G2_RANK, sizeof(float));
    if (ad->a == NULL || ad->b == NULL || ad->u == NULL) {
        free(ad->a);
        free(ad->b);
        free(ad->u);
        memset(ad, 0, sizeof(*ad));
        return 0;
    }
    return 1;
}

static void g2_free_deployment_adapter(stage4_adapter_t *ad)
{
    free(ad->a);
    free(ad->b);
    free(ad->u);
    memset(ad, 0, sizeof(*ad));
}

static int g2_adapter_values_finite(const stage4_adapter_t *ad)
{
    size_t i;

    for (i = 0; i < ad->a_count; i++)
        if (!isfinite(ad->a[i]))
            return 0;
    for (i = 0; i < ad->b_count; i++)
        if (!isfinite(ad->b[i]))
            return 0;
    return 1;
}

static int g2_logits_finite(const float *logits, int vocab)
{
    int i;

    for (i = 0; i < vocab; i++)
        if (!isfinite(logits[i]))
            return 0;
    return 1;
}

static int g2_format_matches(double value, const char *reference)
{
    char formatted[64];

    if (snprintf(formatted, sizeof(formatted), "%.12f", value) < 0)
        return 0;
    return strcmp(formatted, reference) == 0;
}

static int g2_evaluate_record(Transformer *tr, Tokenizer *tok,
    g2_record_t *record, stage4_adapter_t *adapter,
    float *adapted_logits, g2_split_stats_t *stats, int *all_losses_finite,
    int *all_logits_finite)
{
    int k;
    uint64_t record_rows = 0U;
    double record_loss = 0.0;
    uint64_t record_top1 = 0U;

    if (!g2_prepare_record_tokens(tok, tr, record))
        return 0;
    if (record->training_count <= 1 || record->first_target_index <= 0)
        return 0;

    clear_run_state(tr);
    for (k = 1; k < record->training_count; k++) {
        const float *base_logits = forward(tr,
            record->training_tokens[k - 1], k - 1);
        double loss;
        int target;
        int prediction;

        if (base_logits == NULL ||
            !g2_logits_finite(base_logits, tr->config.vocab_size)) {
            *all_logits_finite = 0;
            return 0;
        }
        if (k < record->first_target_index)
            continue;
        if (tr->state.x == NULL || !g2_logits_finite(tr->state.x, tr->config.dim)) {
            *all_logits_finite = 0;
            return 0;
        }
        stage4_adapter_forward_logits(adapter, tr->state.x,
            base_logits, adapted_logits);
        if (!g2_logits_finite(adapted_logits, tr->config.vocab_size)) {
            *all_logits_finite = 0;
            return 0;
        }
        target = record->training_tokens[k];
        if (!g2_loss_from_logits(adapted_logits,
            tr->config.vocab_size, target, &loss)) {
            *all_losses_finite = 0;
            return 0;
        }
        prediction = g2_argmax_token(adapted_logits,
            tr->config.vocab_size);
        record_loss += loss;
        record_rows++;
        if (!isfinite(record_loss)) {
            *all_losses_finite = 0;
            return 0;
        }
        if (prediction == target)
            record_top1++;
    }

    if (record->target_count <= 0 ||
        record_rows != (uint64_t)record->target_count)
        return 0;
    stats->records++;
    stats->target_rows += (uint64_t)record->target_count;
    stats->top1_correct += record_top1;
    stats->total_loss += record_loss;
    return isfinite(stats->total_loss);
}

static int g2_write_report(const char *base_sha, const char *tokenizer_sha,
    const char *dataset_sha, const char *adapter_sha, const char *g1_report_sha,
    const stage4_adapter_t *adapter, const g2_split_stats_t *train,
    const g2_split_stats_t *validation, int adapter_finite,
    int all_losses_finite, int all_logits_finite, int train_loss_match,
    int validation_loss_match, int train_top1_match,
    int validation_top1_match, int pass)
{
    FILE *report = fopen(G2_REPORT, "wb");
    int ok = 1;

    if (report == NULL)
        return 0;
    if (fprintf(report,
        "stage=4G2\nmode=frozen_adapter_live_full_context_equivalence\n"
        "base_model_path=%s\nbase_model_sha256=%s\n"
        "tokenizer_path=%s\ntokenizer_sha256=%s\n"
        "dataset_path=%s\ndataset_sha256=%s\n"
        "adapter_path=%s\nadapter_sha256=%s\n"
        "g1_report_path=%s\ng1_report_sha256=%s\n"
        "adapter_format_version=2\nadapter_layout_version=1\n"
        "adapter_rank=%d\nadapter_input_dim=%d\nadapter_output_vocab=%d\nadapter_scale=%.9g\n"
        "adapter_values_finite=%s\n"
        "inference_source=live_transformer_full_context\n"
        "training_checkpoint_loaded=no\noptimizer_state_loaded=no\n"
        "hidden_state_cache_used_for_inference=no\n",
        G2_BASE_MODEL, base_sha, G2_TOKENIZER, tokenizer_sha,
        G2_DATASET, dataset_sha, G2_ADAPTER, adapter_sha,
        G2_G1_REPORT, g1_report_sha, adapter->rank, adapter->dim,
        adapter->vocab, (double)adapter->scale,
        adapter_finite ? "yes" : "no") < 0)
        ok = 0;
    if (fprintf(report,
        "train_records_evaluated=%llu\ntrain_target_rows_evaluated=%llu\n"
        "train_total_loss=%.17g\ntrain_average_loss=%.12f\n"
        "train_top1_correct=%llu\ntrain_top1_total=%llu\ntrain_top1_fraction=%.12f\n"
        "validation_records_evaluated=%llu\nvalidation_target_rows_evaluated=%llu\n"
        "validation_total_loss=%.17g\nvalidation_average_loss=%.12f\n"
        "validation_top1_correct=%llu\nvalidation_top1_total=%llu\nvalidation_top1_fraction=%.12f\n",
        (unsigned long long)train->records,
        (unsigned long long)train->target_rows, train->total_loss,
        train->target_rows ? train->total_loss / (double)train->target_rows : 0.0,
        (unsigned long long)train->top1_correct,
        (unsigned long long)train->target_rows,
        train->target_rows ? (double)train->top1_correct / (double)train->target_rows : 0.0,
        (unsigned long long)validation->records,
        (unsigned long long)validation->target_rows, validation->total_loss,
        validation->target_rows ? validation->total_loss / (double)validation->target_rows : 0.0,
        (unsigned long long)validation->top1_correct,
        (unsigned long long)validation->target_rows,
        validation->target_rows ? (double)validation->top1_correct / (double)validation->target_rows : 0.0) < 0)
        ok = 0;
    if (fprintf(report,
        "test_records_evaluated=0\ntest_target_rows_evaluated=0\n"
        "gradient_rows_consumed=0\noptimizer_updates=0\n"
        "reference_train_average_loss=%s\nreference_validation_average_loss=%s\n"
        "reference_train_top1_correct=4\nreference_train_top1_total=111\n"
        "reference_validation_top1_correct=4\nreference_validation_top1_total=33\n"
        "loss_reference_comparison=formatted_12_decimal_string\n"
        "train_loss_reference_match=%s\nvalidation_loss_reference_match=%s\n"
        "train_top1_reference_match=%s\nvalidation_top1_reference_match=%s\n"
        "all_losses_finite=%s\nall_logits_finite=%s\n"
        "gate_adapter_identity=yes\ngate_live_full_context=yes\n"
        "gate_no_training_checkpoint=yes\ngate_no_optimizer=yes\n"
        "gate_no_test_access=yes\n"
        "gate_train_reference_match=%s\ngate_validation_reference_match=%s\n"
        "gate_all_finite=%s\ngate_stage4g2=%s\n",
        G2_TRAIN_REFERENCE, G2_VALIDATION_REFERENCE,
        train_loss_match ? "yes" : "no",
        validation_loss_match ? "yes" : "no",
        train_top1_match ? "yes" : "no",
        validation_top1_match ? "yes" : "no",
        all_losses_finite ? "yes" : "no",
        all_logits_finite ? "yes" : "no",
        train_loss_match && train_top1_match ? "yes" : "no",
        validation_loss_match && validation_top1_match ? "yes" : "no",
        adapter_finite && all_losses_finite && all_logits_finite ? "yes" : "no",
        pass ? "PASS" : "FAIL") < 0)
        ok = 0;
    if (fclose(report) != 0)
        ok = 0;
    return ok;
}

int main(void)
{
    Transformer transformer;
    Tokenizer tokenizer;
    g2_dataset_t dataset;
    stage4_adapter_t adapter;
    stage4_base_identity_t base_identity;
    g2_split_stats_t train, validation;
    char base_sha[65], tokenizer_sha[65], dataset_sha[65];
    char adapter_sha[65], g1_report_sha[65];
    float *adapted_logits = NULL;
    uint64_t base_bytes = 0U;
    size_t i;
    uint32_t train_records_seen = 0U, validation_records_seen = 0U;
    uint32_t test_records_seen = 0U;
    int adapter_ready = 0, transformer_ready = 0, tokenizer_ready = 0;
    int all_losses_finite = 1, all_logits_finite = 1;
    int adapter_finite = 0, train_loss_match = 0, validation_loss_match = 0;
    int train_top1_match = 0, validation_top1_match = 0, pass = 0;
    FILE *report_probe;

    memset(&transformer, 0, sizeof(transformer));
    memset(&tokenizer, 0, sizeof(tokenizer));
    memset(&dataset, 0, sizeof(dataset));
    memset(&adapter, 0, sizeof(adapter));
    memset(&base_identity, 0, sizeof(base_identity));
    memset(&train, 0, sizeof(train));
    memset(&validation, 0, sizeof(validation));

    report_probe = fopen(G2_REPORT, "rb");
    if (report_probe != NULL) {
        fclose(report_probe);
        fprintf(stderr, "error: refusing to overwrite existing G2 report\n");
        return EXIT_FAILURE;
    }
    if (!g2_hash_matches(G2_BASE_MODEL, G2_BASE_SHA, base_sha) ||
        !g2_hash_matches(G2_TOKENIZER, G2_TOKENIZER_SHA, tokenizer_sha) ||
        !g2_hash_matches(G2_DATASET, G2_DATASET_SHA, dataset_sha) ||
        !g2_hash_matches(G2_ADAPTER, G2_ADAPTER_SHA, adapter_sha) ||
        !g2_hash_matches(G2_G1_REPORT, G2_G1_REPORT_SHA, g1_report_sha)) {
        fprintf(stderr, "error: pinned G2 input SHA-256 mismatch\n");
        goto cleanup;
    }
    if (!stage4_file_size_bytes(G2_BASE_MODEL, &base_bytes)) {
        fprintf(stderr, "error: cannot read base model size\n");
        goto cleanup;
    }
    strcpy(base_identity.base_sha256, base_sha);
    base_identity.base_checkpoint_bytes = base_bytes;

    if (!g2_load_dataset(G2_DATASET, &dataset)) {
        fprintf(stderr, "error: accepted dataset parse failed\n");
        goto cleanup;
    }
    load_transformer(&transformer, G2_BASE_MODEL);
    transformer_ready = 1;
    if (transformer.config.dim != G2_DIM ||
        transformer.config.vocab_size != G2_VOCAB) {
        fprintf(stderr, "error: base model shape mismatch\n");
        goto cleanup;
    }
    load_tokenizer(&tokenizer, G2_TOKENIZER, transformer.config.vocab_size);
    tokenizer_ready = 1;

    if (!g2_allocate_deployment_adapter(&adapter)) {
        fprintf(stderr, "error: deployment adapter allocation failed\n");
        goto cleanup;
    }
    adapter_ready = 1;
    if (!stage4_adapter_load_file(G2_ADAPTER, &adapter, &base_identity, 0)) {
        fprintf(stderr, "error: G1 deployment adapter load/metadata validation failed\n");
        goto cleanup;
    }
    adapter_finite = g2_adapter_values_finite(&adapter);
    if (!adapter_finite || adapter.rank != G2_RANK || adapter.dim != G2_DIM ||
        adapter.vocab != G2_VOCAB || adapter.scale != G2_SCALE) {
        fprintf(stderr, "error: loaded adapter shape/scale/value validation failed\n");
        goto cleanup;
    }
    adapted_logits = calloc((size_t)G2_VOCAB, sizeof(float));
    if (adapted_logits == NULL) {
        fprintf(stderr, "error: adapted logits allocation failed\n");
        goto cleanup;
    }

    for (i = 0; i < dataset.count; i++) {
        g2_record_t *record = &dataset.items[i];
        g2_split_stats_t *stats;

        if (strcmp(record->split, "train") == 0) {
            train_records_seen++;
            stats = &train;
        } else if (strcmp(record->split, "validation") == 0) {
            validation_records_seen++;
            stats = &validation;
        } else {
            if (strcmp(record->split, "test") == 0)
                test_records_seen++;
            continue;
        }
        if (!g2_evaluate_record(&transformer, &tokenizer, record,
            &adapter, adapted_logits, stats, &all_losses_finite,
            &all_logits_finite)) {
            fprintf(stderr, "error: live full-context evaluation failed for %s\n",
                record->id);
            goto cleanup;
        }
    }

    if (train_records_seen != 20U || validation_records_seen != 5U ||
        test_records_seen != 5U || train.records != 20U ||
        train.target_rows != 111U || validation.records != 5U ||
        validation.target_rows != 33U) {
        fprintf(stderr, "error: live evaluation split counts mismatch\n");
        goto cleanup;
    }
    train_loss_match = g2_format_matches(
        train.total_loss / (double)train.target_rows, G2_TRAIN_REFERENCE);
    validation_loss_match = g2_format_matches(
        validation.total_loss / (double)validation.target_rows,
        G2_VALIDATION_REFERENCE);
    train_top1_match = train.top1_correct == 4U && train.target_rows == 111U;
    validation_top1_match = validation.top1_correct == 4U &&
        validation.target_rows == 33U;
    pass = train_records_seen == 20U && validation_records_seen == 5U &&
        test_records_seen == 5U && train_loss_match && validation_loss_match &&
        train_top1_match && validation_top1_match && adapter_finite &&
        all_losses_finite && all_logits_finite;

    if (!g2_write_report(base_sha, tokenizer_sha, dataset_sha, adapter_sha,
        g1_report_sha, &adapter, &train, &validation, adapter_finite,
        all_losses_finite, all_logits_finite, train_loss_match,
        validation_loss_match, train_top1_match, validation_top1_match,
        pass)) {
        fprintf(stderr, "error: G2 report write failed\n");
        pass = 0;
        goto cleanup;
    }
    printf("train_average_loss=%.12f\nvalidation_average_loss=%.12f\n"
        "train_top1_correct=%llu\nvalidation_top1_correct=%llu\n"
        "test_records_evaluated=0\ntest_target_rows_evaluated=0\n"
        "gate_stage4g2=%s\n",
        train.total_loss / (double)train.target_rows,
        validation.total_loss / (double)validation.target_rows,
        (unsigned long long)train.top1_correct,
        (unsigned long long)validation.top1_correct,
        pass ? "PASS" : "FAIL");

cleanup:
    free(adapted_logits);
    if (adapter_ready)
        g2_free_deployment_adapter(&adapter);
    if (tokenizer_ready)
        free_tokenizer(&tokenizer);
    if (transformer_ready)
        free_transformer(&transformer);
    g2_dataset_free(&dataset);
    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
