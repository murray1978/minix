#include "stage4_adapter_train.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int minix_llama_embedded_main(int argc, char **argv);

#define main minix_llama_embedded_main
#include "../minix_llama.c"
#undef main

#define D2_CACHE_HEADER_SIZE 172U
#define D2_CACHE_ENTRY_SIZE 1168U
#define D2_CACHE_DIM 288U
#define D2_CACHE_VOCAB 32000U
#define D2_CACHE_RECORDS 30U
#define D2_CACHE_ROWS 165U
#define D2_CACHE_BYTES 192892U
#define D2_SPLIT_COUNT 4U
#define D2_RANK 8
#define D2_SCALE 1.0f
#define D2_LEARNING_RATE 0.001
#define D2_BETA1 0.9
#define D2_BETA2 0.999
#define D2_EPSILON 1.0e-8
#define D2_WEIGHT_DECAY 0.0
#define D2_GRADIENT_CLIP 1.0
#define D2_SPLIT_TRAIN 0U
#define D2_SPLIT_VALIDATION 1U
#define D2_SPLIT_TEST 2U
#define D2_SPLIT_REGRESSION 3U

static const unsigned char d2_cache_magic[8] = {
    'S', '4', 'T', 'C', 'A', 'C', 'H', '1'
};
#ifndef STAGE4E_D2_CORE_ONLY
static const char d2_expected_dataset_sha256[] =
    "71744cea4de049e2e192cb2a4494c8a4a68e769b3a7f3b8ed4880433e4583b84";
static const char d2_expected_cache_sha256[] =
    "d6288ad3d128c7182c20088aa57a1bca8bcaa1e13314fb0cdbd60eba14278def";
static const char d2_expected_checkpoint_sha256[] =
    "cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a";
static const char d2_expected_tokenizer_sha256[] =
    "50a52ef822ee9e83de5ce9d0be0a025a773d019437f58b5ff9dcafb063ece361";
static const char d2_expected_d1_report_sha256[] =
    "45ebaa750f17d02d80feed332b16c9c29c3782a544aed7544da72a62161c76a6";
#endif
static const uint32_t d2_expected_split_records[D2_SPLIT_COUNT] = {
    20U, 5U, 5U, 0U
};
static const uint32_t d2_expected_split_rows[D2_SPLIT_COUNT] = {
    111U, 33U, 21U, 0U
};
static const char *d2_pre_total_loss[3] = {
    "1158.750195944566", "300.506654007981", "223.961209448908"
};
static const char *d2_pre_average_loss[3] = {
    "10.439190954456", "9.106262242666", "10.664819497567"
};
static const uint32_t d2_pre_top1[3] = { 0U, 2U, 0U };

typedef struct {
    uint32_t format_version;
    uint32_t header_size;
    uint32_t entry_size;
    uint32_t float_format;
    uint32_t byte_order;
    uint32_t model_dim;
    uint32_t vocab_size;
    uint32_t record_count;
    uint32_t target_count;
    uint32_t split_records[D2_SPLIT_COUNT];
    uint32_t split_targets[D2_SPLIT_COUNT];
    unsigned char dataset_sha[32];
    unsigned char checkpoint_sha[32];
    unsigned char tokenizer_sha[32];
} d2_cache_header_t;

typedef struct {
    uint32_t record_index;
    uint32_t input_position;
    uint32_t target_token_id;
    uint32_t split_id;
    float hidden[D2_CACHE_DIM];
} d2_cache_row_t;

typedef struct {
    uint32_t target_count;
    uint64_t top1_correct;
    double loss_sum;
} d2_split_metrics_t;

typedef struct {
    uint32_t evaluated_rows;
    uint32_t rows_seen[D2_SPLIT_COUNT];
    uint64_t overall_top1_correct;
    double overall_loss_sum;
    d2_split_metrics_t splits[3];
} d2_metrics_t;

typedef struct {
    uint32_t a_nonzero;
    uint32_t b_nonzero;
    double a_max_abs;
    double b_max_abs;
} d2_gradient_stats_t;

static uint32_t d2_decode_u32_le(const unsigned char bytes[4])
{
    return (uint32_t)bytes[0] |
        ((uint32_t)bytes[1] << 8) |
        ((uint32_t)bytes[2] << 16) |
        ((uint32_t)bytes[3] << 24);
}

static int d2_read_exact(FILE *file, void *buffer, size_t size)
{
    return size == 0U || fread(buffer, 1, size, file) == size;
}

static int d2_read_u32_le(FILE *file, uint32_t *value)
{
    unsigned char bytes[4];
    if (!d2_read_exact(file, bytes, sizeof(bytes)))
        return 0;
    *value = d2_decode_u32_le(bytes);
    return 1;
}

static int d2_read_f32_le(FILE *file, float *value)
{
    uint32_t bits;
    if (!d2_read_u32_le(file, &bits))
        return 0;
    memcpy(value, &bits, sizeof(bits));
    return 1;
}

static int d2_read_header(FILE *file, d2_cache_header_t *header)
{
    unsigned char magic[8];
    uint32_t i;

    memset(header, 0, sizeof(*header));
    if (!d2_read_exact(file, magic, sizeof(magic)) ||
        memcmp(magic, d2_cache_magic, sizeof(magic)) != 0)
        return 0;
    if (!d2_read_u32_le(file, &header->format_version) ||
        !d2_read_u32_le(file, &header->header_size) ||
        !d2_read_u32_le(file, &header->entry_size) ||
        !d2_read_u32_le(file, &header->float_format) ||
        !d2_read_u32_le(file, &header->byte_order) ||
        !d2_read_u32_le(file, &header->model_dim) ||
        !d2_read_u32_le(file, &header->vocab_size) ||
        !d2_read_u32_le(file, &header->record_count) ||
        !d2_read_u32_le(file, &header->target_count))
        return 0;
    for (i = 0U; i < D2_SPLIT_COUNT; i++) {
        if (!d2_read_u32_le(file, &header->split_records[i]))
            return 0;
    }
    for (i = 0U; i < D2_SPLIT_COUNT; i++) {
        if (!d2_read_u32_le(file, &header->split_targets[i]))
            return 0;
    }
    return d2_read_exact(file, header->dataset_sha, 32U) &&
        d2_read_exact(file, header->checkpoint_sha, 32U) &&
        d2_read_exact(file, header->tokenizer_sha, 32U);
}

static int d2_read_row(FILE *file, d2_cache_row_t *row)
{
    uint32_t i;
    if (!d2_read_u32_le(file, &row->record_index) ||
        !d2_read_u32_le(file, &row->input_position) ||
        !d2_read_u32_le(file, &row->target_token_id) ||
        !d2_read_u32_le(file, &row->split_id))
        return 0;
    for (i = 0U; i < D2_CACHE_DIM; i++) {
        if (!d2_read_f32_le(file, &row->hidden[i]) || !isfinite(row->hidden[i]))
            return 0;
    }
    return 1;
}

static int d2_hex_to_raw(const char *hex, unsigned char raw[32])
{
    size_t i;
    if (hex == NULL || strlen(hex) != 64U)
        return 0;
    for (i = 0U; i < 32U; i++) {
        int high;
        int low;
        char h = hex[i * 2U];
        char l = hex[i * 2U + 1U];
        high = h >= '0' && h <= '9' ? h - '0' :
            (h >= 'a' && h <= 'f' ? h - 'a' + 10 : -1);
        low = l >= '0' && l <= '9' ? l - '0' :
            (l >= 'a' && l <= 'f' ? l - 'a' + 10 : -1);
        if (high < 0 || low < 0)
            return 0;
        raw[i] = (unsigned char)((high << 4) | low);
    }
    return 1;
}

static int d2_float_format_supported(void)
{
    const uint16_t endian_test = 1U;
    float one = 1.0f;
    uint32_t bits = 0U;
    if (sizeof(float) != 4U || sizeof(uint32_t) != 4U ||
        FLT_RADIX != 2 || FLT_MANT_DIG != 24 || FLT_MAX_EXP != 128 ||
        FLT_MIN_EXP != -125 || *((const unsigned char *)&endian_test) != 1U)
        return 0;
    memcpy(&bits, &one, sizeof(bits));
    return bits == UINT32_C(0x3f800000);
}

static int d2_header_valid(const d2_cache_header_t *header,
    const unsigned char dataset_sha[32],
    const unsigned char checkpoint_sha[32],
    const unsigned char tokenizer_sha[32])
{
    uint32_t i;
    if (header->format_version != 1U ||
        header->header_size != D2_CACHE_HEADER_SIZE ||
        header->entry_size != D2_CACHE_ENTRY_SIZE ||
        header->float_format != 1U || header->byte_order != 1U ||
        header->model_dim != D2_CACHE_DIM ||
        header->vocab_size != D2_CACHE_VOCAB ||
        header->record_count != D2_CACHE_RECORDS ||
        header->target_count != D2_CACHE_ROWS ||
        memcmp(header->dataset_sha, dataset_sha, 32U) != 0 ||
        memcmp(header->checkpoint_sha, checkpoint_sha, 32U) != 0 ||
        memcmp(header->tokenizer_sha, tokenizer_sha, 32U) != 0)
        return 0;
    for (i = 0U; i < D2_SPLIT_COUNT; i++) {
        if (header->split_records[i] != d2_expected_split_records[i] ||
            header->split_targets[i] != d2_expected_split_rows[i])
            return 0;
    }
    return 1;
}

static uint32_t d2_float_bits(float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static int d2_loss_from_logits(const float *logits, int target, double *loss_out)
{
    double max_logit;
    double sum = 0.0;
    double logprob;
    int i;
    if (target < 0 || target >= (int)D2_CACHE_VOCAB || !isfinite(logits[0]))
        return 0;
    max_logit = logits[0];
    for (i = 1; i < (int)D2_CACHE_VOCAB; i++) {
        if (!isfinite(logits[i]))
            return 0;
        if (logits[i] > max_logit)
            max_logit = logits[i];
    }
    for (i = 0; i < (int)D2_CACHE_VOCAB; i++)
        sum += exp((double)logits[i] - max_logit);
    if (!(sum > 0.0) || !isfinite(sum))
        return 0;
    logprob = (double)logits[target] - max_logit - log(sum);
    *loss_out = -logprob;
    return isfinite(*loss_out);
}

static int d2_argmax(const float *logits)
{
    int best = 0;
    int i;
    float best_value = logits[0];
    for (i = 1; i < (int)D2_CACHE_VOCAB; i++) {
        if (logits[i] > best_value) {
            best_value = logits[i];
            best = i;
        }
    }
    return best;
}

static int d2_format_matches(double value, const char *expected)
{
    char formatted[64];
    if (snprintf(formatted, sizeof(formatted), "%.12f", value) < 0)
        return 0;
    return strcmp(formatted, expected) == 0;
}

static void d2_flush_record(d2_metrics_t *metrics, uint32_t split,
    uint32_t record_rows, double record_loss, uint64_t record_top1)
{
    metrics->splits[split].target_count += record_rows;
    metrics->splits[split].loss_sum += record_loss;
    metrics->splits[split].top1_correct += record_top1;
    metrics->overall_loss_sum += record_loss;
    metrics->overall_top1_correct += record_top1;
}

static int d2_evaluate(const d2_cache_row_t *rows, stage4_adapter_t *adapter,
    Transformer *transformer, d2_metrics_t *metrics,
    uint32_t *zero_correction_rows, uint32_t *nonzero_correction_rows)
{
    float *base_logits = calloc(D2_CACHE_VOCAB, sizeof(float));
    float *final_logits = calloc(D2_CACHE_VOCAB, sizeof(float));
    uint32_t row_index;
    uint32_t current_record = 0U;
    uint32_t current_split = 0U;
    uint32_t record_rows = 0U;
    uint64_t record_top1 = 0U;
    double record_loss = 0.0;
    int have_record = 0;

    if (base_logits == NULL || final_logits == NULL) {
        free(base_logits);
        free(final_logits);
        return 0;
    }
    memset(metrics, 0, sizeof(*metrics));
    if (zero_correction_rows != NULL)
        *zero_correction_rows = 0U;
    if (nonzero_correction_rows != NULL)
        *nonzero_correction_rows = 0U;

    for (row_index = 0U; row_index < D2_CACHE_ROWS; row_index++) {
        const d2_cache_row_t *row = &rows[row_index];
        uint32_t vocab_index;
        int row_correction_nonzero = 0;
        double loss;
        int prediction;

        if (row->split_id >= D2_SPLIT_REGRESSION) {
            free(base_logits);
            free(final_logits);
            return 0;
        }
        if (!have_record || row->record_index != current_record) {
            if (have_record)
                d2_flush_record(metrics, current_split, record_rows,
                    record_loss, record_top1);
            current_record = row->record_index;
            current_split = row->split_id;
            record_rows = 0U;
            record_top1 = 0U;
            record_loss = 0.0;
            have_record = 1;
        }
        if (row->split_id != current_split) {
            free(base_logits);
            free(final_logits);
            return 0;
        }

        matmul(base_logits, row->hidden, transformer->weights.wcls,
            (int)D2_CACHE_DIM, (int)D2_CACHE_VOCAB);
        stage4_adapter_forward_logits(adapter, row->hidden,
            base_logits, final_logits);
        if (nonzero_correction_rows != NULL) {
            for (vocab_index = 0U; vocab_index < D2_CACHE_VOCAB; vocab_index++) {
                double correction = 0.0;
                uint32_t rank_index;
                for (rank_index = 0U; rank_index < D2_RANK; rank_index++) {
                    correction += (double)adapter->b[(size_t)vocab_index * D2_RANK +
                        rank_index] * (double)adapter->u[rank_index];
                }
                if (correction != 0.0)
                    row_correction_nonzero = 1;
            }
            if (row_correction_nonzero)
                (*nonzero_correction_rows)++;
            else if (zero_correction_rows != NULL)
                (*zero_correction_rows)++;
        }
        if (!d2_loss_from_logits(final_logits, (int)row->target_token_id, &loss)) {
            free(base_logits);
            free(final_logits);
            return 0;
        }
        prediction = d2_argmax(final_logits);
        record_loss += loss;
        record_rows++;
        metrics->evaluated_rows++;
        metrics->rows_seen[row->split_id]++;
        if (prediction == (int)row->target_token_id)
            record_top1++;
    }
    if (have_record)
        d2_flush_record(metrics, current_split, record_rows, record_loss, record_top1);
    free(base_logits);
    free(final_logits);
    return metrics->evaluated_rows == D2_CACHE_ROWS;
}

static int d2_preupdate_matches(const d2_metrics_t *metrics)
{
    uint32_t split;
    for (split = 0U; split < 3U; split++) {
        if (metrics->splits[split].target_count != d2_expected_split_rows[split] ||
            metrics->splits[split].top1_correct != d2_pre_top1[split] ||
            !d2_format_matches(metrics->splits[split].loss_sum,
                d2_pre_total_loss[split]) ||
            !d2_format_matches(metrics->splits[split].loss_sum /
                (double)metrics->splits[split].target_count,
                d2_pre_average_loss[split]))
            return 0;
    }
    return metrics->evaluated_rows == D2_CACHE_ROWS &&
        metrics->rows_seen[D2_SPLIT_TRAIN] == 111U &&
        metrics->rows_seen[D2_SPLIT_VALIDATION] == 33U &&
        metrics->rows_seen[D2_SPLIT_TEST] == 21U;
}

static void d2_collect_gradient_stats(const stage4_adapter_t *adapter,
    d2_gradient_stats_t *stats)
{
    size_t i;
    memset(stats, 0, sizeof(*stats));
    for (i = 0U; i < adapter->a_count; i++) {
        double value = fabs((double)adapter->grad_a[i]);
        if (adapter->grad_a[i] != 0.0f)
            stats->a_nonzero++;
        if (value > stats->a_max_abs)
            stats->a_max_abs = value;
    }
    for (i = 0U; i < adapter->b_count; i++) {
        double value = fabs((double)adapter->grad_b[i]);
        if (adapter->grad_b[i] != 0.0f)
            stats->b_nonzero++;
        if (value > stats->b_max_abs)
            stats->b_max_abs = value;
    }
}

static void d2_parameter_deltas(const stage4_adapter_t *adapter,
    const float *a_before, const float *b_before,
    double *a_max_abs_delta, double *b_max_abs_delta)
{
    size_t i;
    *a_max_abs_delta = 0.0;
    *b_max_abs_delta = 0.0;
    for (i = 0U; i < adapter->a_count; i++) {
        double delta = fabs((double)adapter->a[i] - (double)a_before[i]);
        if (delta > *a_max_abs_delta)
            *a_max_abs_delta = delta;
    }
    for (i = 0U; i < adapter->b_count; i++) {
        double delta = fabs((double)adapter->b[i] - (double)b_before[i]);
        if (delta > *b_max_abs_delta)
            *b_max_abs_delta = delta;
    }
}

static void d2_moment_nonzero_counts(const stage4_adapter_t *adapter,
    uint32_t *m1_a, uint32_t *m2_a, uint32_t *m1_b, uint32_t *m2_b)
{
    size_t i;
    *m1_a = 0U;
    *m2_a = 0U;
    *m1_b = 0U;
    *m2_b = 0U;
    for (i = 0U; i < adapter->a_count; i++) {
        if (adapter->m1_a[i] != 0.0f)
            (*m1_a)++;
        if (adapter->m2_a[i] != 0.0f)
            (*m2_a)++;
    }
    for (i = 0U; i < adapter->b_count; i++) {
        if (adapter->m1_b[i] != 0.0f)
            (*m1_b)++;
        if (adapter->m2_b[i] != 0.0f)
            (*m2_b)++;
    }
}

#ifndef STAGE4E_D2_CORE_ONLY
static int d2_write_report(const char *path,
    const char *dataset_sha, const char *cache_sha,
    const char *checkpoint_sha, const char *tokenizer_sha,
    const char *d1_report_sha, const stage4_adapter_t *adapter,
    uint32_t gradient_train_rows, uint32_t gradient_validation_rows,
    uint32_t gradient_test_rows, const d2_gradient_stats_t *gradient_stats,
    double norm_before, double norm_after, int clip_applied,
    int clip_consistency, int adapter_a_changed, int adapter_b_changed,
    double a_max_abs_delta, double b_max_abs_delta,
    uint64_t optimizer_steps_before, uint64_t optimizer_steps_after,
    uint32_t m1_a_nonzero, uint32_t m2_a_nonzero,
    uint32_t m1_b_nonzero, uint32_t m2_b_nonzero,
    const d2_metrics_t *pre, const d2_metrics_t *post,
    double train_loss_delta, double validation_loss_delta,
    double test_loss_delta, uint32_t post_zero_correction_rows,
    uint32_t post_nonzero_correction_rows,
    int preupdate_match, int passed)
{
    FILE *report = fopen(path, "wb");
    uint32_t split;
    int ok;
    static const char *split_short[3] = { "train", "validation", "test" };

    if (report == NULL)
        return 0;
    ok = fprintf(report,
        "dataset_sha256=%s\n"
        "cache_sha256=%s\n"
        "base_checkpoint_sha256=%s\n"
        "tokenizer_sha256=%s\n"
        "d1_report_sha256=%s\n"
        "rank=%d\n"
        "scale=%.1f\n"
        "adapter_parameter_count=%lu\n"
        "learning_rate=%.12g\n"
        "beta1=%.12g\n"
        "beta2=%.12g\n"
        "epsilon=%.12g\n"
        "weight_decay=%.12g\n"
        "gradient_clip_threshold=%.1f\n"
        "gradient_train_rows=%u\n"
        "gradient_validation_rows=%u\n"
        "gradient_test_rows=%u\n"
        "grad_A_nonzero_elements=%u\n"
        "grad_A_max_abs=%.12g\n"
        "grad_B_nonzero_elements=%u\n"
        "grad_B_max_abs=%.12g\n"
        "gradient_norm_before_clip=%.12g\n"
        "gradient_norm_after_clip=%.12g\n"
        "gradient_clip_applied=%s\n"
        "gradient_clip_consistency=%s\n"
        "adapter_A_changed=%s\n"
        "adapter_B_changed=%s\n"
        "adapter_A_max_abs_delta=%.12g\n"
        "adapter_B_max_abs_delta=%.12g\n"
        "optimizer_steps_before=%llu\n"
        "optimizer_steps_after=%llu\n"
        "m1_A_nonzero_elements=%u\n"
        "m2_A_nonzero_elements=%u\n"
        "m1_B_nonzero_elements=%u\n"
        "m2_B_nonzero_elements=%u\n"
        "preupdate_d1_metrics_match=%s\n"
        "post_train_total_loss=%.12f\n"
        "post_train_average_loss=%.12f\n"
        "post_train_top1_correct=%llu\n"
        "post_validation_total_loss=%.12f\n"
        "post_validation_average_loss=%.12f\n"
        "post_validation_top1_correct=%llu\n"
        "post_test_total_loss=%.12f\n"
        "post_test_average_loss=%.12f\n"
        "post_test_top1_correct=%llu\n"
        "train_loss_delta=%.12f\n"
        "validation_loss_delta=%.12f\n"
        "test_loss_delta=%.12f\n"
        "postupdate_zero_correction_rows=%u\n"
        "postupdate_nonzero_correction_rows=%u\n"
        "post_update_evaluation_completed=yes\n",
        dataset_sha, cache_sha, checkpoint_sha, tokenizer_sha, d1_report_sha,
        adapter->rank, (double)adapter->scale,
        (unsigned long)(adapter->a_count + adapter->b_count),
        (double)D2_LEARNING_RATE, (double)D2_BETA1, (double)D2_BETA2,
        (double)D2_EPSILON, (double)D2_WEIGHT_DECAY, D2_GRADIENT_CLIP,
        gradient_train_rows, gradient_validation_rows, gradient_test_rows,
        gradient_stats->a_nonzero, gradient_stats->a_max_abs,
        gradient_stats->b_nonzero, gradient_stats->b_max_abs,
        norm_before, norm_after, clip_applied ? "yes" : "no",
        clip_consistency ? "yes" : "no",
        adapter_a_changed ? "yes" : "no", adapter_b_changed ? "yes" : "no",
        a_max_abs_delta, b_max_abs_delta,
        (unsigned long long)optimizer_steps_before,
        (unsigned long long)optimizer_steps_after,
        m1_a_nonzero, m2_a_nonzero, m1_b_nonzero, m2_b_nonzero,
        preupdate_match ? "yes" : "no",
        post->splits[D2_SPLIT_TRAIN].loss_sum,
        post->splits[D2_SPLIT_TRAIN].loss_sum /
            (double)post->splits[D2_SPLIT_TRAIN].target_count,
        (unsigned long long)post->splits[D2_SPLIT_TRAIN].top1_correct,
        post->splits[D2_SPLIT_VALIDATION].loss_sum,
        post->splits[D2_SPLIT_VALIDATION].loss_sum /
            (double)post->splits[D2_SPLIT_VALIDATION].target_count,
        (unsigned long long)post->splits[D2_SPLIT_VALIDATION].top1_correct,
        post->splits[D2_SPLIT_TEST].loss_sum,
        post->splits[D2_SPLIT_TEST].loss_sum /
            (double)post->splits[D2_SPLIT_TEST].target_count,
        (unsigned long long)post->splits[D2_SPLIT_TEST].top1_correct,
        train_loss_delta, validation_loss_delta, test_loss_delta,
        post_zero_correction_rows, post_nonzero_correction_rows) >= 0;
    for (split = 0U; ok && split < 3U; split++) {
        ok = fprintf(report,
            "pre_%s_target_count=%u\n"
            "pre_%s_total_loss=%.12f\n"
            "pre_%s_average_loss=%.12f\n"
            "pre_%s_top1_correct=%llu\n"
            "post_%s_target_count=%u\n",
            split_short[split], pre->splits[split].target_count,
            split_short[split], pre->splits[split].loss_sum,
            split_short[split], pre->splits[split].loss_sum /
                (double)pre->splits[split].target_count,
            split_short[split], (unsigned long long)pre->splits[split].top1_correct,
            split_short[split], post->splits[split].target_count) >= 0;
    }
    if (ok)
        ok = fprintf(report, "d2_one_step_pass=%s\n", passed ? "yes" : "no") >= 0;
    if (fclose(report) != 0)
        ok = 0;
    return ok;
}

static void d2_usage(const char *program)
{
    fprintf(stderr,
        "Usage: %s <checkpoint.bin> -z <tokenizer.bin> -d <dataset> "
        "-c <cache.bin> -1 <d1-report> -r <report.txt>\n",
        program);
}

int main(int argc, char **argv)
{
    const char *checkpoint_path = NULL;
    const char *tokenizer_path = NULL;
    const char *dataset_path = NULL;
    const char *cache_path = NULL;
    const char *d1_report_path = NULL;
    const char *report_path = "stage4e-d2-a4-one-step-report.txt";
    char dataset_sha[65] = "";
    char cache_sha[65] = "";
    char checkpoint_sha[65] = "";
    char tokenizer_sha[65] = "";
    char d1_report_sha[65] = "";
    unsigned char dataset_raw[32];
    unsigned char checkpoint_raw[32];
    unsigned char tokenizer_raw[32];
    unsigned char expected_dataset_raw[32];
    unsigned char expected_checkpoint_raw[32];
    unsigned char expected_tokenizer_raw[32];
    d2_cache_header_t header;
    d2_cache_row_t *rows = NULL;
    d2_metrics_t pre_metrics;
    d2_metrics_t post_metrics;
    d2_gradient_stats_t gradient_stats;
    stage4_adapter_t adapter;
    Transformer transformer;
    float *base_logits = NULL;
    float *final_logits = NULL;
    float *a_before_update = NULL;
    float *b_before_update = NULL;
    FILE *cache_file = NULL;
    uint32_t split_rows[D2_SPLIT_COUNT] = { 0U, 0U, 0U, 0U };
    uint32_t row_index;
    uint32_t gradient_train_rows = 0U;
    uint32_t gradient_validation_rows = 0U;
    uint32_t gradient_test_rows = 0U;
    uint32_t post_zero_correction_rows = 0U;
    uint32_t post_nonzero_correction_rows = 0U;
    uint32_t m1_a_nonzero = 0U;
    uint32_t m2_a_nonzero = 0U;
    uint32_t m1_b_nonzero = 0U;
    uint32_t m2_b_nonzero = 0U;
    uint64_t optimizer_steps_before = 0U;
    uint64_t optimizer_steps_after = 0U;
    uint64_t cache_size = 0U;
    double norm_before = 0.0;
    double norm_after = 0.0;
    double update_norm = 0.0;
    double a_max_abs_delta = 0.0;
    double b_max_abs_delta = 0.0;
    double train_loss_delta = 0.0;
    double validation_loss_delta = 0.0;
    double test_loss_delta = 0.0;
    int clip_applied = 0;
    int clip_consistency = 0;
    int adapter_a_changed = 0;
    int adapter_b_changed = 0;
    int preupdate_match = 0;
    int passed = 0;
    int failed = 0;
    int transformer_ready = 0;
    int adapter_ready = 0;
    int argi;

    memset(&header, 0, sizeof(header));
    memset(&pre_metrics, 0, sizeof(pre_metrics));
    memset(&post_metrics, 0, sizeof(post_metrics));
    memset(&gradient_stats, 0, sizeof(gradient_stats));
    memset(&adapter, 0, sizeof(adapter));
    memset(&transformer, 0, sizeof(transformer));
    memset(dataset_raw, 0, sizeof(dataset_raw));
    memset(checkpoint_raw, 0, sizeof(checkpoint_raw));
    memset(tokenizer_raw, 0, sizeof(tokenizer_raw));
    memset(expected_dataset_raw, 0, sizeof(expected_dataset_raw));
    memset(expected_checkpoint_raw, 0, sizeof(expected_checkpoint_raw));
    memset(expected_tokenizer_raw, 0, sizeof(expected_tokenizer_raw));

    if (argc < 2) {
        d2_usage(argv[0]);
        return EXIT_FAILURE;
    }
    checkpoint_path = argv[1];
    for (argi = 2; argi < argc; argi++) {
        if (strcmp(argv[argi], "-z") == 0 && argi + 1 < argc)
            tokenizer_path = argv[++argi];
        else if (strcmp(argv[argi], "-d") == 0 && argi + 1 < argc)
            dataset_path = argv[++argi];
        else if (strcmp(argv[argi], "-c") == 0 && argi + 1 < argc)
            cache_path = argv[++argi];
        else if (strcmp(argv[argi], "-1") == 0 && argi + 1 < argc)
            d1_report_path = argv[++argi];
        else if (strcmp(argv[argi], "-r") == 0 && argi + 1 < argc)
            report_path = argv[++argi];
        else {
            d2_usage(argv[0]);
            return EXIT_FAILURE;
        }
    }
    if (tokenizer_path == NULL || dataset_path == NULL || cache_path == NULL ||
        d1_report_path == NULL || strcmp(report_path, cache_path) == 0 ||
        strcmp(report_path, dataset_path) == 0 ||
        strcmp(report_path, d1_report_path) == 0 ||
        strcmp(report_path, checkpoint_path) == 0) {
        d2_usage(argv[0]);
        return EXIT_FAILURE;
    }

    if (!stage4_hash_file_sha256_hex(dataset_path, dataset_sha) ||
        !stage4_hash_file_sha256_hex(cache_path, cache_sha) ||
        !stage4_hash_file_sha256_hex(checkpoint_path, checkpoint_sha) ||
        !stage4_hash_file_sha256_hex(tokenizer_path, tokenizer_sha) ||
        !stage4_hash_file_sha256_hex(d1_report_path, d1_report_sha) ||
        !d2_hex_to_raw(dataset_sha, dataset_raw) ||
        !d2_hex_to_raw(checkpoint_sha, checkpoint_raw) ||
        !d2_hex_to_raw(tokenizer_sha, tokenizer_raw) ||
        !d2_hex_to_raw(d2_expected_dataset_sha256, expected_dataset_raw) ||
        !d2_hex_to_raw(d2_expected_checkpoint_sha256, expected_checkpoint_raw) ||
        !d2_hex_to_raw(d2_expected_tokenizer_sha256, expected_tokenizer_raw)) {
        fprintf(stderr, "error: cannot hash or decode required D2 input identities\n");
        failed = 1;
        goto cleanup;
    }
    if (strcmp(dataset_sha, d2_expected_dataset_sha256) != 0 ||
        strcmp(cache_sha, d2_expected_cache_sha256) != 0 ||
        strcmp(checkpoint_sha, d2_expected_checkpoint_sha256) != 0 ||
        strcmp(tokenizer_sha, d2_expected_tokenizer_sha256) != 0 ||
        strcmp(d1_report_sha, d2_expected_d1_report_sha256) != 0) {
        fprintf(stderr, "error: one or more pinned D2 input identities mismatch\n");
        failed = 1;
        goto cleanup;
    }
    if (!d2_float_format_supported() ||
        !stage4_file_size_bytes(cache_path, &cache_size) ||
        cache_size != D2_CACHE_BYTES) {
        fprintf(stderr, "error: A4 cache size or host float format mismatch\n");
        failed = 1;
        goto cleanup;
    }
    cache_file = fopen(cache_path, "rb");
    if (cache_file == NULL || !d2_read_header(cache_file, &header) ||
        !d2_header_valid(&header, dataset_raw, checkpoint_raw, tokenizer_raw)) {
        fprintf(stderr, "error: invalid A4 cache header or identities\n");
        failed = 1;
        goto cleanup;
    }
    rows = calloc(D2_CACHE_ROWS, sizeof(*rows));
    if (rows == NULL) {
        fprintf(stderr, "error: cannot allocate A4 cache rows\n");
        failed = 1;
        goto cleanup;
    }
    for (row_index = 0U; row_index < D2_CACHE_ROWS; row_index++) {
        if (!d2_read_row(cache_file, &rows[row_index]) ||
            rows[row_index].record_index >= D2_CACHE_RECORDS ||
            rows[row_index].target_token_id >= D2_CACHE_VOCAB ||
            rows[row_index].split_id >= D2_SPLIT_COUNT ||
            rows[row_index].split_id == D2_SPLIT_REGRESSION ||
            (row_index > 0U &&
                rows[row_index].record_index < rows[row_index - 1U].record_index) ||
            (row_index > 0U &&
                rows[row_index].record_index == rows[row_index - 1U].record_index &&
                (rows[row_index].split_id != rows[row_index - 1U].split_id ||
                 rows[row_index].input_position <= rows[row_index - 1U].input_position))) {
            fprintf(stderr, "error: invalid or truncated A4 cache row %u\n", row_index);
            failed = 1;
            goto cleanup;
        }
        split_rows[rows[row_index].split_id]++;
    }
    if (fgetc(cache_file) != EOF || ferror(cache_file)) {
        fprintf(stderr, "error: A4 cache has trailing data or read error\n");
        failed = 1;
        goto cleanup;
    }
    if (fclose(cache_file) != 0) {
        cache_file = NULL;
        fprintf(stderr, "error: closing A4 cache failed\n");
        failed = 1;
        goto cleanup;
    }
    cache_file = NULL;
    if (split_rows[D2_SPLIT_TRAIN] != 111U ||
        split_rows[D2_SPLIT_VALIDATION] != 33U ||
        split_rows[D2_SPLIT_TEST] != 21U ||
        split_rows[D2_SPLIT_REGRESSION] != 0U) {
        fprintf(stderr, "error: A4 cache split row counts mismatch\n");
        failed = 1;
        goto cleanup;
    }

    unload_adapter_runtime();
    load_transformer(&transformer, checkpoint_path);
    transformer_ready = 1;
    if (transformer.config.dim != (int)D2_CACHE_DIM ||
        transformer.config.vocab_size != (int)D2_CACHE_VOCAB) {
        fprintf(stderr, "error: checkpoint dimensions disagree with A4 cache\n");
        failed = 1;
        goto cleanup;
    }
    if (!stage4_adapter_init(&adapter, transformer.config.dim,
        transformer.config.vocab_size, D2_RANK, D2_SCALE)) {
        fprintf(stderr, "error: rank-8 adapter initialization failed\n");
        failed = 1;
        goto cleanup;
    }
    adapter_ready = 1;
    stage4_adapter_fill_a_deterministic(&adapter, 1U, 0.01f);
    stage4_adapter_zero_b(&adapter);
    stage4_adapter_zero_moments(&adapter);
    if (adapter.rank != D2_RANK || adapter.scale != D2_SCALE ||
        adapter.a_count != 2304U || adapter.b_count != 256000U) {
        fprintf(stderr, "error: D2 adapter shape or scale mismatch\n");
        failed = 1;
        goto cleanup;
    }
    {
        size_t i;
        for (i = 0U; i < adapter.b_count; i++) {
            if (d2_float_bits(adapter.b[i]) != 0U) {
                fprintf(stderr, "error: initial D2 B is not bitwise zero\n");
                failed = 1;
                goto cleanup;
            }
        }
    }
    base_logits = calloc(D2_CACHE_VOCAB, sizeof(float));
    final_logits = calloc(D2_CACHE_VOCAB, sizeof(float));
    if (base_logits == NULL || final_logits == NULL) {
        fprintf(stderr, "error: cannot allocate D2 logit buffers\n");
        failed = 1;
        goto cleanup;
    }

    if (!d2_evaluate(rows, &adapter, &transformer, &pre_metrics, NULL, NULL)) {
        fprintf(stderr, "error: pre-update cached evaluation failed\n");
        failed = 1;
        goto cleanup;
    }
    preupdate_match = d2_preupdate_matches(&pre_metrics);
    if (!preupdate_match) {
        fprintf(stderr, "error: pre-update metrics do not reproduce D1\n");
        failed = 1;
        goto cleanup;
    }

    stage4_adapter_zero_grad(&adapter);
    optimizer_steps_before = 0U;
    for (row_index = 0U; row_index < D2_CACHE_ROWS; row_index++) {
        const d2_cache_row_t *row = &rows[row_index];
        double gradient_loss;
        if (row->split_id == D2_SPLIT_VALIDATION || row->split_id == D2_SPLIT_TEST)
            continue;
        if (row->split_id != D2_SPLIT_TRAIN) {
            fprintf(stderr, "error: regression row encountered in gradient pass\n");
            failed = 1;
            goto cleanup;
        }
        matmul(base_logits, row->hidden, transformer.weights.wcls,
            (int)D2_CACHE_DIM, (int)D2_CACHE_VOCAB);
        stage4_adapter_forward_logits(&adapter, row->hidden,
            base_logits, final_logits);
        if (!stage4_softmax_loss_and_dlogits(final_logits, adapter.vocab,
            (int)row->target_token_id, &gradient_loss, adapter.dlogits)) {
            fprintf(stderr, "error: train-row loss/gradient calculation failed at row %u\n",
                row_index);
            failed = 1;
            goto cleanup;
        }
        stage4_adapter_backward(&adapter, row->hidden, adapter.dlogits);
        gradient_train_rows++;
    }
    if (gradient_train_rows != 111U || gradient_validation_rows != 0U ||
        gradient_test_rows != 0U) {
        fprintf(stderr, "error: D2 gradient row discipline check failed\n");
        failed = 1;
        goto cleanup;
    }
    stage4_adapter_scale_gradients(&adapter, 1.0f / (float)gradient_train_rows);
    d2_collect_gradient_stats(&adapter, &gradient_stats);
    norm_before = stage4_adapter_gradient_norm(&adapter);
    if (gradient_stats.a_nonzero != 0U || gradient_stats.a_max_abs != 0.0 ||
        gradient_stats.b_nonzero == 0U || !(gradient_stats.b_max_abs > 0.0) ||
        !isfinite(norm_before) || !(norm_before > 0.0)) {
        fprintf(stderr,
            "error: first-step gradient invariant failed; Adam update not performed\n");
        failed = 1;
        goto cleanup;
    }
    clip_applied = norm_before > D2_GRADIENT_CLIP;
    stage4_adapter_clip_gradients(&adapter, D2_GRADIENT_CLIP);
    norm_after = stage4_adapter_gradient_norm(&adapter);
    if (clip_applied)
        clip_consistency = isfinite(norm_after) && fabs(norm_after - D2_GRADIENT_CLIP) <= 0.000001;
    else
        clip_consistency = norm_after == norm_before;
    if (!clip_consistency) {
        fprintf(stderr, "error: gradient clipping norm consistency check failed\n");
        failed = 1;
        goto cleanup;
    }

    a_before_update = malloc(adapter.a_count * sizeof(float));
    b_before_update = malloc(adapter.b_count * sizeof(float));
    if (a_before_update == NULL || b_before_update == NULL) {
        fprintf(stderr, "error: cannot allocate parameter snapshots\n");
        failed = 1;
        goto cleanup;
    }
    memcpy(a_before_update, adapter.a, adapter.a_count * sizeof(float));
    memcpy(b_before_update, adapter.b, adapter.b_count * sizeof(float));
    update_norm = stage4_adapter_adam_step(&adapter,
        (float)D2_LEARNING_RATE, (float)D2_BETA1, (float)D2_BETA2,
        (float)D2_EPSILON, (float)D2_WEIGHT_DECAY, 1U);
    optimizer_steps_after = 1U;
    if (!isfinite(update_norm) || !(update_norm > 0.0) ||
        !stage4_adapter_parameters_are_finite(&adapter)) {
        fprintf(stderr, "error: the single Adam update was not effective\n");
        failed = 1;
        goto cleanup;
    }
    adapter_a_changed = memcmp(a_before_update, adapter.a,
        adapter.a_count * sizeof(float)) != 0;
    adapter_b_changed = memcmp(b_before_update, adapter.b,
        adapter.b_count * sizeof(float)) != 0;
    d2_parameter_deltas(&adapter, a_before_update, b_before_update,
        &a_max_abs_delta, &b_max_abs_delta);
    d2_moment_nonzero_counts(&adapter, &m1_a_nonzero, &m2_a_nonzero,
        &m1_b_nonzero, &m2_b_nonzero);

    if (!d2_evaluate(rows, &adapter, &transformer, &post_metrics,
        &post_zero_correction_rows, &post_nonzero_correction_rows)) {
        fprintf(stderr, "error: post-update evaluation failed\n");
        failed = 1;
        goto cleanup;
    }
    train_loss_delta =
        post_metrics.splits[D2_SPLIT_TRAIN].loss_sum /
            (double)post_metrics.splits[D2_SPLIT_TRAIN].target_count -
        pre_metrics.splits[D2_SPLIT_TRAIN].loss_sum /
            (double)pre_metrics.splits[D2_SPLIT_TRAIN].target_count;
    validation_loss_delta =
        post_metrics.splits[D2_SPLIT_VALIDATION].loss_sum /
            (double)post_metrics.splits[D2_SPLIT_VALIDATION].target_count -
        pre_metrics.splits[D2_SPLIT_VALIDATION].loss_sum /
            (double)pre_metrics.splits[D2_SPLIT_VALIDATION].target_count;
    test_loss_delta =
        post_metrics.splits[D2_SPLIT_TEST].loss_sum /
            (double)post_metrics.splits[D2_SPLIT_TEST].target_count -
        pre_metrics.splits[D2_SPLIT_TEST].loss_sum /
            (double)pre_metrics.splits[D2_SPLIT_TEST].target_count;

    passed = optimizer_steps_before == 0U && optimizer_steps_after == 1U &&
        adapter_a_changed == 0 && adapter_b_changed != 0 &&
        a_max_abs_delta == 0.0 && b_max_abs_delta > 0.0 &&
        gradient_train_rows == 111U && gradient_validation_rows == 0U &&
        gradient_test_rows == 0U && gradient_stats.a_nonzero == 0U &&
        gradient_stats.a_max_abs == 0.0 && gradient_stats.b_nonzero > 0U &&
        gradient_stats.b_max_abs > 0.0 && m1_a_nonzero == 0U &&
        m2_a_nonzero == 0U && m1_b_nonzero > 0U && m2_b_nonzero > 0U &&
        post_metrics.splits[D2_SPLIT_TRAIN].target_count == 111U &&
        post_metrics.splits[D2_SPLIT_VALIDATION].target_count == 33U &&
        post_metrics.splits[D2_SPLIT_TEST].target_count == 21U &&
        post_metrics.splits[D2_SPLIT_TRAIN].loss_sum /
            (double)post_metrics.splits[D2_SPLIT_TRAIN].target_count <
        pre_metrics.splits[D2_SPLIT_TRAIN].loss_sum /
            (double)pre_metrics.splits[D2_SPLIT_TRAIN].target_count &&
        post_nonzero_correction_rows > 0U && clip_consistency;

    if (!d2_write_report(report_path, dataset_sha, cache_sha, checkpoint_sha,
        tokenizer_sha, d1_report_sha, &adapter, gradient_train_rows,
        gradient_validation_rows, gradient_test_rows, &gradient_stats,
        norm_before, norm_after, clip_applied, clip_consistency,
        adapter_a_changed, adapter_b_changed, a_max_abs_delta, b_max_abs_delta,
        optimizer_steps_before, optimizer_steps_after, m1_a_nonzero,
        m2_a_nonzero, m1_b_nonzero, m2_b_nonzero, &pre_metrics, &post_metrics,
        train_loss_delta, validation_loss_delta, test_loss_delta,
        post_zero_correction_rows, post_nonzero_correction_rows,
        preupdate_match, passed)) {
        fprintf(stderr, "error: cannot write D2 report '%s'\n", report_path);
        failed = 1;
        goto cleanup;
    }

    printf("base_checkpoint_identity=PASS\n");
    printf("tokenizer_identity=PASS\n");
    printf("dataset_identity=PASS\n");
    printf("cache_identity=PASS\n");
    printf("d1_report_identity=PASS\n");
    printf("rank=%d\n", adapter.rank);
    printf("scale=%.1f\n", (double)adapter.scale);
    printf("adapter_parameter_count=%lu\n",
        (unsigned long)(adapter.a_count + adapter.b_count));
    printf("gradient_train_rows=%u\n", gradient_train_rows);
    printf("gradient_validation_rows=%u\n", gradient_validation_rows);
    printf("gradient_test_rows=%u\n", gradient_test_rows);
    printf("grad_A_nonzero_elements=%u\n", gradient_stats.a_nonzero);
    printf("grad_A_max_abs=%.12g\n", gradient_stats.a_max_abs);
    printf("grad_B_nonzero_elements=%u\n", gradient_stats.b_nonzero);
    printf("grad_B_max_abs=%.12g\n", gradient_stats.b_max_abs);
    printf("gradient_norm_before_clip=%.12g\n", norm_before);
    printf("gradient_norm_after_clip=%.12g\n", norm_after);
    printf("gradient_clip_applied=%s\n", clip_applied ? "yes" : "no");
    printf("adapter_A_changed=%s\n", adapter_a_changed ? "yes" : "no");
    printf("adapter_B_changed=%s\n", adapter_b_changed ? "yes" : "no");
    printf("adapter_A_max_abs_delta=%.12g\n", a_max_abs_delta);
    printf("adapter_B_max_abs_delta=%.12g\n", b_max_abs_delta);
    printf("optimizer_steps_before=%llu\n",
        (unsigned long long)optimizer_steps_before);
    printf("optimizer_steps_after=%llu\n",
        (unsigned long long)optimizer_steps_after);
    printf("m1_A_nonzero_elements=%u\n", m1_a_nonzero);
    printf("m2_A_nonzero_elements=%u\n", m2_a_nonzero);
    printf("m1_B_nonzero_elements=%u\n", m1_b_nonzero);
    printf("m2_B_nonzero_elements=%u\n", m2_b_nonzero);
    printf("pre_train_average_loss=%.12f\n",
        pre_metrics.splits[D2_SPLIT_TRAIN].loss_sum /
            (double)pre_metrics.splits[D2_SPLIT_TRAIN].target_count);
    printf("post_train_average_loss=%.12f\n",
        post_metrics.splits[D2_SPLIT_TRAIN].loss_sum /
            (double)post_metrics.splits[D2_SPLIT_TRAIN].target_count);
    printf("train_loss_delta=%.12f\n", train_loss_delta);
    printf("pre_validation_average_loss=%.12f\n",
        pre_metrics.splits[D2_SPLIT_VALIDATION].loss_sum /
            (double)pre_metrics.splits[D2_SPLIT_VALIDATION].target_count);
    printf("post_validation_average_loss=%.12f\n",
        post_metrics.splits[D2_SPLIT_VALIDATION].loss_sum /
            (double)post_metrics.splits[D2_SPLIT_VALIDATION].target_count);
    printf("validation_loss_delta=%.12f\n", validation_loss_delta);
    printf("pre_test_average_loss=%.12f\n",
        pre_metrics.splits[D2_SPLIT_TEST].loss_sum /
            (double)pre_metrics.splits[D2_SPLIT_TEST].target_count);
    printf("post_test_average_loss=%.12f\n",
        post_metrics.splits[D2_SPLIT_TEST].loss_sum /
            (double)post_metrics.splits[D2_SPLIT_TEST].target_count);
    printf("test_loss_delta=%.12f\n", test_loss_delta);
    printf("postupdate_nonzero_correction_rows=%u\n",
        post_nonzero_correction_rows);
    printf("preupdate_d1_metrics_match=%s\n", preupdate_match ? "yes" : "no");
    printf("Stage4E-D2-A4=%s\n", passed ? "PASS" : "FAIL");
    if (!passed)
        failed = 1;

cleanup:
    if (cache_file != NULL)
        fclose(cache_file);
    free(rows);
    free(base_logits);
    free(final_logits);
    free(a_before_update);
    free(b_before_update);
    if (adapter_ready)
        stage4_adapter_free(&adapter);
    if (transformer_ready)
        free_transformer(&transformer);
    return failed || !passed ? EXIT_FAILURE : EXIT_SUCCESS;
}
#endif
