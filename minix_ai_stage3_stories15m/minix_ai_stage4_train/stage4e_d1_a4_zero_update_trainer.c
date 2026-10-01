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

#define D1_CACHE_HEADER_SIZE 172U
#define D1_CACHE_ENTRY_SIZE 1168U
#define D1_CACHE_DIM 288U
#define D1_CACHE_VOCAB 32000U
#define D1_CACHE_RECORDS 30U
#define D1_CACHE_ROWS 165U
#define D1_CACHE_BYTES 192892U
#define D1_SPLIT_COUNT 4U
#define D1_RANK 8
#define D1_SCALE 1.0f

enum {
    D1_SPLIT_TRAIN = 0,
    D1_SPLIT_VALIDATION = 1,
    D1_SPLIT_TEST = 2,
    D1_SPLIT_REGRESSION = 3
};

static const unsigned char d1_cache_magic[8] = {
    'S', '4', 'T', 'C', 'A', 'C', 'H', '1'
};
static const char d1_expected_dataset_sha256[] =
    "71744cea4de049e2e192cb2a4494c8a4a68e769b3a7f3b8ed4880433e4583b84";
static const char d1_expected_cache_sha256[] =
    "d6288ad3d128c7182c20088aa57a1bca8bcaa1e13314fb0cdbd60eba14278def";
static const char d1_expected_checkpoint_sha256[] =
    "cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a";
static const char d1_expected_tokenizer_sha256[] =
    "50a52ef822ee9e83de5ce9d0be0a025a773d019437f58b5ff9dcafb063ece361";
static const char d1_expected_b1r_sha256[] =
    "b92264d62bfa603d98b7f00580f1e7e0bf6b2c6cefb8e5ec5db7753e17249b2a";
static const char d1_expected_b2r_sha256[] =
    "1c50c0868358349977d627529ae9f9b04ee155580cbf1e3740d276a5fd968c40";
static const char d1_expected_c2c_sha256[] =
    "e62a40d4fe6dabc3edad4a065e89c78aaadca5496dacaee6da31255e807c35ad";
static const uint32_t d1_expected_split_records[D1_SPLIT_COUNT] = {
    20U, 5U, 5U, 0U
};
static const uint32_t d1_expected_split_targets[D1_SPLIT_COUNT] = {
    111U, 33U, 21U, 0U
};
static const char *d1_split_names[3] = { "train", "validation", "test" };
static const char *d1_expected_total_loss = "1683.218059401455";
static const char *d1_expected_average_loss = "10.201321572130";
static const char *d1_expected_split_total_loss[3] = {
    "1158.750195944566", "300.506654007981", "223.961209448908"
};
static const char *d1_expected_split_average_loss[3] = {
    "10.439190954456", "9.106262242666", "10.664819497567"
};
static const uint32_t d1_expected_split_top1[3] = { 0U, 2U, 0U };
static const uint32_t d1_expected_split_rows[3] = { 111U, 33U, 21U };

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
    uint32_t split_records[D1_SPLIT_COUNT];
    uint32_t split_targets[D1_SPLIT_COUNT];
    unsigned char dataset_sha[32];
    unsigned char checkpoint_sha[32];
    unsigned char tokenizer_sha[32];
} d1_cache_header_t;

typedef struct {
    uint32_t record_index;
    uint32_t input_position;
    uint32_t target_token_id;
    uint32_t split_id;
} d1_cache_entry_t;

typedef struct {
    uint32_t target_count;
    uint64_t top1_correct;
    double loss_sum;
} d1_split_metrics_t;

typedef struct {
    uint32_t evaluated_rows;
    uint32_t train_rows_seen;
    uint32_t validation_rows_seen;
    uint32_t test_rows_seen;
    uint32_t zero_correction_rows;
    uint32_t nonzero_correction_rows;
    uint32_t final_vs_base_logit_bit_mismatches;
    uint64_t overall_top1_correct;
    double overall_loss_sum;
    d1_split_metrics_t splits[3];
} d1_metrics_t;

static uint32_t d1_decode_u32_le(const unsigned char bytes[4])
{
    return (uint32_t)bytes[0] |
        ((uint32_t)bytes[1] << 8) |
        ((uint32_t)bytes[2] << 16) |
        ((uint32_t)bytes[3] << 24);
}

static int d1_read_exact(FILE *file, void *buffer, size_t size)
{
    return size == 0U || fread(buffer, 1, size, file) == size;
}

static int d1_read_u32_le(FILE *file, uint32_t *value)
{
    unsigned char bytes[4];
    if (!d1_read_exact(file, bytes, sizeof(bytes)))
        return 0;
    *value = d1_decode_u32_le(bytes);
    return 1;
}

static int d1_read_f32_le(FILE *file, float *value)
{
    uint32_t bits;
    if (!d1_read_u32_le(file, &bits))
        return 0;
    memcpy(value, &bits, sizeof(bits));
    return 1;
}

static int d1_read_header(FILE *file, d1_cache_header_t *header)
{
    unsigned char magic[8];
    uint32_t i;

    memset(header, 0, sizeof(*header));
    if (!d1_read_exact(file, magic, sizeof(magic)) ||
        memcmp(magic, d1_cache_magic, sizeof(magic)) != 0)
        return 0;
    if (!d1_read_u32_le(file, &header->format_version) ||
        !d1_read_u32_le(file, &header->header_size) ||
        !d1_read_u32_le(file, &header->entry_size) ||
        !d1_read_u32_le(file, &header->float_format) ||
        !d1_read_u32_le(file, &header->byte_order) ||
        !d1_read_u32_le(file, &header->model_dim) ||
        !d1_read_u32_le(file, &header->vocab_size) ||
        !d1_read_u32_le(file, &header->record_count) ||
        !d1_read_u32_le(file, &header->target_count))
        return 0;
    for (i = 0U; i < D1_SPLIT_COUNT; i++) {
        if (!d1_read_u32_le(file, &header->split_records[i]))
            return 0;
    }
    for (i = 0U; i < D1_SPLIT_COUNT; i++) {
        if (!d1_read_u32_le(file, &header->split_targets[i]))
            return 0;
    }
    return d1_read_exact(file, header->dataset_sha, 32U) &&
        d1_read_exact(file, header->checkpoint_sha, 32U) &&
        d1_read_exact(file, header->tokenizer_sha, 32U);
}

static int d1_read_entry(FILE *file, d1_cache_entry_t *entry, float hidden[D1_CACHE_DIM])
{
    uint32_t i;
    if (!d1_read_u32_le(file, &entry->record_index) ||
        !d1_read_u32_le(file, &entry->input_position) ||
        !d1_read_u32_le(file, &entry->target_token_id) ||
        !d1_read_u32_le(file, &entry->split_id))
        return 0;
    for (i = 0U; i < D1_CACHE_DIM; i++) {
        if (!d1_read_f32_le(file, &hidden[i]) || !isfinite(hidden[i]))
            return 0;
    }
    return 1;
}

static int d1_hex_to_raw(const char *hex, unsigned char raw[32])
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

static int d1_float_format_supported(void)
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

static int d1_header_valid(const d1_cache_header_t *header,
    const unsigned char dataset_sha[32],
    const unsigned char checkpoint_sha[32],
    const unsigned char tokenizer_sha[32])
{
    uint32_t i;

    if (header->format_version != 1U ||
        header->header_size != D1_CACHE_HEADER_SIZE ||
        header->entry_size != D1_CACHE_ENTRY_SIZE ||
        header->float_format != 1U || header->byte_order != 1U ||
        header->model_dim != D1_CACHE_DIM ||
        header->vocab_size != D1_CACHE_VOCAB ||
        header->record_count != D1_CACHE_RECORDS ||
        header->target_count != D1_CACHE_ROWS ||
        memcmp(header->dataset_sha, dataset_sha, 32U) != 0 ||
        memcmp(header->checkpoint_sha, checkpoint_sha, 32U) != 0 ||
        memcmp(header->tokenizer_sha, tokenizer_sha, 32U) != 0)
        return 0;
    for (i = 0U; i < D1_SPLIT_COUNT; i++) {
        if (header->split_records[i] != d1_expected_split_records[i] ||
            header->split_targets[i] != d1_expected_split_targets[i])
            return 0;
    }
    return 1;
}

static uint32_t d1_float_bits(float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static int d1_loss_from_logits(const float *logits, int target, double *loss_out)
{
    double max_logit;
    double sum = 0.0;
    double logprob;
    int i;

    if (target < 0 || target >= (int)D1_CACHE_VOCAB || !isfinite(logits[0]))
        return 0;
    max_logit = logits[0];
    for (i = 1; i < (int)D1_CACHE_VOCAB; i++) {
        if (!isfinite(logits[i]))
            return 0;
        if (logits[i] > max_logit)
            max_logit = logits[i];
    }
    for (i = 0; i < (int)D1_CACHE_VOCAB; i++)
        sum += exp((double)logits[i] - max_logit);
    if (!(sum > 0.0) || !isfinite(sum))
        return 0;
    logprob = (double)logits[target] - max_logit - log(sum);
    *loss_out = -logprob;
    return isfinite(*loss_out);
}

static int d1_argmax(const float *logits)
{
    int best = 0;
    int i;
    float best_value = logits[0];

    for (i = 1; i < (int)D1_CACHE_VOCAB; i++) {
        if (logits[i] > best_value) {
            best_value = logits[i];
            best = i;
        }
    }
    return best;
}

static int d1_format_matches(double value, const char *expected)
{
    char formatted[64];
    if (snprintf(formatted, sizeof(formatted), "%.12f", value) < 0)
        return 0;
    return strcmp(formatted, expected) == 0;
}

static int d1_baseline_loss_matches(const d1_metrics_t *metrics)
{
    uint32_t split;
    if (metrics->evaluated_rows != D1_CACHE_ROWS ||
        !d1_format_matches(metrics->overall_loss_sum, d1_expected_total_loss) ||
        !d1_format_matches(metrics->overall_loss_sum / D1_CACHE_ROWS,
            d1_expected_average_loss))
        return 0;
    for (split = 0U; split < 3U; split++) {
        if (metrics->splits[split].target_count != d1_expected_split_rows[split] ||
            !d1_format_matches(metrics->splits[split].loss_sum,
                d1_expected_split_total_loss[split]) ||
            !d1_format_matches(metrics->splits[split].loss_sum /
                (double)metrics->splits[split].target_count,
                d1_expected_split_average_loss[split]))
            return 0;
    }
    return 1;
}

static int d1_baseline_top1_matches(const d1_metrics_t *metrics)
{
    uint32_t split;
    if (metrics->overall_top1_correct != 2U)
        return 0;
    for (split = 0U; split < 3U; split++) {
        if (metrics->splits[split].top1_correct != d1_expected_split_top1[split])
            return 0;
    }
    return 1;
}

static void d1_snapshot_optimizer_state(const stage4_adapter_t *adapter,
    float *snapshot)
{
    size_t offset = 0U;
    memcpy(snapshot + offset, adapter->m1_a, adapter->a_count * sizeof(float));
    offset += adapter->a_count;
    memcpy(snapshot + offset, adapter->m1_b, adapter->b_count * sizeof(float));
    offset += adapter->b_count;
    memcpy(snapshot + offset, adapter->m2_a, adapter->a_count * sizeof(float));
    offset += adapter->a_count;
    memcpy(snapshot + offset, adapter->m2_b, adapter->b_count * sizeof(float));
}

static int d1_optimizer_state_matches(const stage4_adapter_t *adapter,
    const float *snapshot)
{
    size_t offset = 0U;
    if (memcmp(snapshot + offset, adapter->m1_a,
        adapter->a_count * sizeof(float)) != 0)
        return 0;
    offset += adapter->a_count;
    if (memcmp(snapshot + offset, adapter->m1_b,
        adapter->b_count * sizeof(float)) != 0)
        return 0;
    offset += adapter->b_count;
    if (memcmp(snapshot + offset, adapter->m2_a,
        adapter->a_count * sizeof(float)) != 0)
        return 0;
    offset += adapter->a_count;
    return memcmp(snapshot + offset, adapter->m2_b,
        adapter->b_count * sizeof(float)) == 0;
}

static int d1_write_report(const char *path,
    const char *dataset_sha,
    const char *cache_sha,
    const char *checkpoint_sha,
    const char *tokenizer_sha,
    const char *b1r_sha,
    const char *b2r_sha,
    const char *c2c_sha,
    const stage4_adapter_t *adapter,
    const d1_metrics_t *metrics,
    int adapter_a_changed,
    int adapter_b_changed,
    uint64_t optimizer_steps_before,
    uint64_t optimizer_steps_after,
    int optimizer_state_changed,
    int b1r_loss_match,
    int b2r_top1_match,
    int passed)
{
    FILE *report;
    uint32_t split;
    int ok;

    report = fopen(path, "wb");
    if (report == NULL)
        return 0;
    ok = fprintf(report,
        "dataset_sha256=%s\n"
        "cache_sha256=%s\n"
        "base_checkpoint_sha256=%s\n"
        "tokenizer_sha256=%s\n"
        "b1r_a4_report_sha256=%s\n"
        "b2r_a4_report_sha256=%s\n"
        "c2c_a4_report_sha256=%s\n"
        "rank=%d\n"
        "scale=%.1f\n"
        "adapter_parameter_count=%lu\n"
        "adapter_A_changed=%s\n"
        "adapter_B_changed=%s\n"
        "optimizer_steps_before=%llu\n"
        "optimizer_steps_after=%llu\n"
        "optimizer_state_changed=%s\n"
        "evaluated_rows=%u\n"
        "train_rows_seen=%u\n"
        "validation_rows_seen=%u\n"
        "test_rows_seen=%u\n"
        "zero_correction_rows=%u\n"
        "nonzero_correction_rows=%u\n"
        "final_vs_base_logit_bit_mismatches=%u\n"
        "overall_target_count=%u\n"
        "overall_total_loss=%.12f\n"
        "overall_average_loss=%.12f\n"
        "overall_top1_correct=%llu\n"
        "validation_baseline_loss=9.106262242666\n"
        "validation_required_relative_improvement=0.05\n"
        "validation_acceptance_loss_max=8.650949130533\n"
        "validation_threshold_applied=no\n",
        dataset_sha, cache_sha, checkpoint_sha, tokenizer_sha, b1r_sha, b2r_sha,
        c2c_sha, adapter->rank, (double)adapter->scale,
        (unsigned long)(adapter->a_count + adapter->b_count),
        adapter_a_changed ? "yes" : "no", adapter_b_changed ? "yes" : "no",
        (unsigned long long)optimizer_steps_before,
        (unsigned long long)optimizer_steps_after,
        optimizer_state_changed ? "yes" : "no",
        metrics->evaluated_rows, metrics->train_rows_seen,
        metrics->validation_rows_seen, metrics->test_rows_seen,
        metrics->zero_correction_rows, metrics->nonzero_correction_rows,
        metrics->final_vs_base_logit_bit_mismatches,
        metrics->evaluated_rows, metrics->overall_loss_sum,
        metrics->overall_loss_sum / (double)metrics->evaluated_rows,
        (unsigned long long)metrics->overall_top1_correct) >= 0;
    for (split = 0U; ok && split < 3U; split++) {
        ok = fprintf(report,
            "%s_target_count=%u\n"
            "%s_total_loss=%.12f\n"
            "%s_average_loss=%.12f\n"
            "%s_top1_correct=%llu\n",
            d1_split_names[split], metrics->splits[split].target_count,
            d1_split_names[split], metrics->splits[split].loss_sum,
            d1_split_names[split], metrics->splits[split].target_count == 0U ? 0.0 :
                metrics->splits[split].loss_sum /
                    (double)metrics->splits[split].target_count,
            d1_split_names[split],
            (unsigned long long)metrics->splits[split].top1_correct) >= 0;
    }
    if (ok) {
        ok = fprintf(report,
            "b1r_a4_loss_metrics_match=%s\n"
            "b2r_a4_top1_metrics_match=%s\n"
            "zero_update_validation_pass=%s\n",
            b1r_loss_match ? "yes" : "no",
            b2r_top1_match ? "yes" : "no",
            passed ? "yes" : "no") >= 0;
    }
    if (fclose(report) != 0)
        ok = 0;
    return ok;
}

static void d1_usage(const char *program)
{
    fprintf(stderr,
        "Usage: %s <checkpoint.bin> -z <tokenizer.bin> -d <dataset> "
        "-c <cache.bin> -1 <b1r-report> -2 <b2r-report> "
        "-3 <c2c-report> -r <report.txt>\n",
        program);
}

int main(int argc, char **argv)
{
    const char *checkpoint_path = NULL;
    const char *tokenizer_path = NULL;
    const char *dataset_path = NULL;
    const char *cache_path = NULL;
    const char *b1r_path = NULL;
    const char *b2r_path = NULL;
    const char *c2c_path = NULL;
    const char *report_path = "stage4e-d1-a4-zero-update-trainer-report.txt";
    char dataset_sha[65] = "";
    char cache_sha[65] = "";
    char checkpoint_sha[65] = "";
    char tokenizer_sha[65] = "";
    char b1r_sha[65] = "";
    char b2r_sha[65] = "";
    char c2c_sha[65] = "";
    unsigned char dataset_raw[32];
    unsigned char checkpoint_raw[32];
    unsigned char tokenizer_raw[32];
    unsigned char expected_dataset_raw[32];
    unsigned char expected_checkpoint_raw[32];
    unsigned char expected_tokenizer_raw[32];
    d1_cache_header_t header;
    d1_cache_entry_t entry;
    d1_metrics_t metrics;
    stage4_adapter_t adapter;
    Transformer transformer;
    FILE *cache_file = NULL;
    float *hidden = NULL;
    float *base_logits = NULL;
    float *final_logits = NULL;
    float *a_snapshot = NULL;
    float *b_snapshot = NULL;
    float *optimizer_snapshot = NULL;
    size_t optimizer_snapshot_count;
    uint32_t split_rows[D1_SPLIT_COUNT] = { 0U, 0U, 0U, 0U };
    uint32_t row_index;
    uint32_t previous_record = 0U;
    uint32_t previous_position = 0U;
    uint32_t current_record = 0U;
    uint32_t current_split = 0U;
    uint32_t current_record_rows = 0U;
    uint64_t current_record_top1 = 0U;
    double current_record_loss = 0.0;
    uint64_t optimizer_steps_before = 0U;
    uint64_t optimizer_steps_after = 0U;
    uint64_t cache_size = 0U;
    int have_current_record = 0;
    int transformer_ready = 0;
    int adapter_ready = 0;
    int adapter_a_changed = 1;
    int adapter_b_changed = 1;
    int optimizer_state_changed = 1;
    int b1r_loss_match = 0;
    int b2r_top1_match = 0;
    int passed = 0;
    int failed = 0;
    int argi;

    memset(&header, 0, sizeof(header));
    memset(&entry, 0, sizeof(entry));
    memset(&metrics, 0, sizeof(metrics));
    memset(&adapter, 0, sizeof(adapter));
    memset(&transformer, 0, sizeof(transformer));
    memset(dataset_raw, 0, sizeof(dataset_raw));
    memset(checkpoint_raw, 0, sizeof(checkpoint_raw));
    memset(tokenizer_raw, 0, sizeof(tokenizer_raw));
    memset(expected_dataset_raw, 0, sizeof(expected_dataset_raw));
    memset(expected_checkpoint_raw, 0, sizeof(expected_checkpoint_raw));
    memset(expected_tokenizer_raw, 0, sizeof(expected_tokenizer_raw));

    if (argc < 2) {
        d1_usage(argv[0]);
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
            b1r_path = argv[++argi];
        else if (strcmp(argv[argi], "-2") == 0 && argi + 1 < argc)
            b2r_path = argv[++argi];
        else if (strcmp(argv[argi], "-3") == 0 && argi + 1 < argc)
            c2c_path = argv[++argi];
        else if (strcmp(argv[argi], "-r") == 0 && argi + 1 < argc)
            report_path = argv[++argi];
        else {
            d1_usage(argv[0]);
            return EXIT_FAILURE;
        }
    }
    if (tokenizer_path == NULL || dataset_path == NULL || cache_path == NULL ||
        b1r_path == NULL || b2r_path == NULL || c2c_path == NULL ||
        strcmp(report_path, cache_path) == 0 || strcmp(report_path, dataset_path) == 0 ||
        strcmp(report_path, b1r_path) == 0 || strcmp(report_path, b2r_path) == 0 ||
        strcmp(report_path, c2c_path) == 0 || strcmp(report_path, checkpoint_path) == 0) {
        d1_usage(argv[0]);
        return EXIT_FAILURE;
    }

    if (!stage4_hash_file_sha256_hex(dataset_path, dataset_sha) ||
        !stage4_hash_file_sha256_hex(cache_path, cache_sha) ||
        !stage4_hash_file_sha256_hex(checkpoint_path, checkpoint_sha) ||
        !stage4_hash_file_sha256_hex(tokenizer_path, tokenizer_sha) ||
        !stage4_hash_file_sha256_hex(b1r_path, b1r_sha) ||
        !stage4_hash_file_sha256_hex(b2r_path, b2r_sha) ||
        !stage4_hash_file_sha256_hex(c2c_path, c2c_sha) ||
        !d1_hex_to_raw(dataset_sha, dataset_raw) ||
        !d1_hex_to_raw(checkpoint_sha, checkpoint_raw) ||
        !d1_hex_to_raw(tokenizer_sha, tokenizer_raw) ||
        !d1_hex_to_raw(d1_expected_dataset_sha256, expected_dataset_raw) ||
        !d1_hex_to_raw(d1_expected_checkpoint_sha256, expected_checkpoint_raw) ||
        !d1_hex_to_raw(d1_expected_tokenizer_sha256, expected_tokenizer_raw)) {
        fprintf(stderr, "error: cannot hash or decode required input identities\n");
        failed = 1;
        goto cleanup;
    }
    if (strcmp(dataset_sha, d1_expected_dataset_sha256) != 0 ||
        strcmp(cache_sha, d1_expected_cache_sha256) != 0 ||
        strcmp(checkpoint_sha, d1_expected_checkpoint_sha256) != 0 ||
        strcmp(tokenizer_sha, d1_expected_tokenizer_sha256) != 0 ||
        strcmp(b1r_sha, d1_expected_b1r_sha256) != 0 ||
        strcmp(b2r_sha, d1_expected_b2r_sha256) != 0 ||
        strcmp(c2c_sha, d1_expected_c2c_sha256) != 0) {
        fprintf(stderr, "error: one or more pinned D1 input identities mismatch\n");
        failed = 1;
        goto cleanup;
    }
    if (!d1_float_format_supported() ||
        !stage4_file_size_bytes(cache_path, &cache_size) ||
        cache_size != D1_CACHE_BYTES) {
        fprintf(stderr, "error: cache size or host float format mismatch\n");
        failed = 1;
        goto cleanup;
    }
    cache_file = fopen(cache_path, "rb");
    if (cache_file == NULL || !d1_read_header(cache_file, &header) ||
        !d1_header_valid(&header, dataset_raw, checkpoint_raw, tokenizer_raw)) {
        fprintf(stderr, "error: invalid A4 cache header or cache identities\n");
        failed = 1;
        goto cleanup;
    }

    unload_adapter_runtime();
    load_transformer(&transformer, checkpoint_path);
    transformer_ready = 1;
    if (transformer.config.dim != (int)D1_CACHE_DIM ||
        transformer.config.vocab_size != (int)D1_CACHE_VOCAB) {
        fprintf(stderr, "error: checkpoint dimensions disagree with A4 cache\n");
        failed = 1;
        goto cleanup;
    }
    if (!stage4_adapter_init(&adapter, transformer.config.dim,
        transformer.config.vocab_size, D1_RANK, D1_SCALE)) {
        fprintf(stderr, "error: cannot initialize rank-8 adapter\n");
        failed = 1;
        goto cleanup;
    }
    adapter_ready = 1;
    stage4_adapter_fill_a_deterministic(&adapter, 1U, 0.01f);
    stage4_adapter_zero_b(&adapter);
    {
        size_t weight_index;
        for (weight_index = 0U; weight_index < adapter.b_count; weight_index++) {
            if (d1_float_bits(adapter.b[weight_index]) != 0U) {
                fprintf(stderr, "error: initialized adapter B is not all zero\n");
                failed = 1;
                goto cleanup;
            }
        }
    }
    if (adapter.rank != D1_RANK || adapter.scale != D1_SCALE ||
        adapter.a_count != 2304U || adapter.b_count != 256000U ||
        adapter.a_count + adapter.b_count != 258304U) {
        fprintf(stderr, "error: adapter parameter count mismatch\n");
        failed = 1;
        goto cleanup;
    }

    hidden = calloc(D1_CACHE_DIM, sizeof(float));
    base_logits = calloc(D1_CACHE_VOCAB, sizeof(float));
    final_logits = calloc(D1_CACHE_VOCAB, sizeof(float));
    a_snapshot = malloc(adapter.a_count * sizeof(float));
    b_snapshot = malloc(adapter.b_count * sizeof(float));
    optimizer_snapshot_count = 2U * adapter.a_count + 2U * adapter.b_count;
    optimizer_snapshot = malloc(optimizer_snapshot_count * sizeof(float));
    if (hidden == NULL || base_logits == NULL || final_logits == NULL ||
        a_snapshot == NULL || b_snapshot == NULL || optimizer_snapshot == NULL) {
        fprintf(stderr, "error: cannot allocate D1 evaluation buffers\n");
        failed = 1;
        goto cleanup;
    }
    memcpy(a_snapshot, adapter.a, adapter.a_count * sizeof(float));
    memcpy(b_snapshot, adapter.b, adapter.b_count * sizeof(float));
    d1_snapshot_optimizer_state(&adapter, optimizer_snapshot);
    optimizer_steps_before = 0U;

    for (row_index = 0U; row_index < D1_CACHE_ROWS; row_index++) {
        int row_zero_correction = 1;
        int live_top1;
        double loss;
        uint32_t vocab_index;
        uint32_t rank_index;

        if (!d1_read_entry(cache_file, &entry, hidden)) {
            fprintf(stderr, "error: truncated or non-finite cache row %u\n", row_index);
            failed = 1;
            goto cleanup;
        }
        if (entry.record_index >= D1_CACHE_RECORDS ||
            entry.target_token_id >= D1_CACHE_VOCAB ||
            entry.split_id >= D1_SPLIT_COUNT ||
            entry.split_id == D1_SPLIT_REGRESSION ||
            (have_current_record && entry.record_index < previous_record) ||
            (have_current_record && entry.record_index == previous_record &&
                (entry.split_id != current_split ||
                 entry.input_position <= previous_position))) {
            fprintf(stderr, "error: invalid cache row metadata at row %u\n", row_index);
            failed = 1;
            goto cleanup;
        }
        if (!have_current_record || entry.record_index != current_record) {
            if (have_current_record) {
                d1_split_metrics_t *split_metrics = &metrics.splits[current_split];
                split_metrics->target_count += current_record_rows;
                split_metrics->loss_sum += current_record_loss;
                split_metrics->top1_correct += current_record_top1;
                metrics.overall_loss_sum += current_record_loss;
                metrics.overall_top1_correct += current_record_top1;
            }
            current_record = entry.record_index;
            current_split = entry.split_id;
            current_record_rows = 0U;
            current_record_top1 = 0U;
            current_record_loss = 0.0;
            have_current_record = 1;
        }
        previous_record = entry.record_index;
        previous_position = entry.input_position;

        matmul(base_logits, hidden, transformer.weights.wcls,
            (int)D1_CACHE_DIM, (int)D1_CACHE_VOCAB);
        stage4_adapter_forward_logits(&adapter, hidden, base_logits, final_logits);
        for (vocab_index = 0U; vocab_index < D1_CACHE_VOCAB; vocab_index++) {
            double correction = 0.0;
            uint32_t base_bits;
            uint32_t final_bits;
            for (rank_index = 0U; rank_index < D1_RANK; rank_index++) {
                correction += (double)adapter.b[(size_t)vocab_index * D1_RANK +
                    rank_index] * (double)adapter.u[rank_index];
            }
            if (correction != 0.0 || adapter.scale * (float)correction != 0.0f)
                row_zero_correction = 0;
            base_bits = d1_float_bits(base_logits[vocab_index]);
            final_bits = d1_float_bits(final_logits[vocab_index]);
            if (base_bits != final_bits)
                metrics.final_vs_base_logit_bit_mismatches++;
        }
        if (row_zero_correction)
            metrics.zero_correction_rows++;
        else
            metrics.nonzero_correction_rows++;

        if (!d1_loss_from_logits(final_logits, (int)entry.target_token_id, &loss)) {
            fprintf(stderr, "error: non-finite target loss at cache row %u\n", row_index);
            failed = 1;
            goto cleanup;
        }
        live_top1 = d1_argmax(final_logits);
        current_record_loss += loss;
        current_record_rows++;
        if (live_top1 == (int)entry.target_token_id)
            current_record_top1++;
        metrics.evaluated_rows++;
        split_rows[entry.split_id]++;
    }
    if (have_current_record) {
        d1_split_metrics_t *split_metrics = &metrics.splits[current_split];
        split_metrics->target_count += current_record_rows;
        split_metrics->loss_sum += current_record_loss;
        split_metrics->top1_correct += current_record_top1;
        metrics.overall_loss_sum += current_record_loss;
        metrics.overall_top1_correct += current_record_top1;
    }
    if (fgetc(cache_file) != EOF || ferror(cache_file)) {
        fprintf(stderr, "error: cache has trailing data or read error\n");
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
    if (!stage4_file_size_bytes(cache_path, &cache_size) ||
        cache_size != D1_CACHE_BYTES) {
        fprintf(stderr, "error: A4 cache size changed during evaluation\n");
        failed = 1;
        goto cleanup;
    }

    metrics.train_rows_seen = split_rows[0];
    metrics.validation_rows_seen = split_rows[1];
    metrics.test_rows_seen = split_rows[2];
    adapter_a_changed = memcmp(a_snapshot, adapter.a,
        adapter.a_count * sizeof(float)) != 0;
    adapter_b_changed = memcmp(b_snapshot, adapter.b,
        adapter.b_count * sizeof(float)) != 0;
    optimizer_state_changed = !d1_optimizer_state_matches(&adapter, optimizer_snapshot);
    optimizer_steps_after = 0U;

    b1r_loss_match = metrics.evaluated_rows == D1_CACHE_ROWS &&
        metrics.train_rows_seen == d1_expected_split_rows[0] &&
        metrics.validation_rows_seen == d1_expected_split_rows[1] &&
        metrics.test_rows_seen == d1_expected_split_rows[2] &&
        d1_baseline_loss_matches(&metrics);
    b2r_top1_match = metrics.evaluated_rows == D1_CACHE_ROWS &&
        d1_baseline_top1_matches(&metrics);
    passed = !adapter_a_changed && !adapter_b_changed && !optimizer_state_changed &&
        optimizer_steps_before == 0U && optimizer_steps_after == 0U &&
        adapter.rank == D1_RANK && adapter.scale == D1_SCALE &&
        adapter.a_count == 2304U && adapter.b_count == 256000U &&
        metrics.evaluated_rows == D1_CACHE_ROWS &&
        metrics.train_rows_seen == 111U && metrics.validation_rows_seen == 33U &&
        metrics.test_rows_seen == 21U &&
        metrics.zero_correction_rows == D1_CACHE_ROWS &&
        metrics.nonzero_correction_rows == 0U &&
        metrics.final_vs_base_logit_bit_mismatches == 0U &&
        b1r_loss_match && b2r_top1_match;

    if (!d1_write_report(report_path, dataset_sha, cache_sha, checkpoint_sha,
        tokenizer_sha, b1r_sha, b2r_sha, c2c_sha, &adapter, &metrics,
        adapter_a_changed, adapter_b_changed, optimizer_steps_before,
        optimizer_steps_after, optimizer_state_changed, b1r_loss_match,
        b2r_top1_match, passed)) {
        fprintf(stderr, "error: cannot write D1 report '%s'\n", report_path);
        failed = 1;
        goto cleanup;
    }
    printf("base_checkpoint_identity=PASS\n");
    printf("tokenizer_identity=PASS\n");
    printf("dataset_identity=PASS\n");
    printf("cache_identity=PASS\n");
    printf("b1r_a4_report_identity=PASS\n");
    printf("b2r_a4_report_identity=PASS\n");
    printf("c2c_a4_report_identity=PASS\n");
    printf("rank=%d\n", adapter.rank);
    printf("scale=%.1f\n", (double)adapter.scale);
    printf("adapter_parameter_count=%lu\n",
        (unsigned long)(adapter.a_count + adapter.b_count));
    printf("evaluated_rows=%u\n", metrics.evaluated_rows);
    printf("train_rows_seen=%u\n", metrics.train_rows_seen);
    printf("validation_rows_seen=%u\n", metrics.validation_rows_seen);
    printf("test_rows_seen=%u\n", metrics.test_rows_seen);
    printf("zero_correction_rows=%u\n", metrics.zero_correction_rows);
    printf("nonzero_correction_rows=%u\n", metrics.nonzero_correction_rows);
    printf("final_vs_base_logit_bit_mismatches=%u\n",
        metrics.final_vs_base_logit_bit_mismatches);
    printf("adapter_A_changed=%s\n", adapter_a_changed ? "yes" : "no");
    printf("adapter_B_changed=%s\n", adapter_b_changed ? "yes" : "no");
    printf("optimizer_steps_before=%llu\n",
        (unsigned long long)optimizer_steps_before);
    printf("optimizer_steps_after=%llu\n",
        (unsigned long long)optimizer_steps_after);
    printf("overall_average_loss=%.12f\n",
        metrics.overall_loss_sum / (double)metrics.evaluated_rows);
    printf("train_average_loss=%.12f\n",
        metrics.splits[0].loss_sum / (double)metrics.splits[0].target_count);
    printf("validation_average_loss=%.12f\n",
        metrics.splits[1].loss_sum / (double)metrics.splits[1].target_count);
    printf("test_average_loss=%.12f\n",
        metrics.splits[2].loss_sum / (double)metrics.splits[2].target_count);
    printf("overall_top1_correct=%llu\n",
        (unsigned long long)metrics.overall_top1_correct);
    printf("train_top1_correct=%llu\n",
        (unsigned long long)metrics.splits[0].top1_correct);
    printf("validation_top1_correct=%llu\n",
        (unsigned long long)metrics.splits[1].top1_correct);
    printf("test_top1_correct=%llu\n",
        (unsigned long long)metrics.splits[2].top1_correct);
    printf("b1r_a4_loss_metrics_match=%s\n", b1r_loss_match ? "yes" : "no");
    printf("b2r_a4_top1_metrics_match=%s\n", b2r_top1_match ? "yes" : "no");
    printf("Stage4E-D1-A4=%s\n", passed ? "PASS" : "FAIL");
    if (!passed)
        failed = 1;

cleanup:
    if (cache_file != NULL)
        fclose(cache_file);
    free(hidden);
    free(base_logits);
    free(final_logits);
    free(a_snapshot);
    free(b_snapshot);
    free(optimizer_snapshot);
    if (adapter_ready)
        stage4_adapter_free(&adapter);
    if (transformer_ready)
        free_transformer(&transformer);
    return failed || !passed ? EXIT_FAILURE : EXIT_SUCCESS;
}
