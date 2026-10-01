#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STAGE4E_GENERATION_EMBEDDED
#include "stage4e_baseline_unadapted_generation.c"
#undef STAGE4E_GENERATION_EMBEDDED

#define CACHE_FORMAT_VERSION 1U
#define CACHE_HEADER_SIZE 172U
#define CACHE_ENTRY_SIZE 1168U
#define CACHE_DIM 288U
#define CACHE_VOCAB 32000U
#define CACHE_RECORDS 30U
#ifdef STAGE4E_CACHE_EQUIVALENCE_A4
#define CACHE_TARGETS 165U
#define CACHE_FILE_SIZE 192892U
#define HIDDEN_FLOAT_COUNT 47520U
#define LOGIT_FLOAT_COUNT 5280000U
#else
#define CACHE_TARGETS 156U
#define CACHE_FILE_SIZE 182380U
#define HIDDEN_FLOAT_COUNT 44928U
#define LOGIT_FLOAT_COUNT 4992000U
#endif
#define CACHE_SPLIT_COUNT 4U
#define CHECKPOINT_HEADER_BYTES 28U
#define LOSS_COMPARE_TOLERANCE 0.0000000000005

typedef struct {
    uint32_t record_index;
    uint32_t input_position;
    uint32_t target_token_id;
    uint32_t split_id;
} expected_entry_t;

typedef struct {
    uint32_t record_index;
    uint32_t input_position;
    uint32_t target_token_id;
    uint32_t split_id;
} cache_entry_t;

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
    uint32_t split_records[CACHE_SPLIT_COUNT];
    uint32_t split_targets[CACHE_SPLIT_COUNT];
    unsigned char dataset_sha[32];
    unsigned char checkpoint_sha[32];
    unsigned char tokenizer_sha[32];
} cache_header_t;

typedef struct {
    uint32_t target_count;
    uint64_t top1_correct;
    double loss_sum;
} split_metrics_t;

typedef struct {
    uint32_t metadata_mismatches;
    uint32_t hidden_bitwise_mismatches;
    uint32_t logit_bitwise_mismatches;
    uint32_t per_position_loss_mismatches;
    uint32_t top1_token_mismatches;
    uint32_t top1_correctness_mismatches;
    uint32_t compared_target_positions;
    uint32_t hidden_float_count;
    uint32_t logit_float_count;
    uint32_t hidden_rows_compared;
    uint32_t logit_rows_compared;
    uint32_t hidden_rows_with_mismatch;
    uint32_t logit_rows_with_mismatch;
    uint32_t expected_cache_rows;
    uint32_t consumed_cache_rows;
    uint32_t remaining_cache_rows;
    uint32_t metadata_record_index_mismatches;
    uint32_t metadata_input_position_mismatches;
    uint32_t metadata_target_token_mismatches;
    uint32_t metadata_split_id_mismatches;
    uint32_t record0_forward_calls_before_position_15;
    uint32_t record0_forward_calls_before_first_target;
    uint64_t record0_kv_checksum_before_position_15;
    int record0_full_context_check;
    double hidden_max_abs_difference;
    double logit_max_abs_difference;
    double maximum_position_loss_difference;
    int first_hidden_mismatch;
    uint32_t first_hidden_record;
    uint32_t first_hidden_position;
    uint32_t first_hidden_dimension;
    uint32_t first_hidden_live_bits;
    uint32_t first_hidden_cache_bits;
    float first_hidden_live_value;
    float first_hidden_cache_value;
    int first_logit_mismatch;
    uint32_t first_logit_record;
    uint32_t first_logit_position;
    uint32_t first_logit_vocab;
    uint32_t first_logit_live_bits;
    uint32_t first_logit_cache_bits;
    float first_logit_live_value;
    float first_logit_cache_value;
} equivalence_stats_t;

static const unsigned char cache_magic[8] = { 'S', '4', 'T', 'C', 'A', 'C', 'H', '1' };
static const char expected_dataset_sha256[] =
#ifdef STAGE4E_CACHE_EQUIVALENCE_A4
    "71744cea4de049e2e192cb2a4494c8a4a68e769b3a7f3b8ed4880433e4583b84";
#else
    "b8aa6dc1dbfad7ed144e19a351850189956f79571b05cb70324d21876db0ea78";
#endif
static const char expected_checkpoint_sha256[] =
    "cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a";
static const char expected_tokenizer_sha256[] =
    "50a52ef822ee9e83de5ce9d0be0a025a773d019437f58b5ff9dcafb063ece361";
#ifdef STAGE4E_CACHE_EQUIVALENCE_A4
static const char expected_cache_sha256[] =
    "d6288ad3d128c7182c20088aa57a1bca8bcaa1e13314fb0cdbd60eba14278def";
static const char expected_b1r_report_sha256[] =
    "b92264d62bfa603d98b7f00580f1e7e0bf6b2c6cefb8e5ec5db7753e17249b2a";
static const char expected_b2r_report_sha256[] =
    "1c50c0868358349977d627529ae9f9b04ee155580cbf1e3740d276a5fd968c40";
#endif
static const uint32_t expected_split_records[CACHE_SPLIT_COUNT] = { 20U, 5U, 5U, 0U };
#ifdef STAGE4E_CACHE_EQUIVALENCE_A4
static const uint32_t expected_split_targets[CACHE_SPLIT_COUNT] = { 111U, 33U, 21U, 0U };
static const uint32_t accepted_top1_correct[CACHE_SPLIT_COUNT] = { 0U, 2U, 0U, 0U };
static const uint32_t accepted_top1_count[CACHE_SPLIT_COUNT] = { 111U, 33U, 21U, 0U };
static const double accepted_split_loss[CACHE_SPLIT_COUNT] = {
    1158.750195944566, 300.506654007981, 223.961209448908, 0.0
};
static const double accepted_average_loss[CACHE_SPLIT_COUNT] = {
    10.439190954456, 9.106262242666, 10.664819497567, 0.0
};
static const double accepted_total_loss = 1683.218059401455;
static const double accepted_overall_average_loss = 10.201321572130;
static const uint32_t accepted_overall_top1_correct = 2U;
static const uint32_t accepted_overall_target_count = 165U;
#else
static const uint32_t expected_split_targets[CACHE_SPLIT_COUNT] = { 111U, 24U, 21U, 0U };
static const uint32_t accepted_top1_correct[CACHE_SPLIT_COUNT] = { 1U, 4U, 0U, 0U };
static const uint32_t accepted_top1_count[CACHE_SPLIT_COUNT] = { 111U, 24U, 21U, 0U };
static const double accepted_average_loss[CACHE_SPLIT_COUNT] = {
    11.932673309778, 9.153772893016, 12.168615024563, 0.0
};
static const double accepted_total_loss = 1799.758202333604;
static const double accepted_overall_average_loss = 11.536911553421;
static const uint32_t accepted_overall_top1_correct = 5U;
static const uint32_t accepted_overall_target_count = 156U;
#endif

static int equivalence_read_exact(FILE *file, void *buffer, size_t size)
{
    return size == 0U || fread(buffer, 1, size, file) == size;
}

static uint32_t equivalence_decode_u32_le(const unsigned char bytes[4])
{
    return (uint32_t)bytes[0] |
        ((uint32_t)bytes[1] << 8) |
        ((uint32_t)bytes[2] << 16) |
        ((uint32_t)bytes[3] << 24);
}

static int equivalence_read_u32_le(FILE *file, uint32_t *value)
{
    unsigned char bytes[4];
    if (!equivalence_read_exact(file, bytes, sizeof(bytes)))
        return 0;
    *value = equivalence_decode_u32_le(bytes);
    return 1;
}

static int equivalence_read_f32_le(FILE *file, float *value, uint32_t *bits_out)
{
    uint32_t bits;
    if (!equivalence_read_u32_le(file, &bits))
        return 0;
    memcpy(value, &bits, sizeof(bits));
    if (bits_out != NULL)
        *bits_out = bits;
    return 1;
}

static int equivalence_hex_value(char ch)
{
    if (ch >= '0' && ch <= '9')
        return ch - '0';
    if (ch >= 'a' && ch <= 'f')
        return ch - 'a' + 10;
    if (ch >= 'A' && ch <= 'F')
        return ch - 'A' + 10;
    return -1;
}

static int equivalence_hex_to_raw(const char *hex, unsigned char raw[32])
{
    size_t i;
    if (hex == NULL || strlen(hex) != 64U)
        return 0;
    for (i = 0; i < 32U; i++) {
        int high = equivalence_hex_value(hex[i * 2U]);
        int low = equivalence_hex_value(hex[i * 2U + 1U]);
        if (high < 0 || low < 0)
            return 0;
        raw[i] = (unsigned char)((high << 4) | low);
    }
    return 1;
}

static int equivalence_read_header(FILE *file, cache_header_t *header)
{
    unsigned char magic[8];
    uint32_t i;

    memset(header, 0, sizeof(*header));
    if (!equivalence_read_exact(file, magic, sizeof(magic)) ||
        memcmp(magic, cache_magic, sizeof(magic)) != 0)
        return 0;
    if (!equivalence_read_u32_le(file, &header->format_version) ||
        !equivalence_read_u32_le(file, &header->header_size) ||
        !equivalence_read_u32_le(file, &header->entry_size) ||
        !equivalence_read_u32_le(file, &header->float_format) ||
        !equivalence_read_u32_le(file, &header->byte_order) ||
        !equivalence_read_u32_le(file, &header->model_dim) ||
        !equivalence_read_u32_le(file, &header->vocab_size) ||
        !equivalence_read_u32_le(file, &header->record_count) ||
        !equivalence_read_u32_le(file, &header->target_count))
        return 0;
    for (i = 0; i < CACHE_SPLIT_COUNT; i++) {
        if (!equivalence_read_u32_le(file, &header->split_records[i]))
            return 0;
    }
    for (i = 0; i < CACHE_SPLIT_COUNT; i++) {
        if (!equivalence_read_u32_le(file, &header->split_targets[i]))
            return 0;
    }
    return equivalence_read_exact(file, header->dataset_sha, 32U) &&
        equivalence_read_exact(file, header->checkpoint_sha, 32U) &&
        equivalence_read_exact(file, header->tokenizer_sha, 32U);
}

static int equivalence_header_valid(const cache_header_t *header,
    const unsigned char dataset_sha[32],
    const unsigned char checkpoint_sha[32],
    const unsigned char tokenizer_sha[32])
{
    uint32_t i;
    if (header->format_version != CACHE_FORMAT_VERSION ||
        header->header_size != CACHE_HEADER_SIZE ||
        header->entry_size != CACHE_ENTRY_SIZE ||
        header->float_format != 1U || header->byte_order != 1U ||
        header->model_dim != CACHE_DIM || header->vocab_size != CACHE_VOCAB ||
        header->record_count != CACHE_RECORDS || header->target_count != CACHE_TARGETS ||
        memcmp(header->dataset_sha, dataset_sha, 32U) != 0 ||
        memcmp(header->checkpoint_sha, checkpoint_sha, 32U) != 0 ||
        memcmp(header->tokenizer_sha, tokenizer_sha, 32U) != 0)
        return 0;
    for (i = 0U; i < CACHE_SPLIT_COUNT; i++) {
        if (header->split_records[i] != expected_split_records[i] ||
            header->split_targets[i] != expected_split_targets[i])
            return 0;
    }
    return 1;
}

static int equivalence_float_format_supported(void)
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

static int equivalence_derive_entries(Tokenizer *tokenizer,
    Transformer *transformer,
    stage4e_dataset_t *dataset,
    expected_entry_t expected[CACHE_TARGETS],
    uint32_t split_records[CACHE_SPLIT_COUNT],
    uint32_t split_targets[CACHE_SPLIT_COUNT],
    uint32_t *expected_count)
{
    uint32_t record_index;
    uint32_t count = 0U;

    memset(split_records, 0, CACHE_SPLIT_COUNT * sizeof(uint32_t));
    memset(split_targets, 0, CACHE_SPLIT_COUNT * sizeof(uint32_t));
    for (record_index = 0U; record_index < (uint32_t)dataset->count; record_index++) {
        stage4e_record_t *record = &dataset->items[record_index];
        int split_id = stage4e_split_index(record->split);
        int k;

        if (split_id < 0 || split_id >= (int)CACHE_SPLIT_COUNT ||
            !stage4e_prepare_record_tokens(tokenizer, transformer, record))
            return 0;
        split_records[split_id]++;
        for (k = record->first_target_index; k < record->training_seq_count; k++) {
            if (count >= CACHE_TARGETS)
                return 0;
            expected[count].record_index = record_index;
            expected[count].input_position = (uint32_t)(k - 1);
            expected[count].target_token_id = (uint32_t)record->training_seq_tokens[k];
            expected[count].split_id = (uint32_t)split_id;
            count++;
            split_targets[split_id]++;
        }
    }
    *expected_count = count;
    return count == CACHE_TARGETS;
}

static int equivalence_loss(const float *logits, int target, double *loss_out)
{
    double max_logit;
    double sum = 0.0;
    double logprob;
    int i;

    if (target < 0 || target >= (int)CACHE_VOCAB)
        return 0;
    max_logit = logits[0];
    for (i = 1; i < (int)CACHE_VOCAB; i++) {
        if (!isfinite(logits[i]))
            return 0;
        if (logits[i] > max_logit)
            max_logit = logits[i];
    }
    if (!isfinite(logits[0]))
        return 0;
    for (i = 0; i < (int)CACHE_VOCAB; i++)
        sum += exp((double)logits[i] - max_logit);
    if (!(sum > 0.0) || !isfinite(sum))
        return 0;
    logprob = (double)logits[target] - max_logit - log(sum);
    *loss_out = -logprob;
    return isfinite(*loss_out);
}

static int equivalence_argmax(const float *logits)
{
    int best = 0;
    float best_value = logits[0];
    int i;
    for (i = 1; i < (int)CACHE_VOCAB; i++) {
        if (logits[i] > best_value) {
            best_value = logits[i];
            best = i;
        }
    }
    return best;
}

static uint32_t float_bits(float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static double abs_difference(float left, float right)
{
    double difference = (double)left - (double)right;
    return difference < 0.0 ? -difference : difference;
}

static int read_cache_entry(FILE *file, cache_entry_t *entry, float hidden[CACHE_DIM])
{
    uint32_t i;
    if (!equivalence_read_u32_le(file, &entry->record_index) ||
        !equivalence_read_u32_le(file, &entry->input_position) ||
        !equivalence_read_u32_le(file, &entry->target_token_id) ||
        !equivalence_read_u32_le(file, &entry->split_id))
        return 0;
    for (i = 0U; i < CACHE_DIM; i++) {
        if (!equivalence_read_f32_le(file, &hidden[i], NULL))
            return 0;
    }
    return 1;
}

static int compare_accepted_loss(double actual, double expected)
{
    double difference = actual - expected;
    if (difference < 0.0)
        difference = -difference;
    return difference <= LOSS_COMPARE_TOLERANCE;
}

#ifdef STAGE4E_CACHE_EQUIVALENCE_A4
static uint64_t equivalence_kv_checksum(const Transformer *tr, int last_position)
{
    const Config *config = &tr->config;
    const RunState *state = &tr->state;
    size_t kv_dim = ((size_t)config->dim * (size_t)config->n_kv_heads) /
        (size_t)config->n_heads;
    uint64_t hash = UINT64_C(1469598103934665603);
    int layer;
    int position;
    size_t index;

    for (layer = 0; layer < config->n_layers; layer++) {
        size_t layer_offset = (size_t)layer * (size_t)config->seq_len * kv_dim;
        for (position = 0; position <= last_position; position++) {
            size_t position_offset = layer_offset + (size_t)position * kv_dim;
            for (index = 0; index < kv_dim; index++) {
                uint32_t key_bits;
                uint32_t value_bits;
                memcpy(&key_bits, &state->key_cache[position_offset + index], 4U);
                memcpy(&value_bits, &state->value_cache[position_offset + index], 4U);
                hash ^= key_bits;
                hash *= UINT64_C(1099511628211);
                hash ^= value_bits;
                hash *= UINT64_C(1099511628211);
            }
        }
    }
    return hash;
}

static int equivalence_record0_context_check(Transformer *transformer,
    Tokenizer *tokenizer,
    stage4e_dataset_t *dataset,
    equivalence_stats_t *stats)
{
    stage4e_record_t *record;
    int position;
    uint32_t forward_calls = 0U;
    uint64_t checksum;

    if (dataset->count != CACHE_RECORDS || strcmp(dataset->items[0].id,
        "svc-status-basic-train-001") != 0)
        return 0;
    record = &dataset->items[0];
    if (record->training_seq_tokens == NULL &&
        !stage4e_prepare_record_tokens(tokenizer, transformer, record))
        return 0;
    clear_run_state(transformer);
    for (position = 0; position < 15; position++) {
        (void)forward(transformer, record->training_seq_tokens[position], position);
        forward_calls++;
    }
    checksum = equivalence_kv_checksum(transformer, 14);
    stats->record0_forward_calls_before_position_15 = forward_calls;
    stats->record0_kv_checksum_before_position_15 = checksum;
    (void)forward(transformer, record->training_seq_tokens[15], 15);
    forward_calls++;
    stats->record0_forward_calls_before_first_target = forward_calls;
    stats->record0_full_context_check = record->training_seq_count == 19 &&
        record->first_target_index == 16 && forward_calls == 16 &&
        record->training_seq_tokens[16] == 2669 &&
        checksum == UINT64_C(0xabb568d3ab418288);
    printf("record0.sequence_count=%d\n", record->training_seq_count);
    printf("record0.first_target_index=%d\n", record->first_target_index);
    printf("record0.forward_calls_before_position_15=%u\n",
        stats->record0_forward_calls_before_position_15);
    printf("record0.forward_calls_before_first_target=%u\n",
        stats->record0_forward_calls_before_first_target);
    printf("record0.kv_checksum_before_position_15=%016llx\n",
        (unsigned long long)checksum);
    printf("record0.first_supervised_input_position=15\n");
    printf("record0.first_supervised_target_token_id=%d\n",
        record->training_seq_tokens[16]);
    printf("record0_full_context_check=%s\n",
        stats->record0_full_context_check ? "PASS" : "FAIL");
    return stats->record0_full_context_check;
}

static int report_equivalence_a4(const char *path,
    const char *cache_sha,
    const char *dataset_sha,
    const char *checkpoint_sha,
    const char *tokenizer_sha,
    const char *b1r_sha,
    const char *b2r_sha,
    const equivalence_stats_t *stats,
    const split_metrics_t live_splits[CACHE_SPLIT_COUNT],
    const split_metrics_t cached_splits[CACHE_SPLIT_COUNT],
    double live_total_loss,
    double cached_total_loss,
    uint64_t live_top1,
    uint64_t cached_top1,
    int live_loss_match,
    int cached_loss_match,
    int live_top1_match,
    int cached_top1_match,
    int equivalence_passed)
{
    static const char *split_names[3] = { "train", "validation", "test" };
    FILE *report = fopen(path, "wb");
    uint32_t split;
    int ok;

    if (report == NULL)
        return 0;
    ok = fprintf(report,
        "cache_sha256=%s\n"
        "dataset_sha256=%s\n"
        "base_checkpoint_sha256=%s\n"
        "tokenizer_sha256=%s\n"
        "b1r_a4_report_sha256=%s\n"
        "b2r_a4_report_sha256=%s\n"
        "record_count=%u\n"
        "record0_full_context_check=%s\n"
        "record0_forward_calls_before_position_15=%u\n"
        "record0_forward_calls_before_first_target=%u\n"
        "record0_kv_checksum_before_position_15=%016llx\n"
        "expected_cache_rows=%u\n"
        "consumed_cache_rows=%u\n"
        "remaining_cache_rows=%u\n"
        "hidden_rows_compared=%u\n"
        "hidden_floats_compared=%u\n"
        "hidden_bit_mismatch_count=%u\n"
        "hidden_rows_with_mismatch=%u\n"
        "maximum_hidden_absolute_difference=%.12g\n"
        "logit_rows_compared=%u\n"
        "logits_compared=%u\n"
        "logit_bit_mismatch_count=%u\n"
        "logit_rows_with_mismatch=%u\n"
        "maximum_logit_absolute_difference=%.12g\n"
        "metadata_record_index_mismatches=%u\n"
        "metadata_input_position_mismatches=%u\n"
        "metadata_target_token_mismatches=%u\n"
        "metadata_split_id_mismatches=%u\n"
        "metadata_mismatch_count=%u\n"
        "per_row_loss_mismatch_count=%u\n"
        "maximum_per_row_loss_difference=%.12f\n"
        "top1_token_mismatch_count=%u\n",
        cache_sha, dataset_sha, checkpoint_sha, tokenizer_sha, b1r_sha, b2r_sha,
        CACHE_RECORDS, stats->record0_full_context_check ? "PASS" : "FAIL",
        stats->record0_forward_calls_before_position_15,
        stats->record0_forward_calls_before_first_target,
        (unsigned long long)stats->record0_kv_checksum_before_position_15,
        CACHE_TARGETS, stats->consumed_cache_rows, stats->remaining_cache_rows,
        stats->hidden_rows_compared, stats->hidden_float_count,
        stats->hidden_bitwise_mismatches, stats->hidden_rows_with_mismatch,
        stats->hidden_max_abs_difference, stats->logit_rows_compared,
        stats->logit_float_count, stats->logit_bitwise_mismatches,
        stats->logit_rows_with_mismatch, stats->logit_max_abs_difference,
        stats->metadata_record_index_mismatches,
        stats->metadata_input_position_mismatches,
        stats->metadata_target_token_mismatches,
        stats->metadata_split_id_mismatches, stats->metadata_mismatches,
        stats->per_position_loss_mismatches,
        stats->maximum_position_loss_difference,
        stats->top1_token_mismatches) >= 0;
    if (ok) {
        ok = fprintf(report,
            "live_total_target_prediction_count=%u\n"
            "live_total_target_loss=%.12f\n"
            "live_average_target_loss=%.12f\n"
            "live_target_top1_correct=%llu\n"
            "live_target_top1_accuracy=%.12f\n"
            "cached_total_target_prediction_count=%u\n"
            "cached_total_target_loss=%.12f\n"
            "cached_average_target_loss=%.12f\n"
            "cached_target_top1_correct=%llu\n"
            "cached_target_top1_accuracy=%.12f\n"
            "live_b1r_a4_loss_metrics_match=%s\n"
            "cached_b1r_a4_loss_metrics_match=%s\n"
            "live_b2r_a4_top1_metrics_match=%s\n"
            "cached_b2r_a4_top1_metrics_match=%s\n",
            stats->compared_target_positions, live_total_loss,
            live_total_loss / (double)stats->compared_target_positions,
            (unsigned long long)live_top1,
            (double)live_top1 / (double)stats->compared_target_positions,
            stats->compared_target_positions, cached_total_loss,
            cached_total_loss / (double)stats->compared_target_positions,
            (unsigned long long)cached_top1,
            (double)cached_top1 / (double)stats->compared_target_positions,
            live_loss_match ? "yes" : "no", cached_loss_match ? "yes" : "no",
            live_top1_match ? "yes" : "no", cached_top1_match ? "yes" : "no") >= 0;
    }
    for (split = 0U; ok && split < 3U; split++) {
        double live_average = live_splits[split].loss_sum /
            (double)live_splits[split].target_count;
        double cached_average = cached_splits[split].loss_sum /
            (double)cached_splits[split].target_count;
        ok = fprintf(report,
            "live_%s_target_prediction_count=%u\n"
            "live_%s_total_target_loss=%.12f\n"
            "live_%s_average_target_loss=%.12f\n"
            "live_%s_target_top1_correct=%llu\n"
            "live_%s_target_top1_accuracy=%.12f\n"
            "cached_%s_target_prediction_count=%u\n"
            "cached_%s_total_target_loss=%.12f\n"
            "cached_%s_average_target_loss=%.12f\n"
            "cached_%s_target_top1_correct=%llu\n"
            "cached_%s_target_top1_accuracy=%.12f\n",
            split_names[split], live_splits[split].target_count,
            split_names[split], live_splits[split].loss_sum,
            split_names[split], live_average,
            split_names[split], (unsigned long long)live_splits[split].top1_correct,
            split_names[split], (double)live_splits[split].top1_correct /
                (double)live_splits[split].target_count,
            split_names[split], cached_splits[split].target_count,
            split_names[split], cached_splits[split].loss_sum,
            split_names[split], cached_average,
            split_names[split], (unsigned long long)cached_splits[split].top1_correct,
            split_names[split], (double)cached_splits[split].top1_correct /
                (double)cached_splits[split].target_count) >= 0;
    }
    if (ok && stats->first_hidden_mismatch) {
        ok = fprintf(report,
            "first_hidden_mismatch.record_index=%u\n"
            "first_hidden_mismatch.input_position=%u\n"
            "first_hidden_mismatch.dimension=%u\n"
            "first_hidden_mismatch.live_bits=%08x\n"
            "first_hidden_mismatch.cached_bits=%08x\n"
            "first_hidden_mismatch.live_value=%.9g\n"
            "first_hidden_mismatch.cached_value=%.9g\n",
            stats->first_hidden_record, stats->first_hidden_position,
            stats->first_hidden_dimension, stats->first_hidden_live_bits,
            stats->first_hidden_cache_bits, stats->first_hidden_live_value,
            stats->first_hidden_cache_value) >= 0;
    }
    if (ok && stats->first_logit_mismatch) {
        ok = fprintf(report,
            "first_logit_mismatch.record_index=%u\n"
            "first_logit_mismatch.input_position=%u\n"
            "first_logit_mismatch.vocab_index=%u\n"
            "first_logit_mismatch.live_bits=%08x\n"
            "first_logit_mismatch.cached_bits=%08x\n"
            "first_logit_mismatch.live_value=%.9g\n"
            "first_logit_mismatch.cached_value=%.9g\n",
            stats->first_logit_record, stats->first_logit_position,
            stats->first_logit_vocab, stats->first_logit_live_bits,
            stats->first_logit_cache_bits, stats->first_logit_live_value,
            stats->first_logit_cache_value) >= 0;
    }
    if (ok) {
        ok = fprintf(report, "equivalence_pass=%s\n",
            equivalence_passed ? "yes" : "no") >= 0;
    }
    if (fclose(report) != 0)
        ok = 0;
    return ok;
}

static int equivalence_run_a4(FILE *cache_file,
    stage4e_dataset_t *dataset,
    const expected_entry_t expected[CACHE_TARGETS],
    uint32_t expected_count,
    Transformer *transformer,
    equivalence_stats_t *stats,
    split_metrics_t live_splits[CACHE_SPLIT_COUNT],
    split_metrics_t cached_splits[CACHE_SPLIT_COUNT],
    double *live_total_loss,
    double *cached_total_loss,
    uint64_t *live_total_top1,
    uint64_t *cached_total_top1)
{
    float cached_hidden[CACHE_DIM];
    float cached_logits[CACHE_VOCAB];
    uint32_t record_index;
    uint32_t cache_row = 0U;

    stats->expected_cache_rows = expected_count;
    for (record_index = 0U; record_index < (uint32_t)dataset->count; record_index++) {
        stage4e_record_t *record = &dataset->items[record_index];
        int split_id = stage4e_split_index(record->split);
        int forward_calls = 0;
        int k;
        uint32_t record_target_count = 0U;
        uint64_t record_live_top1 = 0U;
        uint64_t record_cached_top1 = 0U;
        double record_live_loss = 0.0;
        double record_cached_loss = 0.0;

        if (split_id < 0 || split_id >= 3)
            return 0;
        clear_run_state(transformer);

        for (k = 1; k < record->training_seq_count; k++) {
            int input_position = k - 1;
            int target_token = record->training_seq_tokens[k];
            const float *live_logits;

            if (record_index == 0U && input_position == 15) {
                uint64_t checksum = equivalence_kv_checksum(transformer, 14);
                stats->record0_forward_calls_before_position_15 =
                    forward_calls;
                stats->record0_kv_checksum_before_position_15 = checksum;
            }

            live_logits = forward(transformer,
                record->training_seq_tokens[input_position], input_position);
            forward_calls++;
            if (record_index == 0U && k == record->first_target_index)
                stats->record0_forward_calls_before_first_target = forward_calls;

            if (k < record->first_target_index)
                continue;

            if (cache_row >= expected_count)
                return 0;
            {
                cache_entry_t actual;
                const expected_entry_t *wanted = &expected[cache_row];
                uint32_t hidden_index;
                uint32_t vocab_index;
                int row_hidden_mismatch = 0;
                int row_logit_mismatch = 0;
                int row_metadata_mismatch;
                double live_loss;
                double cached_loss;
                int live_top1;
                int cached_top1;
                int live_correct;
                int cached_correct;

                if (!read_cache_entry(cache_file, &actual, cached_hidden))
                    return 0;
                stats->consumed_cache_rows++;
                row_metadata_mismatch =
                    actual.record_index != wanted->record_index ||
                    actual.input_position != wanted->input_position ||
                    actual.target_token_id != wanted->target_token_id ||
                    actual.target_token_id != (uint32_t)target_token ||
                    actual.split_id != wanted->split_id ||
                    actual.split_id != (uint32_t)split_id;
                if (actual.record_index != wanted->record_index)
                    stats->metadata_record_index_mismatches++;
                if (actual.input_position != wanted->input_position)
                    stats->metadata_input_position_mismatches++;
                if (actual.target_token_id != wanted->target_token_id ||
                    actual.target_token_id != (uint32_t)target_token)
                    stats->metadata_target_token_mismatches++;
                if (actual.split_id != wanted->split_id ||
                    actual.split_id != (uint32_t)split_id)
                    stats->metadata_split_id_mismatches++;
                if (row_metadata_mismatch)
                    stats->metadata_mismatches++;

                for (hidden_index = 0U; hidden_index < CACHE_DIM; hidden_index++) {
                    float live_value = transformer->state.x[hidden_index];
                    float cached_value = cached_hidden[hidden_index];
                    uint32_t live_bits = float_bits(live_value);
                    uint32_t cached_bits = float_bits(cached_value);
                    double difference = abs_difference(live_value, cached_value);
                    stats->hidden_float_count++;
                    if (difference > stats->hidden_max_abs_difference)
                        stats->hidden_max_abs_difference = difference;
                    if (live_bits != cached_bits) {
                        stats->hidden_bitwise_mismatches++;
                        row_hidden_mismatch = 1;
                        if (!stats->first_hidden_mismatch) {
                            stats->first_hidden_mismatch = 1;
                            stats->first_hidden_record = record_index;
                            stats->first_hidden_position = (uint32_t)input_position;
                            stats->first_hidden_dimension = hidden_index;
                            stats->first_hidden_live_bits = live_bits;
                            stats->first_hidden_cache_bits = cached_bits;
                            stats->first_hidden_live_value = live_value;
                            stats->first_hidden_cache_value = cached_value;
                        }
                    }
                }
                stats->hidden_rows_compared++;
                if (row_hidden_mismatch)
                    stats->hidden_rows_with_mismatch++;

                matmul(cached_logits, cached_hidden, transformer->weights.wcls,
                    (int)CACHE_DIM, (int)CACHE_VOCAB);
                for (vocab_index = 0U; vocab_index < CACHE_VOCAB; vocab_index++) {
                    float live_value = live_logits[vocab_index];
                    float cached_value = cached_logits[vocab_index];
                    uint32_t live_bits = float_bits(live_value);
                    uint32_t cached_bits = float_bits(cached_value);
                    double difference = abs_difference(live_value, cached_value);
                    stats->logit_float_count++;
                    if (difference > stats->logit_max_abs_difference)
                        stats->logit_max_abs_difference = difference;
                    if (live_bits != cached_bits) {
                        stats->logit_bitwise_mismatches++;
                        row_logit_mismatch = 1;
                        if (!stats->first_logit_mismatch) {
                            stats->first_logit_mismatch = 1;
                            stats->first_logit_record = record_index;
                            stats->first_logit_position = (uint32_t)input_position;
                            stats->first_logit_vocab = vocab_index;
                            stats->first_logit_live_bits = live_bits;
                            stats->first_logit_cache_bits = cached_bits;
                            stats->first_logit_live_value = live_value;
                            stats->first_logit_cache_value = cached_value;
                        }
                    }
                }
                stats->logit_rows_compared++;
                if (row_logit_mismatch)
                    stats->logit_rows_with_mismatch++;

                if (!equivalence_loss(live_logits, target_token, &live_loss) ||
                    !equivalence_loss(cached_logits, target_token, &cached_loss))
                    return 0;
                {
                    double difference = live_loss - cached_loss;
                    if (difference < 0.0)
                        difference = -difference;
                    if (difference > stats->maximum_position_loss_difference)
                        stats->maximum_position_loss_difference = difference;
                    if (live_loss != cached_loss)
                        stats->per_position_loss_mismatches++;
                }

                live_top1 = equivalence_argmax(live_logits);
                cached_top1 = equivalence_argmax(cached_logits);
                if (live_top1 != cached_top1)
                    stats->top1_token_mismatches++;
                live_correct = live_top1 == target_token;
                cached_correct = cached_top1 == target_token;
                record_live_loss += live_loss;
                record_cached_loss += cached_loss;
                record_target_count++;
                if (live_correct)
                    record_live_top1++;
                if (cached_correct)
                    record_cached_top1++;
                cache_row++;
                stats->compared_target_positions++;
            }
        }

        if (record_target_count == 0U)
            return 0;
        live_splits[split_id].target_count += record_target_count;
        live_splits[split_id].loss_sum += record_live_loss;
        live_splits[split_id].top1_correct += record_live_top1;
        cached_splits[split_id].target_count += record_target_count;
        cached_splits[split_id].loss_sum += record_cached_loss;
        cached_splits[split_id].top1_correct += record_cached_top1;
        *live_total_loss += record_live_loss;
        *cached_total_loss += record_cached_loss;
        *live_total_top1 += record_live_top1;
        *cached_total_top1 += record_cached_top1;

        if (record_index == 0U) {
            stats->record0_full_context_check =
                record->training_seq_count == 19 &&
                record->first_target_index == 16 &&
                stats->record0_forward_calls_before_position_15 == 15U &&
                stats->record0_forward_calls_before_first_target == 16U &&
                record->training_seq_tokens[16] == 2669 &&
                stats->record0_kv_checksum_before_position_15 ==
                    UINT64_C(0xabb568d3ab418288);
        }
    }
    stats->remaining_cache_rows = expected_count >= cache_row ?
        expected_count - cache_row : 0U;
    if (cache_row != expected_count)
        return 0;
    return stats->record0_full_context_check;
}
#endif

#ifndef STAGE4E_CACHE_EQUIVALENCE_A4
static int report_equivalence(const char *path,
    const char *cache_sha,
    const char *dataset_sha,
    const char *checkpoint_sha,
    const char *tokenizer_sha,
    const equivalence_stats_t *stats,
    const split_metrics_t splits[CACHE_SPLIT_COUNT],
    double total_loss,
    uint64_t total_top1_correct,
    int accepted_loss_match,
    int accepted_top1_match)
{
    FILE *report = fopen(path, "wb");
    uint32_t split;
    int ok;

    if (report == NULL)
        return 0;
    ok = fprintf(report,
        "cache_sha256=%s\n"
        "dataset_sha256=%s\n"
        "checkpoint_sha256=%s\n"
        "tokenizer_sha256=%s\n"
        "compared_target_positions=%u\n"
        "metadata_mismatches=%u\n"
        "hidden_float_count=%u\n"
        "hidden_bitwise_mismatches=%u\n"
        "hidden_max_abs_difference=%.12g\n"
        "logit_float_count=%u\n"
        "logit_bitwise_mismatches=%u\n"
        "logit_max_abs_difference=%.12g\n"
        "per_position_loss_mismatches=%u\n"
        "maximum_position_loss_difference=%.12g\n"
        "top1_token_mismatches=%u\n"
        "top1_correctness_mismatches=%u\n"
        "cached_target_prediction_count=%u\n"
        "cached_total_target_loss=%.12f\n"
        "cached_average_target_loss=%.12f\n"
        "cached_target_top1_correct=%llu\n"
        "cached_target_top1_accuracy=%.12f\n",
        cache_sha, dataset_sha, checkpoint_sha, tokenizer_sha,
        stats->compared_target_positions, stats->metadata_mismatches,
        stats->hidden_float_count, stats->hidden_bitwise_mismatches,
        stats->hidden_max_abs_difference,
        stats->logit_float_count, stats->logit_bitwise_mismatches,
        stats->logit_max_abs_difference,
        stats->per_position_loss_mismatches,
        stats->maximum_position_loss_difference,
        stats->top1_token_mismatches, stats->top1_correctness_mismatches,
        stats->compared_target_positions, total_loss,
        total_loss / (double)stats->compared_target_positions,
        (unsigned long long)total_top1_correct,
        (double)total_top1_correct / (double)stats->compared_target_positions) >= 0;
    if (ok) {
        ok = fprintf(report, "accepted_baseline_loss_match=%s\naccepted_baseline_top1_match=%s\n",
            accepted_loss_match ? "yes" : "no",
            accepted_top1_match ? "yes" : "no") >= 0;
    }
    for (split = 0U; ok && split < 3U; split++) {
        const char *name = split == 0U ? "train" : (split == 1U ? "validation" : "test");
        double average_loss = splits[split].target_count == 0U ? 0.0 :
            splits[split].loss_sum / (double)splits[split].target_count;
        double top1_accuracy = splits[split].target_count == 0U ? 0.0 :
            (double)splits[split].top1_correct / (double)splits[split].target_count;
        ok = fprintf(report,
            "cached_%s_target_prediction_count=%u\n"
            "cached_%s_total_target_loss=%.12f\n"
            "cached_%s_average_target_loss=%.12f\n"
            "cached_%s_target_top1_correct=%llu\n"
            "cached_%s_target_top1_accuracy=%.12f\n",
            name, splits[split].target_count,
            name, splits[split].loss_sum,
            name, average_loss,
            name, (unsigned long long)splits[split].top1_correct,
            name, top1_accuracy) >= 0;
    }
    if (stats->first_hidden_mismatch) {
        ok = ok && fprintf(report,
            "first_hidden_mismatch.record_index=%u\n"
            "first_hidden_mismatch.input_position=%u\n"
            "first_hidden_mismatch.dimension=%u\n"
            "first_hidden_mismatch.live_bits=%08x\n"
            "first_hidden_mismatch.cached_bits=%08x\n"
            "first_hidden_mismatch.live_value=%.9g\n"
            "first_hidden_mismatch.cached_value=%.9g\n",
            stats->first_hidden_record, stats->first_hidden_position,
            stats->first_hidden_dimension, stats->first_hidden_live_bits,
            stats->first_hidden_cache_bits, stats->first_hidden_live_value,
            stats->first_hidden_cache_value) >= 0;
    }
    if (stats->first_logit_mismatch) {
        ok = ok && fprintf(report,
            "first_logit_mismatch.record_index=%u\n"
            "first_logit_mismatch.input_position=%u\n"
            "first_logit_mismatch.vocab_index=%u\n"
            "first_logit_mismatch.uncached_bits=%08x\n"
            "first_logit_mismatch.cached_bits=%08x\n"
            "first_logit_mismatch.uncached_value=%.9g\n"
            "first_logit_mismatch.cached_value=%.9g\n",
            stats->first_logit_record, stats->first_logit_position,
            stats->first_logit_vocab, stats->first_logit_live_bits,
            stats->first_logit_cache_bits, stats->first_logit_live_value,
            stats->first_logit_cache_value) >= 0;
    }
    if (fclose(report) != 0)
        ok = 0;
    return ok;
}
#endif

static void equivalence_usage(const char *program)
{
#ifdef STAGE4E_CACHE_EQUIVALENCE_A4
    fprintf(stderr,
        "Usage: %s <checkpoint.bin> -d <approved.records> -z <tokenizer.bin> "
        "-c <cache.bin> -r <report.txt> -1 <b1r-report.txt> "
        "-2 <b2r-report.txt> [--record0-context-check]\n",
        program);
#else
    fprintf(stderr,
        "Usage: %s <checkpoint.bin> -d <approved.records> -z <tokenizer.bin> "
        "-c <cache.bin> -r <report.txt>\n",
        program);
#endif
}

int main(int argc, char **argv)
{
    const char *checkpoint_path = NULL;
    const char *dataset_path = NULL;
    const char *tokenizer_path = NULL;
#ifdef STAGE4E_CACHE_EQUIVALENCE_A4
    const char *cache_path = "stage4e-target-cache-a4.bin";
    const char *report_path = "stage4e-c2c-a4-full-context-equivalence-report.txt";
    const char *b1r_report_path = "stage4e-baseline-unadapted-full-context-a4-report.txt";
    const char *b2r_report_path = "stage4e-baseline-unadapted-full-context-a4-top1-report.txt";
    int record0_context_check = 0;
#else
    const char *cache_path = "stage4e-target-cache.bin";
    const char *report_path = "stage4e-cache-equivalence-report.txt";
#endif
    char dataset_sha[65] = "";
    char checkpoint_sha[65] = "";
    char tokenizer_sha[65] = "";
    char cache_sha[65] = "";
#ifdef STAGE4E_CACHE_EQUIVALENCE_A4
    char b1r_report_sha[65] = "";
    char b2r_report_sha[65] = "";
#endif
    unsigned char dataset_raw[32];
    unsigned char checkpoint_raw[32];
    unsigned char tokenizer_raw[32];
    unsigned char cache_header_dataset[32];
    unsigned char cache_header_checkpoint[32];
    unsigned char cache_header_tokenizer[32];
    unsigned char expected_dataset_raw[32];
    unsigned char expected_checkpoint_raw[32];
    unsigned char expected_tokenizer_raw[32];
    FILE *cache_file = NULL;
    cache_header_t header;
    stage4e_dataset_t dataset;
    Transformer transformer;
    Tokenizer tokenizer;
    expected_entry_t expected[CACHE_TARGETS];
    split_metrics_t split_metrics[CACHE_SPLIT_COUNT];
#ifdef STAGE4E_CACHE_EQUIVALENCE_A4
    split_metrics_t live_split_metrics[CACHE_SPLIT_COUNT];
#endif
    equivalence_stats_t stats;
#ifndef STAGE4E_CACHE_EQUIVALENCE_A4
    float cached_hidden[CACHE_DIM];
    float cached_logits[CACHE_VOCAB];
#endif
    uint32_t observed_split_records[CACHE_SPLIT_COUNT] = { 0U, 0U, 0U, 0U };
    uint32_t observed_split_targets[CACHE_SPLIT_COUNT] = { 0U, 0U, 0U, 0U };
    uint32_t expected_count = 0U;
    uint32_t entry_index;
    uint64_t total_top1_correct = 0U;
    double total_loss = 0.0;
#ifdef STAGE4E_CACHE_EQUIVALENCE_A4
    uint64_t live_total_top1_correct = 0U;
    double live_total_loss = 0.0;
    int live_b1r_match = 0;
    int cached_b1r_match = 0;
    int live_b2r_match = 0;
    int cached_b2r_match = 0;
#endif
    uint64_t cache_size = 0U;
    int input_identities_valid = 0;
    int cache_size_valid = 0;
    int header_valid = 0;
    int argi;
    int failed = 0;
#ifndef STAGE4E_CACHE_EQUIVALENCE_A4
    int accepted_loss_match = 0;
    int accepted_top1_match = 0;
#endif
    int report_written = 0;

    memset(dataset_raw, 0, sizeof(dataset_raw));
    memset(checkpoint_raw, 0, sizeof(checkpoint_raw));
    memset(tokenizer_raw, 0, sizeof(tokenizer_raw));
    memset(cache_header_dataset, 0, sizeof(cache_header_dataset));
    memset(cache_header_checkpoint, 0, sizeof(cache_header_checkpoint));
    memset(cache_header_tokenizer, 0, sizeof(cache_header_tokenizer));
    memset(expected_dataset_raw, 0, sizeof(expected_dataset_raw));
    memset(expected_checkpoint_raw, 0, sizeof(expected_checkpoint_raw));
    memset(expected_tokenizer_raw, 0, sizeof(expected_tokenizer_raw));
    memset(&header, 0, sizeof(header));
    memset(&dataset, 0, sizeof(dataset));
    memset(&transformer, 0, sizeof(transformer));
    memset(&tokenizer, 0, sizeof(tokenizer));
    memset(expected, 0, sizeof(expected));
    memset(split_metrics, 0, sizeof(split_metrics));
#ifdef STAGE4E_CACHE_EQUIVALENCE_A4
    memset(live_split_metrics, 0, sizeof(live_split_metrics));
#endif
    memset(&stats, 0, sizeof(stats));

    if (argc < 2) {
        equivalence_usage(argv[0]);
        return EXIT_FAILURE;
    }
    checkpoint_path = argv[1];
    for (argi = 2; argi < argc; argi++) {
        if (strcmp(argv[argi], "-d") == 0 && argi + 1 < argc)
            dataset_path = argv[++argi];
        else if (strcmp(argv[argi], "-z") == 0 && argi + 1 < argc)
            tokenizer_path = argv[++argi];
        else if (strcmp(argv[argi], "-c") == 0 && argi + 1 < argc)
            cache_path = argv[++argi];
        else if (strcmp(argv[argi], "-r") == 0 && argi + 1 < argc)
            report_path = argv[++argi];
#ifdef STAGE4E_CACHE_EQUIVALENCE_A4
        else if (strcmp(argv[argi], "-1") == 0 && argi + 1 < argc)
            b1r_report_path = argv[++argi];
        else if (strcmp(argv[argi], "-2") == 0 && argi + 1 < argc)
            b2r_report_path = argv[++argi];
        else if (strcmp(argv[argi], "--record0-context-check") == 0)
            record0_context_check = 1;
#endif
        else {
            equivalence_usage(argv[0]);
            return EXIT_FAILURE;
        }
    }
    if (dataset_path == NULL || tokenizer_path == NULL ||
        strcmp(cache_path, report_path) == 0) {
        equivalence_usage(argv[0]);
        return EXIT_FAILURE;
    }

    if (!stage4_hash_file_sha256_hex(dataset_path, dataset_sha) ||
        !stage4_hash_file_sha256_hex(checkpoint_path, checkpoint_sha) ||
        !stage4_hash_file_sha256_hex(tokenizer_path, tokenizer_sha) ||
        !stage4_hash_file_sha256_hex(cache_path, cache_sha) ||
        !equivalence_hex_to_raw(dataset_sha, dataset_raw) ||
        !equivalence_hex_to_raw(checkpoint_sha, checkpoint_raw) ||
        !equivalence_hex_to_raw(tokenizer_sha, tokenizer_raw) ||
        !equivalence_hex_to_raw(expected_dataset_sha256, expected_dataset_raw) ||
        !equivalence_hex_to_raw(expected_checkpoint_sha256, expected_checkpoint_raw) ||
        !equivalence_hex_to_raw(expected_tokenizer_sha256, expected_tokenizer_raw)) {
        fprintf(stderr, "error: failed to hash or decode an input identity\n");
        failed = 1;
        goto cleanup;
    }
#ifdef STAGE4E_CACHE_EQUIVALENCE_A4
    if (!stage4_hash_file_sha256_hex(b1r_report_path, b1r_report_sha) ||
        !stage4_hash_file_sha256_hex(b2r_report_path, b2r_report_sha)) {
        fprintf(stderr, "error: failed to hash accepted A4 baseline reports\n");
        failed = 1;
        goto cleanup;
    }
    if (strcmp(cache_sha, expected_cache_sha256) != 0 ||
        strcmp(b1r_report_sha, expected_b1r_report_sha256) != 0 ||
        strcmp(b2r_report_sha, expected_b2r_report_sha256) != 0) {
        fprintf(stderr,
            "error: A4 cache or baseline report identity mismatch\n"
            "cache expected=%s actual=%s\n"
            "B1R report expected=%s actual=%s\n"
            "B2R report expected=%s actual=%s\n",
            expected_cache_sha256, cache_sha,
            expected_b1r_report_sha256, b1r_report_sha,
            expected_b2r_report_sha256, b2r_report_sha);
        failed = 1;
        goto cleanup;
    }
#endif
    input_identities_valid =
        strcmp(dataset_sha, expected_dataset_sha256) == 0 &&
        strcmp(checkpoint_sha, expected_checkpoint_sha256) == 0 &&
        strcmp(tokenizer_sha, expected_tokenizer_sha256) == 0;
    if (!input_identities_valid) {
        fprintf(stderr, "error: dataset/checkpoint/tokenizer identity mismatch\n");
        failed = 1;
        goto cleanup;
    }
    if (!equivalence_float_format_supported()) {
        fprintf(stderr, "error: host float format is not little-endian IEEE-754 binary32\n");
        failed = 1;
        goto cleanup;
    }
    cache_size_valid = stage4_file_size_bytes(cache_path, &cache_size) &&
        cache_size == CACHE_FILE_SIZE;
    cache_file = fopen(cache_path, "rb");
    if (cache_file == NULL || !equivalence_read_header(cache_file, &header)) {
        fprintf(stderr, "error: cannot read complete cache header\n");
        failed = 1;
        goto cleanup;
    }
    header_valid = equivalence_header_valid(&header, dataset_raw,
        checkpoint_raw, tokenizer_raw);
    if (!header_valid || !cache_size_valid) {
        fprintf(stderr, "error: cache header or file size mismatch\n");
        failed = 1;
        goto cleanup;
    }
    memcpy(cache_header_dataset, header.dataset_sha, 32U);
    memcpy(cache_header_checkpoint, header.checkpoint_sha, 32U);
    memcpy(cache_header_tokenizer, header.tokenizer_sha, 32U);
    if (memcmp(cache_header_dataset, expected_dataset_raw, 32U) != 0 ||
        memcmp(cache_header_checkpoint, expected_checkpoint_raw, 32U) != 0 ||
        memcmp(cache_header_tokenizer, expected_tokenizer_raw, 32U) != 0) {
        fprintf(stderr, "error: cache header identities mismatch\n");
        failed = 1;
        goto cleanup;
    }

    unload_adapter_runtime();
    load_transformer(&transformer, checkpoint_path);
    load_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);
    if (!stage4e_load_records(dataset_path, &dataset) ||
        dataset.count != CACHE_RECORDS ||
        !equivalence_derive_entries(&tokenizer, &transformer, &dataset,
            expected, observed_split_records, observed_split_targets,
            &expected_count)) {
        fprintf(stderr, "error: cannot derive expected Stage 4E target positions\n");
        failed = 1;
        goto cleanup;
    }
    if (expected_count != CACHE_TARGETS ||
        memcmp(observed_split_records, header.split_records,
            sizeof(observed_split_records)) != 0 ||
        memcmp(observed_split_targets, header.split_targets,
            sizeof(observed_split_targets)) != 0) {
        fprintf(stderr, "error: derived sequence counts disagree with cache header\n");
        failed = 1;
        goto cleanup;
    }

#ifdef STAGE4E_CACHE_EQUIVALENCE_A4
    if (record0_context_check) {
        int passed = equivalence_record0_context_check(&transformer,
            &tokenizer, &dataset, &stats);
        if (cache_file != NULL) {
            fclose(cache_file);
            cache_file = NULL;
        }
        stage4e_dataset_free(&dataset);
        free_tokenizer(&tokenizer);
        free_transformer(&transformer);
        return passed ? EXIT_SUCCESS : EXIT_FAILURE;
    }
#endif

#ifdef STAGE4E_CACHE_EQUIVALENCE_A4
    if (!equivalence_run_a4(cache_file, &dataset, expected, expected_count,
        &transformer, &stats, live_split_metrics, split_metrics,
        &live_total_loss, &total_loss, &live_total_top1_correct,
        &total_top1_correct)) {
        fprintf(stderr, "error: full-context live/cache comparison failed\n");
        failed = 1;
        goto cleanup;
    }
    if (fgetc(cache_file) != EOF) {
        fprintf(stderr, "error: cache has trailing bytes\n");
        failed = 1;
        goto cleanup;
    }
    if (fclose(cache_file) != 0) {
        cache_file = NULL;
        fprintf(stderr, "error: closing cache failed\n");
        failed = 1;
        goto cleanup;
    }
    cache_file = NULL;
    if (!stage4_file_size_bytes(cache_path, &cache_size) ||
        cache_size != CACHE_FILE_SIZE) {
        fprintf(stderr, "error: cache file size changed during comparison\n");
        failed = 1;
        goto cleanup;
    }

    live_b1r_match = stats.compared_target_positions == CACHE_TARGETS &&
        compare_accepted_loss(live_total_loss, accepted_total_loss) &&
        compare_accepted_loss(live_total_loss / (double)CACHE_TARGETS,
            accepted_overall_average_loss);
    cached_b1r_match = stats.compared_target_positions == CACHE_TARGETS &&
        compare_accepted_loss(total_loss, accepted_total_loss) &&
        compare_accepted_loss(total_loss / (double)CACHE_TARGETS,
            accepted_overall_average_loss);
    live_b2r_match = stats.compared_target_positions == accepted_overall_target_count &&
        live_total_top1_correct == accepted_overall_top1_correct;
    cached_b2r_match = stats.compared_target_positions == accepted_overall_target_count &&
        total_top1_correct == accepted_overall_top1_correct;
    for (entry_index = 0U; entry_index < 3U; entry_index++) {
        double live_average = live_split_metrics[entry_index].loss_sum /
            (double)live_split_metrics[entry_index].target_count;
        double cached_average = split_metrics[entry_index].loss_sum /
            (double)split_metrics[entry_index].target_count;
        if (live_split_metrics[entry_index].target_count !=
                accepted_top1_count[entry_index] ||
            !compare_accepted_loss(live_split_metrics[entry_index].loss_sum,
                accepted_split_loss[entry_index]) ||
            !compare_accepted_loss(live_average,
                accepted_average_loss[entry_index]))
            live_b1r_match = 0;
        if (split_metrics[entry_index].target_count !=
                accepted_top1_count[entry_index] ||
            !compare_accepted_loss(split_metrics[entry_index].loss_sum,
                accepted_split_loss[entry_index]) ||
            !compare_accepted_loss(cached_average,
                accepted_average_loss[entry_index]))
            cached_b1r_match = 0;
        if (live_split_metrics[entry_index].top1_correct !=
                accepted_top1_correct[entry_index])
            live_b2r_match = 0;
        if (split_metrics[entry_index].top1_correct !=
                accepted_top1_correct[entry_index])
            cached_b2r_match = 0;
    }
    if (stats.expected_cache_rows != CACHE_TARGETS ||
        stats.consumed_cache_rows != CACHE_TARGETS ||
        stats.remaining_cache_rows != 0U ||
        stats.compared_target_positions != CACHE_TARGETS ||
        stats.hidden_rows_compared != CACHE_TARGETS ||
        stats.logit_rows_compared != CACHE_TARGETS ||
        stats.hidden_rows_with_mismatch != 0U ||
        stats.logit_rows_with_mismatch != 0U ||
        stats.hidden_float_count != HIDDEN_FLOAT_COUNT ||
        stats.logit_float_count != LOGIT_FLOAT_COUNT ||
        stats.metadata_mismatches != 0U ||
        stats.hidden_bitwise_mismatches != 0U ||
        stats.logit_bitwise_mismatches != 0U ||
        stats.per_position_loss_mismatches != 0U ||
        stats.top1_token_mismatches != 0U ||
        !stats.record0_full_context_check || !live_b1r_match ||
        !cached_b1r_match || !live_b2r_match || !cached_b2r_match)
        failed = 1;
#else
    {
        uint32_t cached_entry;
        for (cached_entry = 0U; cached_entry < CACHE_TARGETS; cached_entry++) {
            cache_entry_t actual;
            const expected_entry_t *wanted = &expected[cached_entry];
            const stage4e_record_t *record;
            const float *uncached_logits;
            uint32_t dimension;
            uint32_t vocab_index;
            uint32_t metadata_bad = 0U;
            int target_for_cache;
            int uncached_top1;
            int cached_top1;
            int uncached_correct;
            int cached_correct;
            double uncached_loss;
            double cached_loss;
            int k;
            uint32_t split_id;

            if (!read_cache_entry(cache_file, &actual, cached_hidden)) {
                fprintf(stderr, "error: truncated cache entry %u\n", cached_entry);
                failed = 1;
                goto cleanup;
            }
            if (actual.record_index != wanted->record_index) metadata_bad++;
            if (actual.input_position != wanted->input_position) metadata_bad++;
            if (actual.target_token_id != wanted->target_token_id) metadata_bad++;
            if (actual.split_id != wanted->split_id) metadata_bad++;
            if (metadata_bad != 0U)
                stats.metadata_mismatches++;

            record = &dataset.items[wanted->record_index];
            clear_run_state(&transformer);
            for (k = 1; k <= (int)wanted->input_position; k++) {
                if (k < record->first_target_index)
                    continue;
                (void)forward(&transformer, record->training_seq_tokens[k - 1], k - 1);
            }
            uncached_logits = forward(&transformer,
                record->training_seq_tokens[wanted->input_position],
                (int)wanted->input_position);

            for (dimension = 0U; dimension < CACHE_DIM; dimension++) {
                float live_value = transformer.state.x[dimension];
                float cache_value = cached_hidden[dimension];
                uint32_t live_bits = float_bits(live_value);
                uint32_t cache_bits = float_bits(cache_value);
                double difference = abs_difference(live_value, cache_value);
                stats.hidden_float_count++;
                if (difference > stats.hidden_max_abs_difference)
                    stats.hidden_max_abs_difference = difference;
                if (live_bits != cache_bits) {
                    stats.hidden_bitwise_mismatches++;
                    if (!stats.first_hidden_mismatch) {
                        stats.first_hidden_mismatch = 1;
                        stats.first_hidden_record = wanted->record_index;
                        stats.first_hidden_position = wanted->input_position;
                        stats.first_hidden_dimension = dimension;
                        stats.first_hidden_live_bits = live_bits;
                        stats.first_hidden_cache_bits = cache_bits;
                        stats.first_hidden_live_value = live_value;
                        stats.first_hidden_cache_value = cache_value;
                    }
                }
            }

            matmul(cached_logits, cached_hidden, transformer.weights.wcls,
                (int)CACHE_DIM, (int)CACHE_VOCAB);
            for (vocab_index = 0U; vocab_index < CACHE_VOCAB; vocab_index++) {
                float live_value = uncached_logits[vocab_index];
                float cache_value = cached_logits[vocab_index];
                uint32_t live_bits = float_bits(live_value);
                uint32_t cache_bits = float_bits(cache_value);
                double difference = abs_difference(live_value, cache_value);
                stats.logit_float_count++;
                if (difference > stats.logit_max_abs_difference)
                    stats.logit_max_abs_difference = difference;
                if (live_bits != cache_bits) {
                    stats.logit_bitwise_mismatches++;
                    if (!stats.first_logit_mismatch) {
                        stats.first_logit_mismatch = 1;
                        stats.first_logit_record = wanted->record_index;
                        stats.first_logit_position = wanted->input_position;
                        stats.first_logit_vocab = vocab_index;
                        stats.first_logit_live_bits = live_bits;
                        stats.first_logit_cache_bits = cache_bits;
                        stats.first_logit_live_value = live_value;
                        stats.first_logit_cache_value = cache_value;
                    }
                }
            }

            target_for_cache = actual.target_token_id < CACHE_VOCAB ?
                (int)actual.target_token_id : (int)wanted->target_token_id;
            if (!equivalence_loss(uncached_logits, (int)wanted->target_token_id,
                &uncached_loss) || !equivalence_loss(cached_logits, target_for_cache,
                &cached_loss)) {
                fprintf(stderr, "error: target loss calculation failed at record %u position %u\n",
                    wanted->record_index, wanted->input_position);
                failed = 1;
                goto cleanup;
            }
            {
                double loss_difference = uncached_loss - cached_loss;
                if (loss_difference < 0.0)
                    loss_difference = -loss_difference;
                if (loss_difference > stats.maximum_position_loss_difference)
                    stats.maximum_position_loss_difference = loss_difference;
                if (uncached_loss != cached_loss)
                    stats.per_position_loss_mismatches++;
            }

            uncached_top1 = equivalence_argmax(uncached_logits);
            cached_top1 = equivalence_argmax(cached_logits);
            if (uncached_top1 != cached_top1)
                stats.top1_token_mismatches++;
            uncached_correct = uncached_top1 == (int)wanted->target_token_id;
            cached_correct = cached_top1 == target_for_cache;
            if (uncached_correct != cached_correct)
                stats.top1_correctness_mismatches++;

            split_id = actual.split_id < CACHE_SPLIT_COUNT ?
                actual.split_id : wanted->split_id;
            if (split_id < 3U) {
                split_metrics[split_id].target_count++;
                split_metrics[split_id].loss_sum += cached_loss;
                if (cached_correct)
                    split_metrics[split_id].top1_correct++;
            }
            total_loss += cached_loss;
            if (cached_correct)
                total_top1_correct++;
            stats.compared_target_positions++;
        }
    }

    if (fgetc(cache_file) != EOF) {
        fprintf(stderr, "error: cache has trailing bytes\n");
        failed = 1;
        goto cleanup;
    }
    if (fclose(cache_file) != 0) {
        cache_file = NULL;
        fprintf(stderr, "error: closing cache failed\n");
        failed = 1;
        goto cleanup;
    }
    cache_file = NULL;
    if (!stage4_file_size_bytes(cache_path, &cache_size) || cache_size != CACHE_FILE_SIZE) {
        fprintf(stderr, "error: cache file size changed during comparison\n");
        failed = 1;
        goto cleanup;
    }

    accepted_loss_match = stats.compared_target_positions == CACHE_TARGETS &&
        compare_accepted_loss(total_loss, accepted_total_loss) &&
        compare_accepted_loss(total_loss / (double)CACHE_TARGETS,
            accepted_overall_average_loss);
    accepted_top1_match = stats.compared_target_positions == accepted_overall_target_count &&
        total_top1_correct == accepted_overall_top1_correct;
    for (entry_index = 0U; entry_index < 3U; entry_index++) {
        double average_loss = split_metrics[entry_index].loss_sum /
            (double)split_metrics[entry_index].target_count;
        if (split_metrics[entry_index].target_count != accepted_top1_count[entry_index] ||
            !compare_accepted_loss(average_loss, accepted_average_loss[entry_index]))
            accepted_loss_match = 0;
        if (split_metrics[entry_index].top1_correct != accepted_top1_correct[entry_index])
            accepted_top1_match = 0;
    }

    if (stats.metadata_mismatches != 0U ||
        stats.hidden_bitwise_mismatches != 0U ||
        stats.logit_bitwise_mismatches != 0U ||
        stats.per_position_loss_mismatches != 0U ||
        stats.top1_token_mismatches != 0U ||
        stats.top1_correctness_mismatches != 0U ||
        !accepted_loss_match || !accepted_top1_match)
        failed = 1;
    #endif

cleanup:
    if (cache_file != NULL)
        fclose(cache_file);
    if (stats.hidden_float_count == 0U)
        stats.hidden_float_count = 0U;
    if (stats.logit_float_count == 0U)
        stats.logit_float_count = 0U;
    if (stats.compared_target_positions == CACHE_TARGETS) {
#ifdef STAGE4E_CACHE_EQUIVALENCE_A4
        if (!report_equivalence_a4(report_path, cache_sha, dataset_sha,
            checkpoint_sha, tokenizer_sha, b1r_report_sha, b2r_report_sha,
            &stats, live_split_metrics, split_metrics, live_total_loss,
            total_loss, live_total_top1_correct, total_top1_correct,
            live_b1r_match, cached_b1r_match, live_b2r_match,
            cached_b2r_match, !failed)) {
#else
        if (!report_equivalence(report_path, cache_sha, dataset_sha, checkpoint_sha,
            tokenizer_sha, &stats, split_metrics, total_loss, total_top1_correct,
            accepted_loss_match, accepted_top1_match)) {
#endif
            fprintf(stderr, "error: cannot write equivalence report '%s'\n", report_path);
            failed = 1;
        } else {
            report_written = 1;
        }
    }
    if (stats.first_hidden_mismatch) {
        printf("first_hidden_mismatch record_index=%u input_position=%u dimension=%u live_bits=%08x cached_bits=%08x live=%.9g cached=%.9g\n",
            stats.first_hidden_record, stats.first_hidden_position,
            stats.first_hidden_dimension, stats.first_hidden_live_bits,
            stats.first_hidden_cache_bits, stats.first_hidden_live_value,
            stats.first_hidden_cache_value);
    }
    if (stats.first_logit_mismatch) {
        printf("first_logit_mismatch record_index=%u input_position=%u vocab_index=%u uncached_bits=%08x cached_bits=%08x uncached=%.9g cached=%.9g\n",
            stats.first_logit_record, stats.first_logit_position,
            stats.first_logit_vocab, stats.first_logit_live_bits,
            stats.first_logit_cache_bits, stats.first_logit_live_value,
            stats.first_logit_cache_value);
    }
    if (report_written) {
#ifdef STAGE4E_CACHE_EQUIVALENCE_A4
        printf("record0_full_context_check=%s\n",
            stats.record0_full_context_check ? "PASS" : "FAIL");
        printf("expected_cache_rows=%u\n", stats.expected_cache_rows);
        printf("consumed_cache_rows=%u\n", stats.consumed_cache_rows);
        printf("remaining_cache_rows=%u\n", stats.remaining_cache_rows);
        printf("metadata_mismatches=%u\n", stats.metadata_mismatches);
        printf("hidden_float_count=%u\n", stats.hidden_float_count);
        printf("hidden_bitwise_mismatches=%u\n", stats.hidden_bitwise_mismatches);
        printf("hidden_max_abs_difference=%.12g\n", stats.hidden_max_abs_difference);
        printf("logit_float_count=%u\n", stats.logit_float_count);
        printf("logit_bitwise_mismatches=%u\n", stats.logit_bitwise_mismatches);
        printf("logit_max_abs_difference=%.12g\n", stats.logit_max_abs_difference);
        printf("per_position_loss_mismatches=%u\n", stats.per_position_loss_mismatches);
        printf("maximum_position_loss_difference=%.12g\n",
            stats.maximum_position_loss_difference);
        printf("top1_token_mismatches=%u\n", stats.top1_token_mismatches);
        printf("live_b1r_a4_loss_metrics_match=%s\n",
            live_b1r_match ? "yes" : "no");
        printf("cached_b1r_a4_loss_metrics_match=%s\n",
            cached_b1r_match ? "yes" : "no");
        printf("live_b2r_a4_top1_metrics_match=%s\n",
            live_b2r_match ? "yes" : "no");
        printf("cached_b2r_a4_top1_metrics_match=%s\n",
            cached_b2r_match ? "yes" : "no");
        printf("equivalence_pass=%s\n", failed ? "no" : "yes");
#else
        printf("metadata_mismatches=%u\n", stats.metadata_mismatches);
        printf("hidden_float_count=%u\n", stats.hidden_float_count);
        printf("hidden_bitwise_mismatches=%u\n", stats.hidden_bitwise_mismatches);
        printf("hidden_max_abs_difference=%.12g\n", stats.hidden_max_abs_difference);
        printf("logit_float_count=%u\n", stats.logit_float_count);
        printf("logit_bitwise_mismatches=%u\n", stats.logit_bitwise_mismatches);
        printf("logit_max_abs_difference=%.12g\n", stats.logit_max_abs_difference);
        printf("per_position_loss_mismatches=%u\n", stats.per_position_loss_mismatches);
        printf("maximum_position_loss_difference=%.12g\n", stats.maximum_position_loss_difference);
        printf("top1_token_mismatches=%u\n", stats.top1_token_mismatches);
        printf("top1_correctness_mismatches=%u\n", stats.top1_correctness_mismatches);
        printf("accepted_baseline_loss_match=%s\n", accepted_loss_match ? "yes" : "no");
        printf("accepted_baseline_top1_match=%s\n", accepted_top1_match ? "yes" : "no");
#endif
    }
    stage4e_dataset_free(&dataset);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
