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
#ifdef STAGE4E_CACHE_A4
#define CACHE_TARGETS 165U
#define CACHE_FILE_SIZE 192892U
#else
#define CACHE_TARGETS 156U
#define CACHE_FILE_SIZE 182380U
#endif
#define CACHE_SPLIT_COUNT 4U
#define CACHE_TEMP_SUFFIX ".tmp"

static const char cache_magic[8] = { 'S', '4', 'T', 'C', 'A', 'C', 'H', '1' };
static const char expected_dataset_sha256[] =
#ifdef STAGE4E_CACHE_A4
    "71744cea4de049e2e192cb2a4494c8a4a68e769b3a7f3b8ed4880433e4583b84";
#else
    "b8aa6dc1dbfad7ed144e19a351850189956f79571b05cb70324d21876db0ea78";
#endif
static const char expected_checkpoint_sha256[] =
    "cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a";
static const char expected_tokenizer_sha256[] =
    "50a52ef822ee9e83de5ce9d0be0a025a773d019437f58b5ff9dcafb063ece361";
static const uint32_t expected_split_records[CACHE_SPLIT_COUNT] = { 20U, 5U, 5U, 0U };
#ifdef STAGE4E_CACHE_A4
static const uint32_t expected_split_targets[CACHE_SPLIT_COUNT] = { 111U, 33U, 21U, 0U };
#else
static const uint32_t expected_split_targets[CACHE_SPLIT_COUNT] = { 111U, 24U, 21U, 0U };
#endif

static void cache_usage(const char *program)
{
    fprintf(stderr,
        "Usage: %s <checkpoint.bin> -d <approved.records> -z <tokenizer.bin> "
        "-o <cache.bin> -r <report.txt>\n",
        program);
}

static int cache_hex_value(char ch)
{
    if (ch >= '0' && ch <= '9')
        return ch - '0';
    if (ch >= 'a' && ch <= 'f')
        return ch - 'a' + 10;
    if (ch >= 'A' && ch <= 'F')
        return ch - 'A' + 10;
    return -1;
}

static int cache_hex_to_raw(const char *hex, unsigned char raw[32])
{
    size_t i;

    if (hex == NULL || strlen(hex) != 64U)
        return 0;
    for (i = 0; i < 32U; i++) {
        int high = cache_hex_value(hex[i * 2U]);
        int low = cache_hex_value(hex[i * 2U + 1U]);
        if (high < 0 || low < 0)
            return 0;
        raw[i] = (unsigned char)((high << 4) | low);
    }
    return 1;
}

static int cache_write_u32_le(FILE *file, uint32_t value)
{
    unsigned char bytes[4];
    bytes[0] = (unsigned char)(value & 0xffU);
    bytes[1] = (unsigned char)((value >> 8) & 0xffU);
    bytes[2] = (unsigned char)((value >> 16) & 0xffU);
    bytes[3] = (unsigned char)((value >> 24) & 0xffU);
    return fwrite(bytes, 1, sizeof(bytes), file) == sizeof(bytes);
}

static int cache_write_f32_le(FILE *file, float value)
{
    uint32_t bits;
    unsigned char bytes[4];

    memcpy(&bits, &value, sizeof(bits));
    bytes[0] = (unsigned char)(bits & 0xffU);
    bytes[1] = (unsigned char)((bits >> 8) & 0xffU);
    bytes[2] = (unsigned char)((bits >> 16) & 0xffU);
    bytes[3] = (unsigned char)((bits >> 24) & 0xffU);
    return fwrite(bytes, 1, sizeof(bytes), file) == sizeof(bytes);
}

static int cache_float_format_supported(void)
{
    float one = 1.0f;
    uint32_t bits = 0;

    if (sizeof(float) != 4U || sizeof(uint32_t) != 4U ||
        FLT_RADIX != 2 || FLT_MANT_DIG != 24 || FLT_MAX_EXP != 128 ||
        FLT_MIN_EXP != -125)
        return 0;
    memcpy(&bits, &one, sizeof(bits));
    return bits == UINT32_C(0x3f800000);
}

static int cache_write_header(FILE *file,
    const unsigned char dataset_sha[32],
    const unsigned char checkpoint_sha[32],
    const unsigned char tokenizer_sha[32],
    const uint32_t split_records[CACHE_SPLIT_COUNT],
    const uint32_t split_targets[CACHE_SPLIT_COUNT])
{
    uint32_t i;

    if (fwrite(cache_magic, 1, sizeof(cache_magic), file) != sizeof(cache_magic))
        return 0;
    if (!cache_write_u32_le(file, CACHE_FORMAT_VERSION) ||
        !cache_write_u32_le(file, CACHE_HEADER_SIZE) ||
        !cache_write_u32_le(file, CACHE_ENTRY_SIZE) ||
        !cache_write_u32_le(file, 1U) ||
        !cache_write_u32_le(file, 1U) ||
        !cache_write_u32_le(file, CACHE_DIM) ||
        !cache_write_u32_le(file, CACHE_VOCAB) ||
        !cache_write_u32_le(file, CACHE_RECORDS) ||
        !cache_write_u32_le(file, CACHE_TARGETS))
        return 0;
    for (i = 0; i < CACHE_SPLIT_COUNT; i++) {
        if (!cache_write_u32_le(file, split_records[i]))
            return 0;
    }
    for (i = 0; i < CACHE_SPLIT_COUNT; i++) {
        if (!cache_write_u32_le(file, split_targets[i]))
            return 0;
    }
    if (fwrite(dataset_sha, 1, 32U, file) != 32U ||
        fwrite(checkpoint_sha, 1, 32U, file) != 32U ||
        fwrite(tokenizer_sha, 1, 32U, file) != 32U)
        return 0;
    return 1;
}

static int cache_write_entry(FILE *file,
    uint32_t record_index,
    uint32_t input_position,
    uint32_t target_token_id,
    uint32_t split_id,
    const float *hidden)
{
    uint32_t i;

    if (!cache_write_u32_le(file, record_index) ||
        !cache_write_u32_le(file, input_position) ||
        !cache_write_u32_le(file, target_token_id) ||
        !cache_write_u32_le(file, split_id))
        return 0;

    for (i = 0; i < CACHE_DIM; i++) {
        if (!isfinite(hidden[i]) || !cache_write_f32_le(file, hidden[i]))
            return 0;
    }
    return 1;
}

static int cache_record_report(FILE *report,
    const char *dataset_sha,
    const char *checkpoint_sha,
    const char *tokenizer_sha,
    uint32_t total_records,
    uint32_t total_targets,
    const uint32_t split_targets[CACHE_SPLIT_COUNT],
    const uint32_t first_values[4],
    const uint32_t last_values[4],
    uint64_t actual_file_size)
{
#ifdef STAGE4E_CACHE_A4
    return fprintf(report,
        "cache_format_version=%u\n"
        "header_size=%u\n"
        "entry_size=%u\n"
        "dataset_sha256=%s\n"
        "base_checkpoint_sha256=%s\n"
        "tokenizer_sha256=%s\n"
        "model_dim=%u\n"
        "vocab_size=%u\n"
        "record_count=%u\n"
        "target_row_count=%u\n"
        "train_target_rows=%u\n"
        "validation_target_rows=%u\n"
        "test_target_rows=%u\n"
        "regression_target_rows=%u\n"
        "cache_file_size=%llu\n"
        "first_row_record_index=%u\n"
        "first_row_input_position=%u\n"
        "first_row_target_token_id=%u\n"
        "first_row_split_id=%u\n"
        "last_row_record_index=%u\n"
        "last_row_input_position=%u\n"
        "last_row_target_token_id=%u\n"
        "last_row_split_id=%u\n"
        "nonfinite_hidden_values=0\n",
        CACHE_FORMAT_VERSION, CACHE_HEADER_SIZE, CACHE_ENTRY_SIZE,
        dataset_sha, checkpoint_sha, tokenizer_sha, CACHE_DIM, CACHE_VOCAB,
        total_records, total_targets,
        split_targets[0], split_targets[1], split_targets[2], split_targets[3],
        (unsigned long long)actual_file_size,
        first_values[0], first_values[1], first_values[2], first_values[3],
        last_values[0], last_values[1], last_values[2], last_values[3]) >= 0;
#else
    return fprintf(report,
        "dataset_sha256=%s\n"
        "base_checkpoint_sha256=%s\n"
        "tokenizer_sha256=%s\n"
        "format_version=%u\n"
        "header_size=%u\n"
        "entry_size=%u\n"
        "total_records=%u\n"
        "cached_target_positions=%u\n"
        "train_target_positions=%u\n"
        "validation_target_positions=%u\n"
        "test_target_positions=%u\n"
        "regression_target_positions=%u\n"
        "first_entry.record_index=%u\n"
        "first_entry.input_position=%u\n"
        "first_entry.target_token_id=%u\n"
        "first_entry.split_id=%u\n"
        "last_entry.record_index=%u\n"
        "last_entry.input_position=%u\n"
        "last_entry.target_token_id=%u\n"
        "last_entry.split_id=%u\n"
        "expected_file_size=%u\n"
        "actual_file_size=%llu\n",
        dataset_sha, checkpoint_sha, tokenizer_sha,
        CACHE_FORMAT_VERSION, CACHE_HEADER_SIZE, CACHE_ENTRY_SIZE,
        total_records, total_targets,
        split_targets[0], split_targets[1], split_targets[2], split_targets[3],
        first_values[0], first_values[1], first_values[2], first_values[3],
        last_values[0], last_values[1], last_values[2], last_values[3],
        CACHE_FILE_SIZE, (unsigned long long)actual_file_size) >= 0;
    #endif
}

static int cache_make_temp_path(const char *path, char *out, size_t out_size)
{
    int n = snprintf(out, out_size, "%s%s", path, CACHE_TEMP_SUFFIX);
    return n >= 0 && (size_t)n < out_size;
}

static void cache_remove_temps(const char *cache_tmp, const char *report_tmp)
{
    if (cache_tmp != NULL)
        remove(cache_tmp);
    if (report_tmp != NULL)
        remove(report_tmp);
}

#ifdef STAGE4E_CACHE_A4
static uint64_t cache_kv_checksum(const Transformer *tr, int last_position)
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

static int cache_record0_context_check(Tokenizer *tokenizer,
    Transformer *transformer,
    stage4e_dataset_t *dataset)
{
    stage4e_record_t *record;
    uint64_t checksum;
    int position;
    int forward_calls = 0;

    if (dataset->count != CACHE_RECORDS ||
        strcmp(dataset->items[0].id, "svc-status-basic-train-001") != 0) {
        fprintf(stderr, "error: A4 record 0 identity/count mismatch\n");
        return 0;
    }
    record = &dataset->items[0];
    if (!stage4e_prepare_record_tokens(tokenizer, transformer, record)) {
        fprintf(stderr, "error: cannot tokenize A4 record 0\n");
        return 0;
    }
    clear_run_state(transformer);
    for (position = 0; position < 15; position++) {
        (void)forward(transformer, record->training_seq_tokens[position], position);
        forward_calls++;
    }
    checksum = cache_kv_checksum(transformer, 14);
    printf("record0.sequence_count=%d\n", record->training_seq_count);
    printf("record0.first_target_index=%d\n", record->first_target_index);
    printf("record0.forward_calls_before_position_15=%d\n", forward_calls);
    printf("record0.kv_checksum_before_position_15=%016llx\n",
        (unsigned long long)checksum);
    (void)forward(transformer, record->training_seq_tokens[15], 15);
    forward_calls++;
    printf("record0.forward_calls_before_first_target=%d\n", forward_calls);
    printf("record0.first_supervised_input_position=15\n");
    printf("record0.first_supervised_target_token_id=%d\n",
        record->training_seq_tokens[16]);
    printf("record0.first_row_record_index=0\n");
    printf("record0.first_row_input_position=15\n");
    printf("record0.first_row_target_token_id=%d\n",
        record->training_seq_tokens[16]);
    printf("record0.first_row_split_id=0\n");
    return record->training_seq_count == 19 &&
        record->first_target_index == 16 && forward_calls == 16 &&
        record->training_seq_tokens[16] == 2669 &&
        checksum == UINT64_C(0xabb568d3ab418288);
}
#endif

int main(int argc, char **argv)
{
    const char *checkpoint_path = NULL;
    const char *dataset_path = NULL;
    const char *tokenizer_path = NULL;
#ifdef STAGE4E_CACHE_A4
    const char *cache_path = "stage4e-target-cache-a4.bin";
    const char *report_path = "stage4e-c2a-a4-cache-report.txt";
    int record0_context_check = 0;
#else
    const char *cache_path = "stage4e-target-cache.bin";
    const char *report_path = "stage4e-cache-build-report.txt";
#endif
    char cache_tmp[1024];
    char report_tmp[1024];
    char dataset_sha_hex[65];
    char checkpoint_sha_hex[65];
    char tokenizer_sha_hex[65];
    unsigned char dataset_sha_raw[32];
    unsigned char checkpoint_sha_raw[32];
    unsigned char tokenizer_sha_raw[32];
    uint32_t split_records[CACHE_SPLIT_COUNT] = { 0U, 0U, 0U, 0U };
    uint32_t split_targets[CACHE_SPLIT_COUNT] = { 0U, 0U, 0U, 0U };
    uint32_t first_entry[4] = { 0U, 0U, 0U, 0U };
    uint32_t last_entry[4] = { 0U, 0U, 0U, 0U };
    uint32_t total_targets = 0U;
    uint32_t previous_record = 0U;
    uint32_t previous_position = 0U;
    uint32_t argi;
    uint32_t record_index;
    uint64_t actual_file_size = 0;
    FILE *cache_file = NULL;
    FILE *report_file = NULL;
    stage4e_dataset_t dataset;
    Transformer transformer;
    Tokenizer tokenizer;
    int have_previous_entry = 0;
    int success = 0;

    memset(&dataset, 0, sizeof(dataset));
    memset(&transformer, 0, sizeof(transformer));
    memset(&tokenizer, 0, sizeof(tokenizer));

    if (argc < 2) {
        cache_usage(argv[0]);
        return EXIT_FAILURE;
    }
    checkpoint_path = argv[1];
    for (argi = 2U; argi < (uint32_t)argc; argi++) {
        if (strcmp(argv[argi], "-d") == 0 && argi + 1U < (uint32_t)argc)
            dataset_path = argv[++argi];
        else if (strcmp(argv[argi], "-z") == 0 && argi + 1U < (uint32_t)argc)
            tokenizer_path = argv[++argi];
        else if (strcmp(argv[argi], "-o") == 0 && argi + 1U < (uint32_t)argc)
            cache_path = argv[++argi];
        else if (strcmp(argv[argi], "-r") == 0 && argi + 1U < (uint32_t)argc)
            report_path = argv[++argi];
#ifdef STAGE4E_CACHE_A4
        else if (strcmp(argv[argi], "--record0-context-check") == 0)
            record0_context_check = 1;
#endif
        else {
            cache_usage(argv[0]);
            return EXIT_FAILURE;
        }
    }
    if (dataset_path == NULL || tokenizer_path == NULL ||
        strcmp(cache_path, report_path) == 0) {
        cache_usage(argv[0]);
        return EXIT_FAILURE;
    }
    if (!cache_float_format_supported()) {
        fprintf(stderr, "error: host float is not IEEE-754 binary32\n");
        return EXIT_FAILURE;
    }
    if (!cache_make_temp_path(cache_path, cache_tmp, sizeof(cache_tmp)) ||
        !cache_make_temp_path(report_path, report_tmp, sizeof(report_tmp))) {
        fprintf(stderr, "error: output path too long\n");
        return EXIT_FAILURE;
    }

    if (!stage4_hash_file_sha256_hex(dataset_path, dataset_sha_hex) ||
        !stage4_hash_file_sha256_hex(checkpoint_path, checkpoint_sha_hex) ||
        !stage4_hash_file_sha256_hex(tokenizer_path, tokenizer_sha_hex)) {
        fprintf(stderr, "error: unable to hash one or more source inputs\n");
        goto cleanup;
    }
    if (strcmp(dataset_sha_hex, expected_dataset_sha256) != 0 ||
        strcmp(checkpoint_sha_hex, expected_checkpoint_sha256) != 0 ||
        strcmp(tokenizer_sha_hex, expected_tokenizer_sha256) != 0) {
        fprintf(stderr,
            "error: source identity mismatch\n"
            "dataset expected=%s actual=%s\n"
            "checkpoint expected=%s actual=%s\n"
            "tokenizer expected=%s actual=%s\n",
            expected_dataset_sha256, dataset_sha_hex,
            expected_checkpoint_sha256, checkpoint_sha_hex,
            expected_tokenizer_sha256, tokenizer_sha_hex);
        goto cleanup;
    }
    if (!cache_hex_to_raw(dataset_sha_hex, dataset_sha_raw) ||
        !cache_hex_to_raw(checkpoint_sha_hex, checkpoint_sha_raw) ||
        !cache_hex_to_raw(tokenizer_sha_hex, tokenizer_sha_raw)) {
        fprintf(stderr, "error: invalid source SHA-256 encoding\n");
        goto cleanup;
    }
    if (!stage4e_load_records(dataset_path, &dataset)) {
        fprintf(stderr, "error: cannot parse approved Stage 4E dataset\n");
        goto cleanup;
    }
    if (dataset.count != CACHE_RECORDS) {
        fprintf(stderr, "error: expected %u records, found %lu\n",
            CACHE_RECORDS, (unsigned long)dataset.count);
        goto cleanup;
    }
    if (!cache_make_temp_path(cache_path, cache_tmp, sizeof(cache_tmp)) ||
        !cache_make_temp_path(report_path, report_tmp, sizeof(report_tmp))) {
        fprintf(stderr, "error: output path too long\n");
        goto cleanup;
    }

    unload_adapter_runtime();
    load_transformer(&transformer, checkpoint_path);
    if ((uint32_t)transformer.config.dim != CACHE_DIM ||
        (uint32_t)transformer.config.vocab_size != CACHE_VOCAB) {
        fprintf(stderr, "error: checkpoint dimensions do not match cache format\n");
        goto cleanup;
    }
    load_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

#ifdef STAGE4E_CACHE_A4
    if (record0_context_check) {
        int passed = cache_record0_context_check(&tokenizer, &transformer, &dataset);
        stage4e_dataset_free(&dataset);
        free_tokenizer(&tokenizer);
        free_transformer(&transformer);
        printf("record0_full_context_check=%s\n", passed ? "PASS" : "FAIL");
        return passed ? EXIT_SUCCESS : EXIT_FAILURE;
    }
#endif

    cache_file = fopen(cache_tmp, "wb");
    if (cache_file == NULL) {
        fprintf(stderr, "error: cannot create cache temporary file\n");
        goto cleanup;
    }
    if (!cache_write_header(cache_file, dataset_sha_raw, checkpoint_sha_raw,
        tokenizer_sha_raw, expected_split_records, expected_split_targets)) {
        fprintf(stderr, "error: writing cache header failed\n");
        goto cleanup;
    }

    for (record_index = 0U; record_index < (uint32_t)dataset.count; record_index++) {
        stage4e_record_t *record = &dataset.items[record_index];
        int split_id = stage4e_split_index(record->split);
        int k;

        if (split_id < 0 || split_id >= (int)CACHE_SPLIT_COUNT) {
            fprintf(stderr, "error: invalid split in record %s\n", record->id);
            goto cleanup;
        }
        split_records[split_id]++;
        if (!stage4e_prepare_record_tokens(&tokenizer, &transformer, record)) {
            fprintf(stderr, "error: token preparation failed for record %s\n", record->id);
            goto cleanup;
        }

        clear_run_state(&transformer);
        for (k = 1; k < record->training_seq_count; k++) {
            int input_position = k - 1;
            int target_token_id = record->training_seq_tokens[k];
            const float *base_logits;
            uint32_t entry[4];
            int j;

            base_logits = forward(&transformer,
                record->training_seq_tokens[input_position], input_position);
            if (k < record->first_target_index)
                continue;
            if (target_token_id < 0 || target_token_id >= (int)CACHE_VOCAB) {
                fprintf(stderr,
                    "error: target token %d out of range for record %s position %d\n",
                    target_token_id, record->id, input_position);
                goto cleanup;
            }
            for (j = 0; j < (int)CACHE_DIM; j++) {
                if (!isfinite(transformer.state.x[j])) {
                    fprintf(stderr,
                        "error: non-finite hidden state for record %s position %d dim %d\n",
                        record->id, input_position, j);
                    goto cleanup;
                }
            }
            entry[0] = record_index;
            entry[1] = (uint32_t)input_position;
            entry[2] = (uint32_t)target_token_id;
            entry[3] = (uint32_t)split_id;
            if (have_previous_entry &&
                (entry[0] < previous_record ||
                (entry[0] == previous_record && entry[1] <= previous_position))) {
                fprintf(stderr, "error: cache entry order is not strictly ascending\n");
                goto cleanup;
            }
            if (!cache_write_entry(cache_file, entry[0], entry[1], entry[2],
                entry[3], transformer.state.x)) {
                fprintf(stderr, "error: writing cache entry failed\n");
                goto cleanup;
            }
            if (total_targets == 0U)
                memcpy(first_entry, entry, sizeof(first_entry));
            memcpy(last_entry, entry, sizeof(last_entry));
            previous_record = entry[0];
            previous_position = entry[1];
            have_previous_entry = 1;
            total_targets++;
            split_targets[split_id]++;

            (void)base_logits;
        }
    }

    if (total_targets != CACHE_TARGETS) {
        fprintf(stderr, "error: expected %u targets, wrote %u\n",
            CACHE_TARGETS, total_targets);
        goto cleanup;
    }
    for (argi = 0U; argi < CACHE_SPLIT_COUNT; argi++) {
        if (split_records[argi] != expected_split_records[argi] ||
            split_targets[argi] != expected_split_targets[argi]) {
            fprintf(stderr,
                "error: split %u count mismatch: records=%u targets=%u\n",
                argi, split_records[argi], split_targets[argi]);
            goto cleanup;
        }
    }
    if (fflush(cache_file) != 0) {
        fprintf(stderr, "error: flushing cache failed\n");
        goto cleanup;
    }
    if (fclose(cache_file) != 0) {
        cache_file = NULL;
        fprintf(stderr, "error: closing cache failed\n");
        goto cleanup;
    }
    cache_file = NULL;
    if (!stage4_file_size_bytes(cache_tmp, &actual_file_size) ||
        actual_file_size != CACHE_FILE_SIZE) {
        fprintf(stderr,
            "error: cache size mismatch: expected %u actual %llu\n",
            CACHE_FILE_SIZE, (unsigned long long)actual_file_size);
        goto cleanup;
    }

    report_file = fopen(report_tmp, "wb");
    if (report_file == NULL) {
        fprintf(stderr, "error: cannot create report temporary file\n");
        goto cleanup;
    }
    if (!cache_record_report(report_file, dataset_sha_hex, checkpoint_sha_hex,
        tokenizer_sha_hex, (uint32_t)dataset.count, total_targets, split_targets,
        first_entry, last_entry, actual_file_size)) {
        fprintf(stderr, "error: writing cache report failed\n");
        goto cleanup;
    }
    if (fflush(report_file) != 0) {
        fprintf(stderr, "error: flushing cache report failed\n");
        goto cleanup;
    }
    if (fclose(report_file) != 0) {
        report_file = NULL;
        fprintf(stderr, "error: closing cache report failed\n");
        goto cleanup;
    }
    report_file = NULL;

    if (rename(cache_tmp, cache_path) != 0) {
        fprintf(stderr, "error: replacing cache output failed\n");
        goto cleanup;
    }
    if (rename(report_tmp, report_path) != 0) {
        fprintf(stderr, "error: replacing report output failed\n");
        goto cleanup;
    }

    printf("cache_records=%u\n", (uint32_t)dataset.count);
    printf("cache_target_positions=%u\n", total_targets);
    printf("train_target_positions=%u\n", split_targets[0]);
    printf("validation_target_positions=%u\n", split_targets[1]);
    printf("test_target_positions=%u\n", split_targets[2]);
    printf("regression_target_positions=%u\n", split_targets[3]);
    printf("expected_cache_bytes=%u\n", CACHE_FILE_SIZE);
    printf("actual_cache_bytes=%llu\n", (unsigned long long)actual_file_size);
    success = 1;

cleanup:
    if (cache_file != NULL)
        fclose(cache_file);
    if (report_file != NULL)
        fclose(report_file);
    if (!success)
        cache_remove_temps(cache_tmp, report_tmp);
    stage4e_dataset_free(&dataset);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
