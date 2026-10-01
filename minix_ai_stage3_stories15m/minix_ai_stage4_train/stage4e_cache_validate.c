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
#ifdef STAGE4E_CACHE_VALIDATE_A4
#define CACHE_TARGETS 165U
#define CACHE_FILE_SIZE 192892U
#else
#define CACHE_TARGETS 156U
#define CACHE_FILE_SIZE 182380U
#endif
#define CACHE_SPLIT_COUNT 4U
#define CHECKPOINT_HEADER_BYTES 28U

typedef struct {
    uint32_t record_index;
    uint32_t input_position;
    uint32_t target_token_id;
    uint32_t split_id;
    int seen;
} expected_entry_t;

typedef struct {
    uint32_t record_index;
    uint32_t input_position;
    uint32_t target_token_id;
    uint32_t split_id;
} actual_entry_t;

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
} decoded_header_t;

typedef struct {
    uint32_t invalid_record_indices;
    uint32_t invalid_input_positions;
    uint32_t invalid_target_token_ids;
    uint32_t target_token_mismatches;
    uint32_t split_mismatches;
    uint32_t record_index_mismatches;
    uint32_t input_position_mismatches;
    uint32_t split_id_mismatches;
    uint32_t row_order_mismatches;
    uint32_t nonfinite_hidden_values;
    uint32_t ordering_errors;
    uint32_t duplicate_entries;
    uint32_t unexpected_entries;
    uint32_t missing_entries;
    uint32_t validated_entries;
    uint32_t expected_entries;
    uint32_t header_errors;
    uint32_t identity_errors;
    uint32_t file_size_errors;
    uint32_t trailing_byte_errors;
    actual_entry_t first_entry;
    actual_entry_t last_entry;
    int have_entry;
} validation_stats_t;

static const unsigned char cache_magic[8] = { 'S', '4', 'T', 'C', 'A', 'C', 'H', '1' };
static const char expected_dataset_sha256[] =
#ifdef STAGE4E_CACHE_VALIDATE_A4
    "71744cea4de049e2e192cb2a4494c8a4a68e769b3a7f3b8ed4880433e4583b84";
#else
    "b8aa6dc1dbfad7ed144e19a351850189956f79571b05cb70324d21876db0ea78";
#endif
static const char expected_checkpoint_sha256[] =
    "cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a";
static const char expected_tokenizer_sha256[] =
    "50a52ef822ee9e83de5ce9d0be0a025a773d019437f58b5ff9dcafb063ece361";
static const uint32_t expected_split_records[CACHE_SPLIT_COUNT] = { 20U, 5U, 5U, 0U };
#ifdef STAGE4E_CACHE_VALIDATE_A4
static const uint32_t expected_split_targets[CACHE_SPLIT_COUNT] = { 111U, 33U, 21U, 0U };
#else
static const uint32_t expected_split_targets[CACHE_SPLIT_COUNT] = { 111U, 24U, 21U, 0U };
#endif

static int cache_read_exact(FILE *file, void *buffer, size_t size)
{
    return size == 0U || fread(buffer, 1, size, file) == size;
}

static uint32_t decode_u32_le(const unsigned char bytes[4])
{
    return (uint32_t)bytes[0] |
        ((uint32_t)bytes[1] << 8) |
        ((uint32_t)bytes[2] << 16) |
        ((uint32_t)bytes[3] << 24);
}

static int read_u32_le(FILE *file, uint32_t *value)
{
    unsigned char bytes[4];
    if (!cache_read_exact(file, bytes, sizeof(bytes)))
        return 0;
    *value = decode_u32_le(bytes);
    return 1;
}

static int read_f32_le(FILE *file, float *value)
{
    uint32_t bits;
    if (!read_u32_le(file, &bits))
        return 0;
    memcpy(value, &bits, sizeof(bits));
    return 1;
}

static int read_digest(FILE *file, unsigned char digest[32])
{
    return cache_read_exact(file, digest, 32U);
}

static int validation_hex_value(char ch)
{
    if (ch >= '0' && ch <= '9')
        return ch - '0';
    if (ch >= 'a' && ch <= 'f')
        return ch - 'a' + 10;
    if (ch >= 'A' && ch <= 'F')
        return ch - 'A' + 10;
    return -1;
}

static int validation_hex_to_raw(const char *hex, unsigned char raw[32])
{
    size_t i;

    if (hex == NULL || strlen(hex) != 64U)
        return 0;
    for (i = 0; i < 32U; i++) {
        int high = validation_hex_value(hex[i * 2U]);
        int low = validation_hex_value(hex[i * 2U + 1U]);
        if (high < 0 || low < 0)
            return 0;
        raw[i] = (unsigned char)((high << 4) | low);
    }
    return 1;
}

static int float_format_supported(void)
{
    float one = 1.0f;
    uint32_t bits = 0U;
    const uint16_t endian_test = 1U;

    if (sizeof(float) != 4U || sizeof(uint32_t) != 4U ||
        FLT_RADIX != 2 || FLT_MANT_DIG != 24 || FLT_MAX_EXP != 128 ||
        FLT_MIN_EXP != -125 || *((const unsigned char *)&endian_test) != 1U)
        return 0;
    memcpy(&bits, &one, sizeof(bits));
    return bits == UINT32_C(0x3f800000);
}

static int read_header(FILE *file, decoded_header_t *header)
{
    unsigned char magic[8];
    uint32_t i;

    memset(header, 0, sizeof(*header));
    if (!cache_read_exact(file, magic, sizeof(magic)) ||
        memcmp(magic, cache_magic, sizeof(magic)) != 0)
        return 0;
    if (!read_u32_le(file, &header->format_version) ||
        !read_u32_le(file, &header->header_size) ||
        !read_u32_le(file, &header->entry_size) ||
        !read_u32_le(file, &header->float_format) ||
        !read_u32_le(file, &header->byte_order) ||
        !read_u32_le(file, &header->model_dim) ||
        !read_u32_le(file, &header->vocab_size) ||
        !read_u32_le(file, &header->record_count) ||
        !read_u32_le(file, &header->target_count))
        return 0;
    for (i = 0; i < CACHE_SPLIT_COUNT; i++) {
        if (!read_u32_le(file, &header->split_records[i]))
            return 0;
    }
    for (i = 0; i < CACHE_SPLIT_COUNT; i++) {
        if (!read_u32_le(file, &header->split_targets[i]))
            return 0;
    }
    return read_digest(file, header->dataset_sha) &&
        read_digest(file, header->checkpoint_sha) &&
        read_digest(file, header->tokenizer_sha);
}

static int header_fields_valid(const decoded_header_t *header)
{
    uint32_t i;

    if (header->format_version != CACHE_FORMAT_VERSION ||
        header->header_size != CACHE_HEADER_SIZE ||
        header->entry_size != CACHE_ENTRY_SIZE ||
        header->float_format != 1U || header->byte_order != 1U ||
        header->model_dim != CACHE_DIM || header->vocab_size != CACHE_VOCAB ||
        header->record_count != CACHE_RECORDS ||
        header->target_count != CACHE_TARGETS)
        return 0;
    for (i = 0; i < CACHE_SPLIT_COUNT; i++) {
        if (header->split_records[i] != expected_split_records[i] ||
            header->split_targets[i] != expected_split_targets[i])
            return 0;
    }
    return 1;
}

static int read_checkpoint_config(const char *path, int *seq_len)
{
    FILE *file;
    unsigned char bytes[CHECKPOINT_HEADER_BYTES];
    uint32_t dim;
    uint32_t raw_vocab;
    uint32_t vocab_size;
    uint32_t sequence_length;

    file = fopen(path, "rb");
    if (file == NULL)
        return 0;
    if (!cache_read_exact(file, bytes, sizeof(bytes))) {
        fclose(file);
        return 0;
    }
    if (fclose(file) != 0)
        return 0;
    dim = decode_u32_le(bytes);
    raw_vocab = decode_u32_le(bytes + 20U);
    sequence_length = decode_u32_le(bytes + 24U);
    if (dim != CACHE_DIM || sequence_length == 0U || sequence_length > 2147483647U)
        return 0;
    vocab_size = (raw_vocab & UINT32_C(0x80000000)) != 0U ?
        (~raw_vocab + 1U) : raw_vocab;
    if (vocab_size != CACHE_VOCAB)
        return 0;
    *seq_len = (int)sequence_length;
    return 1;
}

static int derive_expected_entries(Tokenizer *tokenizer,
    Transformer *transformer,
    stage4e_dataset_t *dataset,
    expected_entry_t entries[CACHE_TARGETS],
    uint32_t split_records[CACHE_SPLIT_COUNT],
    uint32_t split_targets[CACHE_SPLIT_COUNT],
    uint32_t *entry_count)
{
    uint32_t record_index;
    uint32_t count = 0U;

    memset(split_records, 0, CACHE_SPLIT_COUNT * sizeof(uint32_t));
    memset(split_targets, 0, CACHE_SPLIT_COUNT * sizeof(uint32_t));
    for (record_index = 0U; record_index < (uint32_t)dataset->count; record_index++) {
        stage4e_record_t *record = &dataset->items[record_index];
        int split_id = stage4e_split_index(record->split);
        int k;

        if (split_id < 0 || split_id >= (int)CACHE_SPLIT_COUNT)
            return 0;
        split_records[split_id]++;
        if (!stage4e_prepare_record_tokens(tokenizer, transformer, record))
            return 0;
        for (k = record->first_target_index; k < record->training_seq_count; k++) {
            if (count >= CACHE_TARGETS)
                return 0;
            entries[count].record_index = record_index;
            entries[count].input_position = (uint32_t)(k - 1);
            entries[count].target_token_id = (uint32_t)record->training_seq_tokens[k];
            entries[count].split_id = (uint32_t)split_id;
            entries[count].seen = 0;
            count++;
            split_targets[split_id]++;
        }
    }
    *entry_count = count;
    return count == CACHE_TARGETS;
}

static int find_expected_entry(expected_entry_t entries[CACHE_TARGETS],
    uint32_t count,
    uint32_t record_index,
    uint32_t input_position)
{
    uint32_t i;
    for (i = 0U; i < count; i++) {
        if (entries[i].record_index == record_index &&
            entries[i].input_position == input_position)
            return (int)i;
    }
    return -1;
}

static int read_and_validate_entries(FILE *file,
    expected_entry_t expected[CACHE_TARGETS],
    uint32_t expected_count,
    const stage4e_dataset_t *dataset,
    validation_stats_t *stats)
{
    uint32_t entry_number;
    int have_previous = 0;
    actual_entry_t previous;

    memset(&previous, 0, sizeof(previous));
    for (entry_number = 0U; entry_number < CACHE_TARGETS; entry_number++) {
        actual_entry_t entry;
        uint32_t hidden_index;
        int expected_index;

        if (!read_u32_le(file, &entry.record_index) ||
            !read_u32_le(file, &entry.input_position) ||
            !read_u32_le(file, &entry.target_token_id) ||
            !read_u32_le(file, &entry.split_id))
            return 0;
        if (entry.record_index != expected[entry_number].record_index)
            stats->record_index_mismatches++;
        if (entry.input_position != expected[entry_number].input_position)
            stats->input_position_mismatches++;
        if (entry.target_token_id != expected[entry_number].target_token_id)
            stats->target_token_mismatches++;
        if (entry.split_id != expected[entry_number].split_id)
            stats->split_id_mismatches++;
        if (entry.record_index != expected[entry_number].record_index ||
            entry.input_position != expected[entry_number].input_position ||
            entry.target_token_id != expected[entry_number].target_token_id ||
            entry.split_id != expected[entry_number].split_id)
            stats->row_order_mismatches++;
        for (hidden_index = 0U; hidden_index < CACHE_DIM; hidden_index++) {
            float hidden_value;
            if (!read_f32_le(file, &hidden_value))
                return 0;
            if (!isfinite(hidden_value))
                stats->nonfinite_hidden_values++;
        }
        stats->validated_entries++;
        if (!stats->have_entry) {
            stats->first_entry = entry;
            stats->have_entry = 1;
        }
        stats->last_entry = entry;

        if (have_previous &&
            (entry.record_index < previous.record_index ||
            (entry.record_index == previous.record_index &&
            entry.input_position <= previous.input_position)))
            stats->ordering_errors++;
        previous = entry;
        have_previous = 1;

        if (entry.target_token_id >= CACHE_VOCAB)
            stats->invalid_target_token_ids++;
        if (entry.split_id >= CACHE_SPLIT_COUNT)
            stats->split_mismatches++;
        if (entry.record_index >= CACHE_RECORDS) {
            stats->invalid_record_indices++;
            stats->unexpected_entries++;
            continue;
        }
        {
            const stage4e_record_t *record = &dataset->items[entry.record_index];
            if (entry.split_id < CACHE_SPLIT_COUNT &&
                entry.split_id != (uint32_t)stage4e_split_index(record->split))
                stats->split_mismatches++;
            if (record->training_seq_count < 2 ||
                entry.input_position >= (uint32_t)(record->training_seq_count - 1) ||
                entry.input_position + 1U < (uint32_t)record->first_target_index) {
                stats->invalid_input_positions++;
                stats->unexpected_entries++;
                continue;
            }
        }
        expected_index = find_expected_entry(expected, expected_count,
            entry.record_index, entry.input_position);
        if (expected_index < 0) {
            stats->unexpected_entries++;
            continue;
        }
        if (expected[expected_index].seen) {
            stats->duplicate_entries++;
        } else {
            expected[expected_index].seen = 1;
        }
        if (entry.split_id < CACHE_SPLIT_COUNT &&
            entry.split_id != expected[expected_index].split_id)
            stats->split_mismatches++;
    }
    return 1;
}

static int report_validation(const char *report_path,
    const char *cache_sha,
    const char *dataset_sha,
    const char *checkpoint_sha,
    const char *tokenizer_sha,
    int header_valid,
    int file_size_valid,
    int source_identities_valid,
    uint64_t actual_file_size,
    const validation_stats_t *stats)
{
    FILE *report = fopen(report_path, "wb");
    int ok;

    if (report == NULL)
        return 0;
#ifdef STAGE4E_CACHE_VALIDATE_A4
    ok = fprintf(report,
        "cache_sha256=%s\n"
        "dataset_sha256=%s\n"
        "base_checkpoint_sha256=%s\n"
        "tokenizer_sha256=%s\n"
        "cache_format_version=%u\n"
        "header_size=%u\n"
        "entry_size=%u\n"
        "model_dim=%u\n"
        "vocab_size=%u\n"
        "record_count=%u\n"
        "target_row_count=%u\n"
        "train_target_rows=%u\n"
        "validation_target_rows=%u\n"
        "test_target_rows=%u\n"
        "regression_target_rows=%u\n"
        "cache_file_size=%llu\n"
        "header_errors=%u\n"
        "identity_errors=%u\n"
        "file_size_errors=%u\n"
        "trailing_byte_errors=%u\n"
        "validated_rows=%u\n"
        "expected_rows=%u\n"
        "duplicate_cache_rows=%u\n"
        "missing_cache_rows=%u\n"
        "unexpected_cache_rows=%u\n"
        "record_index_mismatches=%u\n"
        "input_position_mismatches=%u\n"
        "target_token_id_mismatches=%u\n"
        "split_id_mismatches=%u\n"
        "row_order_mismatches=%u\n"
        "nonfinite_hidden_values=%u\n"
        "first_row_record_index=%u\n"
        "first_row_input_position=%u\n"
        "first_row_target_token_id=%u\n"
        "first_row_split_id=%u\n"
        "last_row_record_index=%u\n"
        "last_row_input_position=%u\n"
        "last_row_target_token_id=%u\n"
        "last_row_split_id=%u\n",
        cache_sha, dataset_sha, checkpoint_sha, tokenizer_sha,
        CACHE_FORMAT_VERSION, CACHE_HEADER_SIZE, CACHE_ENTRY_SIZE,
        CACHE_DIM, CACHE_VOCAB, CACHE_RECORDS, CACHE_TARGETS,
        expected_split_targets[0], expected_split_targets[1],
        expected_split_targets[2], expected_split_targets[3],
        (unsigned long long)actual_file_size, stats->header_errors, stats->identity_errors,
        stats->file_size_errors, stats->trailing_byte_errors,
        stats->validated_entries, stats->expected_entries,
        stats->duplicate_entries, stats->missing_entries,
        stats->unexpected_entries, stats->record_index_mismatches,
        stats->input_position_mismatches, stats->target_token_mismatches,
        stats->split_id_mismatches, stats->row_order_mismatches,
        stats->nonfinite_hidden_values,
        stats->first_entry.record_index, stats->first_entry.input_position,
        stats->first_entry.target_token_id, stats->first_entry.split_id,
        stats->last_entry.record_index, stats->last_entry.input_position,
        stats->last_entry.target_token_id, stats->last_entry.split_id) >= 0;
#else
    ok = fprintf(report,
        "cache_sha256=%s\n"
        "dataset_sha256=%s\n"
        "checkpoint_sha256=%s\n"
        "tokenizer_sha256=%s\n"
        "header_valid=%s\n"
        "file_size_valid=%s\n"
        "source_identities_valid=%s\n"
        "validated_entries=%u\n"
        "expected_entries=%u\n"
        "duplicate_entries=%u\n"
        "missing_entries=%u\n"
        "unexpected_entries=%u\n"
        "invalid_record_indices=%u\n"
        "invalid_input_positions=%u\n"
        "invalid_target_token_ids=%u\n"
        "target_token_mismatches=%u\n"
        "split_mismatches=%u\n"
        "nonfinite_hidden_values=%u\n"
        "ordering_errors=%u\n"
        "first_entry.record_index=%u\n"
        "first_entry.input_position=%u\n"
        "first_entry.target_token_id=%u\n"
        "first_entry.split_id=%u\n"
        "last_entry.record_index=%u\n"
        "last_entry.input_position=%u\n"
        "last_entry.target_token_id=%u\n"
        "last_entry.split_id=%u\n",
        cache_sha, dataset_sha, checkpoint_sha, tokenizer_sha,
        header_valid ? "yes" : "no", file_size_valid ? "yes" : "no",
        source_identities_valid ? "yes" : "no",
        stats->validated_entries, stats->expected_entries,
        stats->duplicate_entries, stats->missing_entries,
        stats->unexpected_entries, stats->invalid_record_indices,
        stats->invalid_input_positions, stats->invalid_target_token_ids,
        stats->target_token_mismatches, stats->split_mismatches,
        stats->nonfinite_hidden_values, stats->ordering_errors,
        stats->first_entry.record_index, stats->first_entry.input_position,
        stats->first_entry.target_token_id, stats->first_entry.split_id,
        stats->last_entry.record_index, stats->last_entry.input_position,
        stats->last_entry.target_token_id, stats->last_entry.split_id) >= 0;
    #endif
    if (fclose(report) != 0)
        ok = 0;
    return ok;
}

static void validation_usage(const char *program)
{
    fprintf(stderr,
        "Usage: %s <checkpoint.bin> -d <approved.records> -z <tokenizer.bin> "
        "-c <cache.bin> -r <report.txt>\n",
        program);
}

int main(int argc, char **argv)
{
    const char *checkpoint_path = NULL;
    const char *dataset_path = NULL;
    const char *tokenizer_path = NULL;
#ifdef STAGE4E_CACHE_VALIDATE_A4
    const char *cache_path = "stage4e-target-cache-a4.bin";
    const char *report_path = "stage4e-c2b-a4-cache-validation-report.txt";
#else
    const char *cache_path = "stage4e-target-cache.bin";
    const char *report_path = "stage4e-cache-validation-report.txt";
#endif
    char dataset_sha[65] = "";
    char checkpoint_sha[65] = "";
    char tokenizer_sha[65] = "";
    char cache_sha[65] = "";
    unsigned char actual_dataset_raw[32];
    unsigned char actual_checkpoint_raw[32];
    unsigned char actual_tokenizer_raw[32];
    FILE *cache_file = NULL;
    decoded_header_t header;
    expected_entry_t expected[CACHE_TARGETS];
    validation_stats_t stats;
    stage4e_dataset_t dataset;
    Transformer transformer;
    Tokenizer tokenizer;
    uint32_t observed_split_records[CACHE_SPLIT_COUNT] = { 0U, 0U, 0U, 0U };
    uint32_t observed_split_targets[CACHE_SPLIT_COUNT] = { 0U, 0U, 0U, 0U };
    uint32_t expected_count = 0U;
    uint64_t cache_size = 0U;
    int seq_len = 0;
    int cache_exists = 0;
    int header_read_ok = 0;
    int header_valid = 0;
    int file_size_valid = 0;
    int source_identities_valid = 0;
    int input_identities_valid = 0;
    int dataset_hash_ok = 0;
    int checkpoint_hash_ok = 0;
    int tokenizer_hash_ok = 0;
    int dataset_ready = 0;
    int float_ok = 0;
    int argi;
    int result = EXIT_FAILURE;
    uint32_t i;

    memset(&header, 0, sizeof(header));
    memset(&stats, 0, sizeof(stats));
    memset(&dataset, 0, sizeof(dataset));
    memset(&transformer, 0, sizeof(transformer));
    memset(&tokenizer, 0, sizeof(tokenizer));
    memset(expected, 0, sizeof(expected));
    memset(actual_dataset_raw, 0, sizeof(actual_dataset_raw));
    memset(actual_checkpoint_raw, 0, sizeof(actual_checkpoint_raw));
    memset(actual_tokenizer_raw, 0, sizeof(actual_tokenizer_raw));
    stats.expected_entries = CACHE_TARGETS;

    if (argc < 2) {
        validation_usage(argv[0]);
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
        else {
            validation_usage(argv[0]);
            return EXIT_FAILURE;
        }
    }
    if (dataset_path == NULL || tokenizer_path == NULL ||
        strcmp(cache_path, report_path) == 0) {
        validation_usage(argv[0]);
        return EXIT_FAILURE;
    }

    float_ok = float_format_supported();
    dataset_hash_ok = stage4_hash_file_sha256_hex(dataset_path, dataset_sha);
    checkpoint_hash_ok = stage4_hash_file_sha256_hex(checkpoint_path, checkpoint_sha);
    tokenizer_hash_ok = stage4_hash_file_sha256_hex(tokenizer_path, tokenizer_sha);
    stage4_hash_file_sha256_hex(cache_path, cache_sha);
    input_identities_valid = dataset_hash_ok && checkpoint_hash_ok && tokenizer_hash_ok &&
        strcmp(dataset_sha, expected_dataset_sha256) == 0 &&
        strcmp(checkpoint_sha, expected_checkpoint_sha256) == 0 &&
        strcmp(tokenizer_sha, expected_tokenizer_sha256) == 0 &&
        validation_hex_to_raw(dataset_sha, actual_dataset_raw) &&
        validation_hex_to_raw(checkpoint_sha, actual_checkpoint_raw) &&
        validation_hex_to_raw(tokenizer_sha, actual_tokenizer_raw);
    source_identities_valid = input_identities_valid;
    if (!input_identities_valid)
        stats.identity_errors++;
    cache_exists = stage4_file_size_bytes(cache_path, &cache_size);
    file_size_valid = cache_exists && cache_size == CACHE_FILE_SIZE;
    if (!file_size_valid)
        stats.file_size_errors++;

    if (cache_exists) {
        cache_file = fopen(cache_path, "rb");
        if (cache_file != NULL) {
            header_read_ok = read_header(cache_file, &header);
            if (header_read_ok) {
                int header_identities_valid =
                    memcmp(header.dataset_sha, actual_dataset_raw, 32U) == 0 &&
                    memcmp(header.checkpoint_sha, actual_checkpoint_raw, 32U) == 0 &&
                    memcmp(header.tokenizer_sha, actual_tokenizer_raw, 32U) == 0;
                int header_metadata_valid = header_fields_valid(&header) && float_ok;
                header_valid = input_identities_valid && header_metadata_valid &&
                    header_identities_valid;
                source_identities_valid = input_identities_valid &&
                    header_identities_valid;
                if (!header_metadata_valid)
                    stats.header_errors++;
                if (!header_identities_valid)
                    stats.identity_errors++;
            } else {
                source_identities_valid = 0;
                stats.header_errors++;
            }
        } else {
            source_identities_valid = 0;
            stats.header_errors++;
        }
    } else {
        source_identities_valid = 0;
        stats.header_errors++;
    }

    if (source_identities_valid && cache_file != NULL &&
        read_checkpoint_config(checkpoint_path, &seq_len)) {
        transformer.config.dim = CACHE_DIM;
        transformer.config.vocab_size = CACHE_VOCAB;
        transformer.config.seq_len = seq_len;
        load_tokenizer(&tokenizer, tokenizer_path, CACHE_VOCAB);
        if (stage4e_load_records(dataset_path, &dataset) &&
            dataset.count == CACHE_RECORDS &&
            derive_expected_entries(&tokenizer, &transformer, &dataset,
                expected, observed_split_records, observed_split_targets,
                &expected_count)) {
            dataset_ready = 1;
            for (i = 0U; i < CACHE_SPLIT_COUNT; i++) {
                if (observed_split_records[i] != expected_split_records[i] ||
                    observed_split_targets[i] != expected_split_targets[i])
                    dataset_ready = 0;
            }
        }
    }

    if (header_read_ok && dataset_ready && float_ok && cache_file != NULL) {
        if (!read_and_validate_entries(cache_file, expected, expected_count,
            &dataset, &stats))
            file_size_valid = 0;
        for (i = 0U; i < expected_count; i++) {
            if (!expected[i].seen)
                stats.missing_entries++;
        }
        if (fgetc(cache_file) != EOF) {
            stats.trailing_byte_errors++;
            file_size_valid = 0;
        }
    }

    if (cache_file != NULL) {
        if (fclose(cache_file) != 0) {
            file_size_valid = 0;
            stats.file_size_errors++;
        }
        cache_file = NULL;
    }
    if (!file_size_valid && stats.file_size_errors == 0U)
        stats.file_size_errors++;

    if (!report_validation(report_path, cache_sha, dataset_sha,
        checkpoint_sha, tokenizer_sha, header_valid, file_size_valid,
        source_identities_valid, cache_size, &stats)) {
        fprintf(stderr, "error: cannot write validation report '%s'\n", report_path);
        goto cleanup;
    }

#ifdef STAGE4E_CACHE_VALIDATE_A4
    result = header_valid && file_size_valid && source_identities_valid &&
        dataset_ready && stats.validated_entries == CACHE_TARGETS &&
        stats.expected_entries == CACHE_TARGETS && stats.duplicate_entries == 0U &&
        stats.missing_entries == 0U && stats.unexpected_entries == 0U &&
        stats.record_index_mismatches == 0U && stats.input_position_mismatches == 0U &&
        stats.target_token_mismatches == 0U && stats.split_id_mismatches == 0U &&
        stats.row_order_mismatches == 0U && stats.nonfinite_hidden_values == 0U &&
        stats.header_errors == 0U && stats.identity_errors == 0U &&
        stats.file_size_errors == 0U && stats.trailing_byte_errors == 0U ?
        EXIT_SUCCESS : EXIT_FAILURE;
#else
    result = header_valid && file_size_valid && source_identities_valid &&
        dataset_ready && stats.validated_entries == CACHE_TARGETS &&
        stats.expected_entries == CACHE_TARGETS && stats.duplicate_entries == 0U &&
        stats.missing_entries == 0U && stats.unexpected_entries == 0U &&
        stats.invalid_record_indices == 0U && stats.invalid_input_positions == 0U &&
        stats.invalid_target_token_ids == 0U && stats.target_token_mismatches == 0U &&
        stats.split_mismatches == 0U && stats.nonfinite_hidden_values == 0U &&
        stats.ordering_errors == 0U ? EXIT_SUCCESS : EXIT_FAILURE;
#endif

    printf("header_valid=%s\n", header_valid ? "yes" : "no");
    printf("file_size_valid=%s\n", file_size_valid ? "yes" : "no");
    printf("source_identities_valid=%s\n", source_identities_valid ? "yes" : "no");
    printf("validated_entries=%u\n", stats.validated_entries);
    printf("expected_entries=%u\n", stats.expected_entries);
    printf("duplicate_entries=%u\n", stats.duplicate_entries);
    printf("missing_entries=%u\n", stats.missing_entries);
    printf("unexpected_entries=%u\n", stats.unexpected_entries);
    printf("invalid_record_indices=%u\n", stats.invalid_record_indices);
    printf("invalid_input_positions=%u\n", stats.invalid_input_positions);
    printf("invalid_target_token_ids=%u\n", stats.invalid_target_token_ids);
    printf("target_token_mismatches=%u\n", stats.target_token_mismatches);
    printf("split_mismatches=%u\n", stats.split_mismatches);
    printf("nonfinite_hidden_values=%u\n", stats.nonfinite_hidden_values);
    printf("ordering_errors=%u\n", stats.ordering_errors);

cleanup:
    if (cache_file != NULL)
        fclose(cache_file);
    stage4e_dataset_free(&dataset);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return result;
}
