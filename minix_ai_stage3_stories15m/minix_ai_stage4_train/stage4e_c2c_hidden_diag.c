#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STAGE4E_GENERATION_EMBEDDED
#include "stage4e_baseline_unadapted_generation.c"
#undef STAGE4E_GENERATION_EMBEDDED

#define DIAG_RECORD_ID "svc-status-basic-train-001"
#define DIAG_LAST_POSITION 15

static void print_escaped(const char *label, const char *text)
{
    const unsigned char *p = (const unsigned char *)text;
    printf("%s=", label);
    while (*p != '\0') {
        if (*p == '\n')
            printf("\\n");
        else if (*p == '\r')
            printf("\\r");
        else if (*p == '\t')
            printf("\\t");
        else
            putchar(*p);
        p++;
    }
    putchar('\n');
}

static uint32_t float_bits(float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static uint64_t kv_checksum(const Transformer *transformer, int last_position)
{
    const Config *config = &transformer->config;
    const RunState *state = &transformer->state;
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
                uint32_t key_bits = float_bits(state->key_cache[position_offset + index]);
                uint32_t value_bits = float_bits(state->value_cache[position_offset + index]);
                hash ^= key_bits;
                hash *= UINT64_C(1099511628211);
                hash ^= value_bits;
                hash *= UINT64_C(1099511628211);
            }
        }
    }
    return hash;
}

static void print_hidden_samples(const char *path,
    int position,
    int input_token,
    int target_token,
    const float *hidden)
{
    static const int dimensions[] = { 0, 1, 2, 287 };
    size_t i;

    printf("path=%s position=%d input_token_id=%d next_target_token_id=%d\n",
        path, position, input_token, target_token);
    for (i = 0; i < sizeof(dimensions) / sizeof(dimensions[0]); i++) {
        int dimension = dimensions[i];
        float value = hidden[dimension];
        printf("  state.x[%d].bits=%08x value=%.9g\n",
            dimension, (unsigned int)float_bits(value), value);
    }
}

int main(int argc, char **argv)
{
    const char *checkpoint_path;
    const char *dataset_path;
    const char *tokenizer_path;
    stage4e_dataset_t dataset;
    Transformer transformer;
    Tokenizer tokenizer;
    stage4e_record_t *record = NULL;
    size_t record_index;
    int token_index;
    int found = 0;
    int k;

    if (argc != 4) {
        fprintf(stderr, "usage: %s checkpoint.bin approved.records tokenizer.bin\n", argv[0]);
        return EXIT_FAILURE;
    }
    checkpoint_path = argv[1];
    dataset_path = argv[2];
    tokenizer_path = argv[3];
    memset(&dataset, 0, sizeof(dataset));
    memset(&transformer, 0, sizeof(transformer));
    memset(&tokenizer, 0, sizeof(tokenizer));

    unload_adapter_runtime();
    load_transformer(&transformer, checkpoint_path);
    load_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);
    if (!stage4e_load_records(dataset_path, &dataset) || dataset.count != 30U) {
        fprintf(stderr, "error: could not load the 30 approved records\n");
        goto fail;
    }
    for (record_index = 0; record_index < dataset.count; record_index++) {
        if (strcmp(dataset.items[record_index].id, DIAG_RECORD_ID) == 0) {
            record = &dataset.items[record_index];
            found = 1;
            break;
        }
    }
    if (!found || record_index != 0U ||
        !stage4e_prepare_record_tokens(&tokenizer, &transformer, record)) {
        fprintf(stderr, "error: record 0 did not match the expected approved record\n");
        goto fail;
    }

    printf("record_index=0\nrecord_id=%s\n", record->id);
    printf("expected_visible_target=%s\n", record->target_visible);
    print_escaped("canonical_prompt_text", record->canonical_prompt);
    print_escaped("canonical_training_text", record->canonical_training_seq);
    printf("sequence_token_count=%d\n", record->training_seq_count);
    printf("first_target_index=%d\n", record->first_target_index);
    for (token_index = 0; token_index <= 16; token_index++) {
        if (token_index >= record->training_seq_count) {
            fprintf(stderr, "error: training sequence shorter than position 16\n");
            goto fail;
        }
        printf("sequence_token[%d]=%d\n", token_index,
            record->training_seq_tokens[token_index]);
    }

    printf("\npath=cache_writer_forward_trace\n");
    clear_run_state(&transformer);
    printf("initial_state.x0_bits=%08x value=%.9g\n",
        (unsigned int)float_bits(transformer.state.x[0]), transformer.state.x[0]);
    printf("initial_kv_checksum_positions_0_14=%016llx\n",
        (unsigned long long)kv_checksum(&transformer, 14));
    for (k = 1; k < record->training_seq_count; k++) {
        int position = k - 1;
        int input_token = record->training_seq_tokens[position];
        int target_token = record->training_seq_tokens[k];
        (void)forward(&transformer, input_token, position);
        if (position <= DIAG_LAST_POSITION) {
            printf("cache_forward loop_k=%d position=%d token_id=%d target_token_id=%d\n",
                k, position, input_token, target_token);
            if (position >= 13 && position <= 15)
                print_hidden_samples("cache_writer", position, input_token,
                    target_token, transformer.state.x);
            if (position == 14)
                printf("cache_kv_checksum_before_position_15=%016llx\n",
                    (unsigned long long)kv_checksum(&transformer, 14));
        }
    }

    printf("\npath=cache_equivalence_live_forward_trace\n");
    clear_run_state(&transformer);
    printf("initial_state.x0_bits=%08x value=%.9g\n",
        (unsigned int)float_bits(transformer.state.x[0]), transformer.state.x[0]);
    printf("initial_kv_checksum_positions_0_14=%016llx\n",
        (unsigned long long)kv_checksum(&transformer, 14));
    for (k = 1; k <= DIAG_LAST_POSITION; k++) {
        int position = k - 1;
        if (k < record->first_target_index) {
            if (position >= 13 && position <= 14) {
                printf("live_loop loop_k=%d position=%d token_id=%d target_token_id=%d forward_called=no reason=prompt_mask\n",
                    k, position, record->training_seq_tokens[position],
                    record->training_seq_tokens[k]);
            }
            continue;
        }
        printf("live_loop loop_k=%d position=%d token_id=%d target_token_id=%d forward_called=yes\n",
            k, position, record->training_seq_tokens[position],
            record->training_seq_tokens[k]);
        (void)forward(&transformer, record->training_seq_tokens[position], position);
    }
    printf("live_kv_checksum_before_position_15=%016llx\n",
        (unsigned long long)kv_checksum(&transformer, 14));
    printf("live_final_call input_position=%d token_id=%d position_argument=%d target_token_id=%d\n",
        DIAG_LAST_POSITION,
        record->training_seq_tokens[DIAG_LAST_POSITION],
        DIAG_LAST_POSITION,
        record->training_seq_tokens[DIAG_LAST_POSITION + 1]);
    (void)forward(&transformer,
        record->training_seq_tokens[DIAG_LAST_POSITION], DIAG_LAST_POSITION);
    print_hidden_samples("cache_equivalence_live", DIAG_LAST_POSITION,
        record->training_seq_tokens[DIAG_LAST_POSITION],
        record->training_seq_tokens[DIAG_LAST_POSITION + 1], transformer.state.x);

    stage4e_dataset_free(&dataset);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return EXIT_SUCCESS;

fail:
    stage4e_dataset_free(&dataset);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return EXIT_FAILURE;
}
