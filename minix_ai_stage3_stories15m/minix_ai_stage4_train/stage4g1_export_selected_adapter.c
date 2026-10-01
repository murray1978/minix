#define STAGE4E_D6_CORE_ONLY
#include "stage4e_d6_a4_checkpoint.c"
#undef STAGE4E_D6_CORE_ONLY

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define G1_CHECKPOINT "stage4e-f2-a4-selected-step20-audit-checkpoint.bin"
#define G1_F2_REPORT "stage4e-f2-a4-selected-step20-audit-test-report.txt"
#define G1_ADAPTER "stage4g1-selected-step20.adapter.bin"
#define G1_REPORT "stage4g1-selected-step20-export-report.txt"
#define G1_BASE_CHECKPOINT "/usr/src/minix_ai_stage3_stories15m/stories15M.bin"
#define G1_EXPECTED_CHECKPOINT_SHA "87148919dfc4d5da22e7a8494216dd47f9b30064bae9a347bee6e37ddd50b1f0"
#define G1_EXPECTED_REPORT_SHA "b9d8c727538d94554cfe9851d14f92cbb896a8f9faf08c929d5ad678320ee707"
#define G1_EXPECTED_BASE_SHA "cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a"
#define G1_EXPECTED_DATASET_SHA "71744cea4de049e2e192cb2a4494c8a4a68e769b3a7f3b8ed4880433e4583b84"
#define G1_EXPECTED_CACHE_SHA "d6288ad3d128c7182c20088aa57a1bca8bcaa1e13314fb0cdbd60eba14278def"
#define G1_EXPECTED_TOKENIZER_SHA "50a52ef822ee9e83de5ce9d0be0a025a773d019437f58b5ff9dcafb063ece361"
#define G1_DIM 288
#define G1_VOCAB 32000
#define G1_RANK 8

static int g1_file_exists(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (f == NULL)
        return 0;
    fclose(f);
    return 1;
}

static int g1_weights_finite(const stage4_adapter_t *ad)
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

static int g1_load_checkpoint(const char *path, const char *dataset_sha,
    const char *cache_sha, const char *base_sha, const char *tokenizer_sha,
    stage4_adapter_t *ad, uint32_t *version_out, uint64_t *step_out)
{
    unsigned char magic[8], dataset[32], cache[32], base[32], tokenizer[32];
    unsigned char expected_dataset[32], expected_cache[32];
    unsigned char expected_base[32], expected_tokenizer[32];
    uint32_t version, header, dim, vocab, rank, a_count, b_count, payload;
    float scale, learning_rate, beta1, beta2, epsilon, weight_decay;
    uint64_t step, clip_bits = 0U, file_bytes = 0U;
    double clip;
    FILE *f;
    int ok;

    f = fopen(path, "rb");
    if (f == NULL)
        return 0;
    ok = fread(magic, 1, sizeof(magic), f) == sizeof(magic) &&
        memcmp(magic, D6_MAGIC, sizeof(magic)) == 0 &&
        d6_read_u32(f, &version) && d6_read_u32(f, &header) &&
        d6_read_u32(f, &dim) && d6_read_u32(f, &vocab) &&
        d6_read_u32(f, &rank) && d6_read_u32(f, &a_count) &&
        d6_read_u32(f, &b_count) && d6_read_f32(f, &scale) &&
        d6_read_u64(f, &step) && d6_read_f32(f, &learning_rate) &&
        d6_read_f32(f, &beta1) && d6_read_f32(f, &beta2) &&
        d6_read_f32(f, &epsilon) && d6_read_f32(f, &weight_decay) &&
        d6_read_u64(f, &clip_bits) && d6_read_u32(f, &payload) &&
        fread(dataset, 1, sizeof(dataset), f) == sizeof(dataset) &&
        fread(cache, 1, sizeof(cache), f) == sizeof(cache) &&
        fread(base, 1, sizeof(base), f) == sizeof(base) &&
        fread(tokenizer, 1, sizeof(tokenizer), f) == sizeof(tokenizer) &&
        d6_hex_raw(dataset_sha, expected_dataset) &&
        d6_hex_raw(cache_sha, expected_cache) &&
        d6_hex_raw(base_sha, expected_base) &&
        d6_hex_raw(tokenizer_sha, expected_tokenizer);
    memcpy(&clip, &clip_bits, sizeof(clip));
    if (!ok || version != D6_FORMAT_VERSION || header != D6_HEADER_BYTES ||
        dim != G1_DIM || vocab != G1_VOCAB || rank != G1_RANK ||
        a_count != ad->a_count || b_count != ad->b_count ||
        payload != D6_PAYLOAD_FLOATS || step != 20U || scale != D4_SCALE ||
        learning_rate != (float)D4_LR || beta1 != (float)D4_BETA1 ||
        beta2 != (float)D4_BETA2 || epsilon != (float)D4_EPSILON ||
        weight_decay != (float)D4_WEIGHT_DECAY || clip != D4_CLIP ||
        memcmp(dataset, expected_dataset, sizeof(dataset)) != 0 ||
        memcmp(cache, expected_cache, sizeof(cache)) != 0 ||
        memcmp(base, expected_base, sizeof(base)) != 0 ||
        memcmp(tokenizer, expected_tokenizer, sizeof(tokenizer)) != 0) {
        fclose(f);
        return 0;
    }

    ok = d6_read_array(f, ad->a, ad->a_count) &&
        d6_read_array(f, ad->b, ad->b_count) &&
        d6_read_array(f, ad->m1_a, ad->a_count) &&
        d6_read_array(f, ad->m2_a, ad->a_count) &&
        d6_read_array(f, ad->m1_b, ad->b_count) &&
        d6_read_array(f, ad->m2_b, ad->b_count) &&
        fgetc(f) == EOF && !ferror(f);
    if (fclose(f) != 0)
        ok = 0;
    if (!ok || !stage4_file_size_bytes(path, &file_bytes) ||
        file_bytes != D6_TOTAL_BYTES)
        return 0;

    *version_out = version;
    *step_out = step;
    return 1;
}

static int g1_write_report(const char *checkpoint_sha, const char *report_sha,
    uint32_t checkpoint_version, uint64_t optimizer_step,
    uint64_t adapter_file_bytes, uint32_t mismatch_a, uint32_t mismatch_b,
    uint32_t mismatch_total, int exported_finite, int reloaded_finite,
    int checkpoint_verified, int adapter_match)
{
    uint64_t a_bytes = 2304U * sizeof(float);
    uint64_t b_bytes = 256000U * sizeof(float);
    uint64_t payload_bytes = a_bytes + b_bytes;
    FILE *f = fopen(G1_REPORT, "wb");
    int ok = 1;

    if (f == NULL)
        return 0;
    if (fprintf(f,
        "stage=4G1\nmode=export_selected_step20_adapter\n"
        "input_checkpoint=%s\ninput_checkpoint_sha256=%s\n"
        "f2_report=%s\nf2_report_sha256=%s\n"
        "input_checkpoint_format_version=%u\ninput_optimizer_step=%llu\n"
        "adapter_output=%s\nadapter_format_version=2\nadapter_layout_version=1\n"
        "adapter_rank=%u\nadapter_input_dim=%u\nadapter_output_vocab=%u\nadapter_scale=%.9g\n"
        "adapter_A_elements=2304\nadapter_B_elements=256000\nadapter_total_elements=258304\n"
        "adapter_A_bytes=%llu\nadapter_B_bytes=%llu\nadapter_payload_bytes=%llu\n"
        "exported_file_size=%llu\n"
        "A_byte_mismatches=%u\nB_byte_mismatches=%u\ntotal_adapter_byte_mismatches=%u\n"
        "all_exported_values_finite=%s\nall_reloaded_values_finite=%s\n"
        "train_rows_evaluated=0\nvalidation_rows_evaluated=0\ntest_rows_evaluated=0\n"
        "gradient_rows_consumed=0\noptimizer_updates=0\n"
        "checkpoint_step20_verified=%s\n"
        "deployment_adapter_roundtrip_bitwise_match=%s\n"
        "gate_stage4g1=%s\n",
        G1_CHECKPOINT, checkpoint_sha, G1_F2_REPORT, report_sha,
        checkpoint_version, (unsigned long long)optimizer_step, G1_ADAPTER,
        G1_RANK, G1_DIM, G1_VOCAB, (double)D4_SCALE,
        (unsigned long long)a_bytes, (unsigned long long)b_bytes,
        (unsigned long long)payload_bytes,
        (unsigned long long)adapter_file_bytes,
        mismatch_a, mismatch_b, mismatch_total,
        exported_finite ? "yes" : "no", reloaded_finite ? "yes" : "no",
        checkpoint_verified ? "yes" : "no", adapter_match ? "yes" : "no",
        checkpoint_verified && adapter_match && exported_finite && reloaded_finite &&
            mismatch_total == 0U ? "PASS" : "FAIL") < 0)
        ok = 0;
    if (fclose(f) != 0)
        ok = 0;
    return ok;
}

int main(void)
{
    stage4_adapter_t selected, reloaded;
    stage4_base_identity_t base_identity;
    char checkpoint_sha[65], report_sha[65], base_sha[65];
    uint64_t base_bytes = 0U, adapter_file_bytes = 0U, optimizer_step = 0U;
    uint32_t checkpoint_version = 0U, mismatch_a = 0U, mismatch_b = 0U;
    uint32_t mismatch_total = 0U;
    int selected_ready = 0, reloaded_ready = 0;
    int output_touched = 0, report_touched = 0;
    int exported_finite = 0, reloaded_finite = 0;
    int checkpoint_verified = 0, adapter_match = 0, success = 0;

    memset(&selected, 0, sizeof(selected));
    memset(&reloaded, 0, sizeof(reloaded));
    memset(&base_identity, 0, sizeof(base_identity));
    if (g1_file_exists(G1_ADAPTER) || g1_file_exists(G1_REPORT)) {
        fprintf(stderr, "error: refusing to overwrite existing G1 output\n");
        return 1;
    }
    if (!stage4_hash_file_sha256_hex(G1_CHECKPOINT, checkpoint_sha) ||
        strcmp(checkpoint_sha, G1_EXPECTED_CHECKPOINT_SHA) != 0 ||
        !stage4_hash_file_sha256_hex(G1_F2_REPORT, report_sha) ||
        strcmp(report_sha, G1_EXPECTED_REPORT_SHA) != 0) {
        fprintf(stderr, "error: accepted F2 checkpoint/report identity mismatch\n");
        goto done;
    }
    if (!stage4_hash_file_sha256_hex(G1_BASE_CHECKPOINT, base_sha) ||
        strcmp(base_sha, G1_EXPECTED_BASE_SHA) != 0 ||
        !stage4_file_size_bytes(G1_BASE_CHECKPOINT, &base_bytes)) {
        fprintf(stderr, "error: base checkpoint identity/size mismatch\n");
        goto done;
    }
    strcpy(base_identity.base_sha256, base_sha);
    base_identity.base_checkpoint_bytes = base_bytes;

    if (!stage4_adapter_init(&selected, G1_DIM, G1_VOCAB, G1_RANK, D4_SCALE)) {
        fprintf(stderr, "error: selected checkpoint adapter allocation failed\n");
        goto done;
    }
    selected_ready = 1;
    if (!g1_load_checkpoint(G1_CHECKPOINT, G1_EXPECTED_DATASET_SHA,
        G1_EXPECTED_CACHE_SHA, G1_EXPECTED_BASE_SHA,
        G1_EXPECTED_TOKENIZER_SHA, &selected, &checkpoint_version,
        &optimizer_step)) {
        fprintf(stderr, "error: accepted F2 checkpoint parse/validation failed\n");
        goto done;
    }
    checkpoint_verified = checkpoint_version == D6_FORMAT_VERSION &&
        optimizer_step == 20U;
    exported_finite = g1_weights_finite(&selected);
    if (!checkpoint_verified || !exported_finite) {
        fprintf(stderr, "error: checkpoint step/version or adapter values invalid\n");
        goto done;
    }

    output_touched = 1;
    if (!stage4_adapter_save_file(G1_ADAPTER, &selected, &base_identity) ||
        !stage4_file_size_bytes(G1_ADAPTER, &adapter_file_bytes)) {
        fprintf(stderr, "error: deployment adapter export failed\n");
        goto done;
    }
    if (!stage4_adapter_init(&reloaded, G1_DIM, G1_VOCAB, G1_RANK, D4_SCALE)) {
        fprintf(stderr, "error: fresh adapter allocation failed\n");
        goto done;
    }
    reloaded_ready = 1;
    if (!stage4_adapter_load_file(G1_ADAPTER, &reloaded, &base_identity, 0)) {
        fprintf(stderr, "error: deployment adapter reload/metadata validation failed\n");
        goto done;
    }
    reloaded_finite = g1_weights_finite(&reloaded);
    mismatch_a = d6_byte_mismatches(selected.a, reloaded.a,
        selected.a_count * sizeof(float));
    mismatch_b = d6_byte_mismatches(selected.b, reloaded.b,
        selected.b_count * sizeof(float));
    mismatch_total = mismatch_a + mismatch_b;
    adapter_match = reloaded_finite && mismatch_total == 0U;
    if (!adapter_match) {
        fprintf(stderr, "error: exported adapter is not a finite bitwise round trip\n");
        goto done;
    }

    report_touched = 1;
    if (!g1_write_report(checkpoint_sha, report_sha, checkpoint_version,
        optimizer_step, adapter_file_bytes, mismatch_a, mismatch_b,
        mismatch_total, exported_finite, reloaded_finite,
        checkpoint_verified, adapter_match)) {
        fprintf(stderr, "error: G1 report write failed\n");
        goto done;
    }
    success = 1;
    printf("adapter_output=%s\nreport=%s\ncheckpoint_step20_verified=yes\n"
        "deployment_adapter_roundtrip_bitwise_match=yes\nStage4G1=PASS\n",
        G1_ADAPTER, G1_REPORT);

done:
    if (!success) {
        if (output_touched)
            remove(G1_ADAPTER);
        if (report_touched)
            remove(G1_REPORT);
    }
    if (selected_ready)
        stage4_adapter_free(&selected);
    if (reloaded_ready)
        stage4_adapter_free(&reloaded);
    return success ? 0 : 1;
}
