#define STAGE4E_D2_CORE_ONLY
#include "stage4e_d2_a4_one_step.c"
#undef STAGE4E_D2_CORE_ONLY

#define D3_EXPECTED_DATASET_SHA256 \
    "71744cea4de049e2e192cb2a4494c8a4a68e769b3a7f3b8ed4880433e4583b84"
#define D3_EXPECTED_CACHE_SHA256 \
    "d6288ad3d128c7182c20088aa57a1bca8bcaa1e13314fb0cdbd60eba14278def"
#define D3_EXPECTED_CHECKPOINT_SHA256 \
    "cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a"
#define D3_EXPECTED_TOKENIZER_SHA256 \
    "50a52ef822ee9e83de5ce9d0be0a025a773d019437f58b5ff9dcafb063ece361"
#define D3_EXPECTED_D1_REPORT_SHA256 \
    "45ebaa750f17d02d80feed332b16c9c29c3782a544aed7544da72a62161c76a6"
#define D3_EXPECTED_D2_REPORT_SHA256 \
    "c27aad7474df093f5a3ff781bb3619b71b3e284edda46290f5814c350e99d97e"
#define D3_BUILD_PROVENANCE "stored-double-compare-fix-2026-10-01-c"

static const char *d3_step1_total_loss[3] = {
    "1158.085676086679", "300.451596646300", "223.897562610626"
};
static const char *d3_step1_average_loss[3] = {
    "10.433204289069", "9.104593837767", "10.661788695744"
};
static const uint32_t d3_step1_top1[3] = { 0U, 2U, 0U };
static const char *d3_step1_grad_b_max = "0.145234793425";
static const char *d3_step1_norm = "0.381956843381";
static const char *d3_step1_b_delta = "0.000999999931082";
static const uint32_t d3_step1_moment_b_nonzero = 256000U;
static const char *d3_validation_baseline = "9.106262242666";
static const char *d3_validation_acceptance_max = "8.650949130533";

typedef struct {
    uint32_t gradient_train_rows;
    uint32_t gradient_validation_rows;
    uint32_t gradient_test_rows;
    d2_gradient_stats_t gradient;
    double norm_before_clip;
    double norm_after_clip;
    int clip_applied;
    int clip_consistency;
    int adapter_a_changed;
    int adapter_b_changed;
    double adapter_a_max_abs_delta;
    double adapter_b_max_abs_delta;
    uint32_t m1_a_nonzero;
    uint32_t m2_a_nonzero;
    uint32_t m1_b_nonzero;
    uint32_t m2_b_nonzero;
    d2_metrics_t evaluation;
} d3_step_t;

static int d3_format_matches_g(double value, const char *expected)
{
    char formatted[64];
    if (snprintf(formatted, sizeof(formatted), "%.12g", value) < 0)
        return 0;
    return strcmp(formatted, expected) == 0;
}

static int d3_accumulate_and_clip(const d2_cache_row_t *rows,
    stage4_adapter_t *adapter,
    Transformer *transformer,
    float *base_logits,
    float *final_logits,
    int step_index,
    d3_step_t *step)
{
    uint32_t row_index;
    uint64_t norm_before_bits;
    uint64_t norm_after_bits;
    int stored_norm_bits_equal;

    stage4_adapter_zero_grad(adapter);
    step->gradient_train_rows = 0U;
    step->gradient_validation_rows = 0U;
    step->gradient_test_rows = 0U;
    for (row_index = 0U; row_index < D2_CACHE_ROWS; row_index++) {
        const d2_cache_row_t *row = &rows[row_index];
        double gradient_loss;

        if (row->split_id == D2_SPLIT_VALIDATION || row->split_id == D2_SPLIT_TEST)
            continue;
        if (row->split_id != D2_SPLIT_TRAIN) {
            fprintf(stderr, "error: non-training split reached gradient step %d\n",
                step_index);
            return 0;
        }
        matmul(base_logits, row->hidden, transformer->weights.wcls,
            (int)D2_CACHE_DIM, (int)D2_CACHE_VOCAB);
        stage4_adapter_forward_logits(adapter, row->hidden,
            base_logits, final_logits);
        if (!stage4_softmax_loss_and_dlogits(final_logits, adapter->vocab,
            (int)row->target_token_id, &gradient_loss, adapter->dlogits)) {
            fprintf(stderr, "error: gradient loss failed at step %d row %u\n",
                step_index, row_index);
            return 0;
        }
        stage4_adapter_backward(adapter, row->hidden, adapter->dlogits);
        step->gradient_train_rows++;
    }
    if (step->gradient_train_rows != 111U ||
        step->gradient_validation_rows != 0U || step->gradient_test_rows != 0U) {
        fprintf(stderr, "error: step %d gradient row discipline failed\n", step_index);
        return 0;
    }

    stage4_adapter_scale_gradients(adapter,
        1.0f / (float)step->gradient_train_rows);
    d2_collect_gradient_stats(adapter, &step->gradient);
    step->norm_before_clip = stage4_adapter_gradient_norm(adapter);
    if (!isfinite(step->norm_before_clip) || !(step->norm_before_clip > 0.0) ||
        step->gradient.b_nonzero == 0U || !(step->gradient.b_max_abs > 0.0) ||
        (step_index == 1 && (step->gradient.a_nonzero != 0U ||
            step->gradient.a_max_abs != 0.0)) ||
        (step_index == 2 && (step->gradient.a_nonzero == 0U ||
            !(step->gradient.a_max_abs > 0.0)))) {
        fprintf(stderr, "error: step %d low-rank gradient invariant failed\n",
            step_index);
        return 0;
    }

    step->clip_applied = step->norm_before_clip > D2_GRADIENT_CLIP;
    stage4_adapter_clip_gradients(adapter, D2_GRADIENT_CLIP);
    step->norm_after_clip = stage4_adapter_gradient_norm(adapter);
    if (step->clip_applied)
        step->clip_consistency = isfinite(step->norm_after_clip) &&
            fabs(step->norm_after_clip - D2_GRADIENT_CLIP) <= 0.000001;
    else
        step->clip_consistency = memcmp(&step->norm_after_clip,
            &step->norm_before_clip, sizeof(step->norm_after_clip)) == 0;
    memcpy(&norm_before_bits, &step->norm_before_clip, sizeof(norm_before_bits));
    memcpy(&norm_after_bits, &step->norm_after_clip, sizeof(norm_after_bits));
    stored_norm_bits_equal = norm_before_bits == norm_after_bits;
    printf("diagnostic_step_index=%d\n"
        "diagnostic_gradient_train_rows=%u\n"
        "diagnostic_gradient_validation_rows=%u\n"
        "diagnostic_gradient_test_rows=%u\n"
        "diagnostic_grad_A_nonzero_elements=%u\n"
        "diagnostic_grad_A_max_abs=%.17g\n"
        "diagnostic_grad_B_nonzero_elements=%u\n"
        "diagnostic_grad_B_max_abs=%.17g\n"
        "diagnostic_norm_before_clip=%.17g\n"
        "diagnostic_clip_threshold=%.17g\n"
        "diagnostic_clip_applied=%s\n"
        "diagnostic_norm_after_clip=%.17g\n"
        "diagnostic_clip_consistency=%s\n"
        "diagnostic_clip_failure=%s\n"
        "diagnostic_norm_before_clip_bits=0x%016llx\n"
        "diagnostic_norm_after_clip_bits=0x%016llx\n"
        "diagnostic_stored_norm_bits_equal=%s\n",
        step_index, step->gradient_train_rows,
        step->gradient_validation_rows, step->gradient_test_rows,
        step->gradient.a_nonzero, step->gradient.a_max_abs,
        step->gradient.b_nonzero, step->gradient.b_max_abs,
        step->norm_before_clip, D2_GRADIENT_CLIP,
        step->clip_applied ? "yes" : "no", step->norm_after_clip,
        step->clip_consistency ? "yes" : "no",
        step->clip_consistency ? "no" : "yes",
        (unsigned long long)norm_before_bits,
        (unsigned long long)norm_after_bits,
        stored_norm_bits_equal ? "yes" : "no");
    return step->clip_consistency;
}

static int d3_step1_matches_d2(const d3_step_t *step,
    const stage4_adapter_t *adapter,
    const d2_metrics_t *metrics,
    int step1_a_changed,
    int step1_b_changed,
    double step1_a_delta,
    double step1_b_delta,
    uint32_t m1_a, uint32_t m2_a, uint32_t m1_b, uint32_t m2_b)
{
    uint32_t split;
    if (step->gradient_train_rows != 111U ||
        step->gradient_validation_rows != 0U || step->gradient_test_rows != 0U ||
        step->gradient.a_nonzero != 0U || step->gradient.a_max_abs != 0.0 ||
        step->gradient.b_nonzero != 256000U ||
        !d3_format_matches_g(step->gradient.b_max_abs, d3_step1_grad_b_max) ||
        !d3_format_matches_g(step->norm_before_clip, d3_step1_norm) ||
        !d3_format_matches_g(step->norm_after_clip, d3_step1_norm) ||
        step->clip_applied || !step->clip_consistency ||
        step1_a_changed || !step1_b_changed || step1_a_delta != 0.0 ||
        !d3_format_matches_g(step1_b_delta, d3_step1_b_delta) ||
        adapter->rank != D2_RANK || adapter->scale != D2_SCALE ||
        adapter->a_count + adapter->b_count != 258304U ||
        m1_a != 0U || m2_a != 0U || m1_b != d3_step1_moment_b_nonzero ||
        m2_b != d3_step1_moment_b_nonzero || metrics->evaluated_rows != D2_CACHE_ROWS)
        return 0;
    for (split = 0U; split < 3U; split++) {
        if (metrics->splits[split].target_count != d2_expected_split_rows[split] ||
            metrics->splits[split].top1_correct != d3_step1_top1[split] ||
            !d2_format_matches(metrics->splits[split].loss_sum,
                d3_step1_total_loss[split]) ||
            !d2_format_matches(metrics->splits[split].loss_sum /
                (double)metrics->splits[split].target_count,
                d3_step1_average_loss[split]))
            return 0;
    }
    return 1;
}

static int d3_write_report(const char *path,
    const char *dataset_sha, const char *cache_sha,
    const char *checkpoint_sha, const char *tokenizer_sha,
    const char *d1_sha, const char *d2_sha,
    const stage4_adapter_t *adapter,
    const d2_metrics_t *initial_metrics,
    const d3_step_t *step1, const d3_step_t *step2,
    int step1_a_changed, int step1_b_changed,
    double step1_a_delta, double step1_b_delta,
    int final_a_changed_initial, int final_b_changed_initial,
    double final_a_max_delta, double final_b_max_delta,
    uint64_t optimizer_steps_before, uint64_t optimizer_steps_after,
    uint32_t final_m1_a, uint32_t final_m2_a,
    uint32_t final_m1_b, uint32_t final_m2_b,
    double step2_vs_step1_delta, double step2_vs_initial_delta,
    int validation_acceptance_reached,
    uint32_t final_zero_correction_rows,
    uint32_t final_nonzero_correction_rows,
    int initial_d1_match, int step1_d2_match, int passed)
{
    static const char *split_names[3] = { "train", "validation", "test" };
    FILE *report = fopen(path, "wb");
    uint32_t split;
    int ok;

    if (report == NULL)
        return 0;
    ok = fprintf(report,
        "dataset_sha256=%s\n"
        "cache_sha256=%s\n"
        "base_checkpoint_sha256=%s\n"
        "tokenizer_sha256=%s\n"
        "d1_report_sha256=%s\n"
        "d2_report_sha256=%s\n"
        "rank=%d\n"
        "scale=%.1f\n"
        "adapter_parameter_count=%lu\n"
        "learning_rate=%.12g\n"
        "beta1=%.12g\n"
        "beta2=%.12g\n"
        "epsilon=%.12g\n"
        "weight_decay=%.12g\n"
        "gradient_clip_threshold=%.1f\n"
        "step1_gradient_train_rows=%u\n"
        "step1_gradient_validation_rows=%u\n"
        "step1_gradient_test_rows=%u\n"
        "step1_grad_A_nonzero_elements=%u\n"
        "step1_grad_A_max_abs=%.12g\n"
        "step1_grad_B_nonzero_elements=%u\n"
        "step1_grad_B_max_abs=%.12g\n"
        "step1_gradient_norm_before_clip=%.12g\n"
        "step1_gradient_norm_after_clip=%.12g\n"
        "step1_gradient_clip_applied=%s\n"
        "step1_adapter_A_changed=%s\n"
        "step1_adapter_B_changed=%s\n"
        "step1_adapter_A_max_abs_delta=%.12g\n"
        "step1_adapter_B_max_abs_delta=%.12g\n"
        "step1_optimizer_steps_after=1\n"
        "step1_m1_A_nonzero_elements=%u\n"
        "step1_m2_A_nonzero_elements=%u\n"
        "step1_m1_B_nonzero_elements=%u\n"
        "step1_m2_B_nonzero_elements=%u\n"
        "step1_d2_metrics_match=%s\n"
        "step2_gradient_train_rows=%u\n"
        "step2_gradient_validation_rows=%u\n"
        "step2_gradient_test_rows=%u\n"
        "step2_grad_A_nonzero_elements=%u\n"
        "step2_grad_A_max_abs=%.12g\n"
        "step2_grad_B_nonzero_elements=%u\n"
        "step2_grad_B_max_abs=%.12g\n"
        "step2_gradient_norm_before_clip=%.12g\n"
        "step2_gradient_norm_after_clip=%.12g\n"
        "step2_gradient_clip_applied=%s\n"
        "step2_adapter_A_changed=%s\n"
        "step2_adapter_B_changed=%s\n"
        "step2_adapter_A_max_abs_delta=%.12g\n"
        "step2_adapter_B_max_abs_delta=%.12g\n"
        "final_adapter_A_changed_from_initial=%s\n"
        "final_adapter_B_changed_from_initial=%s\n"
        "final_adapter_A_max_abs_delta=%.12g\n"
        "final_adapter_B_max_abs_delta=%.12g\n"
        "optimizer_steps_before=%llu\n"
        "optimizer_steps_after=%llu\n"
        "final_m1_A_nonzero_elements=%u\n"
        "final_m2_A_nonzero_elements=%u\n"
        "final_m1_B_nonzero_elements=%u\n"
        "final_m2_B_nonzero_elements=%u\n"
        "initial_d1_metrics_match=%s\n"
        "step1_d2_metrics_match=%s\n"
        "step2_vs_step1_train_loss_delta=%.12f\n"
        "step2_vs_initial_train_loss_delta=%.12f\n"
        "validation_baseline_loss=%s\n"
        "validation_acceptance_loss_max=%s\n"
        "step2_validation_acceptance_reached=%s\n"
        "final_zero_correction_rows=%u\n"
        "final_nonzero_correction_rows=%u\n"
        "d3_two_step_pass=%s\n",
        dataset_sha, cache_sha, checkpoint_sha, tokenizer_sha, d1_sha, d2_sha,
        adapter->rank, (double)adapter->scale,
        (unsigned long)(adapter->a_count + adapter->b_count),
        D2_LEARNING_RATE, D2_BETA1, D2_BETA2, D2_EPSILON, D2_WEIGHT_DECAY,
        D2_GRADIENT_CLIP,
        step1->gradient_train_rows, step1->gradient_validation_rows,
        step1->gradient_test_rows, step1->gradient.a_nonzero,
        step1->gradient.a_max_abs, step1->gradient.b_nonzero,
        step1->gradient.b_max_abs, step1->norm_before_clip,
        step1->norm_after_clip, step1->clip_applied ? "yes" : "no",
        step1_a_changed ? "yes" : "no", step1_b_changed ? "yes" : "no",
        step1_a_delta, step1_b_delta,
        step1->m1_a_nonzero, step1->m2_a_nonzero,
        step1->m1_b_nonzero, step1->m2_b_nonzero,
        step1_d2_match ? "yes" : "no",
        step2->gradient_train_rows, step2->gradient_validation_rows,
        step2->gradient_test_rows, step2->gradient.a_nonzero,
        step2->gradient.a_max_abs, step2->gradient.b_nonzero,
        step2->gradient.b_max_abs, step2->norm_before_clip,
        step2->norm_after_clip, step2->clip_applied ? "yes" : "no",
        step2->adapter_a_changed ? "yes" : "no",
        step2->adapter_b_changed ? "yes" : "no",
        step2->adapter_a_max_abs_delta, step2->adapter_b_max_abs_delta,
        final_a_changed_initial ? "yes" : "no",
        final_b_changed_initial ? "yes" : "no", final_a_max_delta,
        final_b_max_delta, (unsigned long long)optimizer_steps_before,
        (unsigned long long)optimizer_steps_after,
        final_m1_a, final_m2_a, final_m1_b, final_m2_b,
        initial_d1_match ? "yes" : "no", step1_d2_match ? "yes" : "no",
        step2_vs_step1_delta, step2_vs_initial_delta,
        d3_validation_baseline, d3_validation_acceptance_max,
        validation_acceptance_reached ? "yes" : "no",
        final_zero_correction_rows, final_nonzero_correction_rows,
        passed ? "yes" : "no") >= 0;

    for (split = 0U; ok && split < 3U; split++) {
        double initial_average = initial_metrics->splits[split].loss_sum /
            (double)initial_metrics->splits[split].target_count;
        double step1_average = step1->evaluation.splits[split].loss_sum /
            (double)step1->evaluation.splits[split].target_count;
        double step2_average = step2->evaluation.splits[split].loss_sum /
            (double)step2->evaluation.splits[split].target_count;
        ok = fprintf(report,
            "initial_%s_target_count=%u\n"
            "initial_%s_total_loss=%.12f\n"
            "initial_%s_average_loss=%.12f\n"
            "initial_%s_top1_correct=%llu\n"
            "step1_%s_target_count=%u\n"
            "step1_%s_total_loss=%.12f\n"
            "step1_%s_average_loss=%.12f\n"
            "step1_%s_top1_correct=%llu\n"
            "step2_%s_target_count=%u\n"
            "step2_%s_total_loss=%.12f\n"
            "step2_%s_average_loss=%.12f\n"
            "step2_%s_top1_correct=%llu\n",
            split_names[split], initial_metrics->splits[split].target_count,
            split_names[split], initial_metrics->splits[split].loss_sum,
            split_names[split], initial_average,
            split_names[split],
            (unsigned long long)initial_metrics->splits[split].top1_correct,
            split_names[split], step1->evaluation.splits[split].target_count,
            split_names[split], step1->evaluation.splits[split].loss_sum,
            split_names[split], step1_average,
            split_names[split],
            (unsigned long long)step1->evaluation.splits[split].top1_correct,
            split_names[split], step2->evaluation.splits[split].target_count,
            split_names[split], step2->evaluation.splits[split].loss_sum,
            split_names[split], step2_average,
            split_names[split],
            (unsigned long long)step2->evaluation.splits[split].top1_correct) >= 0;
    }
    if (fclose(report) != 0)
        ok = 0;
    return ok;
}

static void d3_usage(const char *program)
{
    fprintf(stderr,
        "Usage: %s <checkpoint.bin> -z <tokenizer.bin> -d <dataset> "
        "-c <cache.bin> -1 <d1-report> -2 <d2-report> -r <report.txt>\n",
        program);
}

int main(int argc, char **argv)
{
    const char *checkpoint_path = NULL;
    const char *tokenizer_path = NULL;
    const char *dataset_path = NULL;
    const char *cache_path = NULL;
    const char *d1_report_path = NULL;
    const char *d2_report_path = NULL;
    const char *report_path = "stage4e-d3-a4-two-step-report.txt";
    char dataset_sha[65] = "";
    char cache_sha[65] = "";
    char checkpoint_sha[65] = "";
    char tokenizer_sha[65] = "";
    char d1_sha[65] = "";
    char d2_sha[65] = "";
    unsigned char dataset_raw[32];
    unsigned char checkpoint_raw[32];
    unsigned char tokenizer_raw[32];
    unsigned char expected_dataset_raw[32];
    unsigned char expected_checkpoint_raw[32];
    unsigned char expected_tokenizer_raw[32];
    d2_cache_header_t header;
    d2_cache_row_t *rows = NULL;
    d2_metrics_t initial_metrics;
    d3_step_t step1;
    d3_step_t step2;
    stage4_adapter_t adapter;
    Transformer transformer;
    float *base_logits = NULL;
    float *final_logits = NULL;
    float *initial_a = NULL;
    float *initial_b = NULL;
    float *step1_a = NULL;
    float *step1_b = NULL;
    FILE *cache_file = NULL;
    uint32_t split_rows[D2_SPLIT_COUNT] = { 0U, 0U, 0U, 0U };
    uint32_t row_index;
    uint32_t final_m1_a = 0U;
    uint32_t final_m2_a = 0U;
    uint32_t final_m1_b = 0U;
    uint32_t final_m2_b = 0U;
    uint32_t final_zero_correction_rows = 0U;
    uint32_t final_nonzero_correction_rows = 0U;
    uint64_t cache_size = 0U;
    uint64_t optimizer_steps_before = 0U;
    uint64_t optimizer_steps_after = 0U;
    double step1_a_delta = 0.0;
    double step1_b_delta = 0.0;
    double update_norm = 0.0;
    double final_a_max_delta = 0.0;
    double final_b_max_delta = 0.0;
    double step2_vs_step1_delta = 0.0;
    double step2_vs_initial_delta = 0.0;
    int initial_a_changed_final = 0;
    int initial_b_changed_final = 0;
    int initial_d1_match = 0;
    int step1_d2_match = 0;
    int validation_acceptance_reached = 0;
    int passed = 0;
    int failed = 0;
    int transformer_ready = 0;
    int adapter_ready = 0;
    int argi;

    printf("d3_build_provenance=%s\n", D3_BUILD_PROVENANCE);
    memset(&header, 0, sizeof(header));
    memset(&initial_metrics, 0, sizeof(initial_metrics));
    memset(&step1, 0, sizeof(step1));
    memset(&step2, 0, sizeof(step2));
    memset(&adapter, 0, sizeof(adapter));
    memset(&transformer, 0, sizeof(transformer));
    memset(dataset_raw, 0, sizeof(dataset_raw));
    memset(checkpoint_raw, 0, sizeof(checkpoint_raw));
    memset(tokenizer_raw, 0, sizeof(tokenizer_raw));
    memset(expected_dataset_raw, 0, sizeof(expected_dataset_raw));
    memset(expected_checkpoint_raw, 0, sizeof(expected_checkpoint_raw));
    memset(expected_tokenizer_raw, 0, sizeof(expected_tokenizer_raw));

    if (argc < 2) {
        d3_usage(argv[0]);
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
        else if (strcmp(argv[argi], "-2") == 0 && argi + 1 < argc)
            d2_report_path = argv[++argi];
        else if (strcmp(argv[argi], "-r") == 0 && argi + 1 < argc)
            report_path = argv[++argi];
        else {
            d3_usage(argv[0]);
            return EXIT_FAILURE;
        }
    }
    if (tokenizer_path == NULL || dataset_path == NULL || cache_path == NULL ||
        d1_report_path == NULL || d2_report_path == NULL ||
        strcmp(report_path, cache_path) == 0 || strcmp(report_path, dataset_path) == 0 ||
        strcmp(report_path, d1_report_path) == 0 || strcmp(report_path, d2_report_path) == 0 ||
        strcmp(report_path, checkpoint_path) == 0) {
        d3_usage(argv[0]);
        return EXIT_FAILURE;
    }

    if (!stage4_hash_file_sha256_hex(dataset_path, dataset_sha) ||
        !stage4_hash_file_sha256_hex(cache_path, cache_sha) ||
        !stage4_hash_file_sha256_hex(checkpoint_path, checkpoint_sha) ||
        !stage4_hash_file_sha256_hex(tokenizer_path, tokenizer_sha) ||
        !stage4_hash_file_sha256_hex(d1_report_path, d1_sha) ||
        !stage4_hash_file_sha256_hex(d2_report_path, d2_sha) ||
        !d2_hex_to_raw(dataset_sha, dataset_raw) ||
        !d2_hex_to_raw(checkpoint_sha, checkpoint_raw) ||
        !d2_hex_to_raw(tokenizer_sha, tokenizer_raw) ||
        !d2_hex_to_raw(D3_EXPECTED_DATASET_SHA256, expected_dataset_raw) ||
        !d2_hex_to_raw(D3_EXPECTED_CHECKPOINT_SHA256, expected_checkpoint_raw) ||
        !d2_hex_to_raw(D3_EXPECTED_TOKENIZER_SHA256, expected_tokenizer_raw)) {
        fprintf(stderr, "error: cannot hash or decode required D3 identities\n");
        failed = 1;
        goto cleanup;
    }
    if (strcmp(dataset_sha, D3_EXPECTED_DATASET_SHA256) != 0 ||
        strcmp(cache_sha, D3_EXPECTED_CACHE_SHA256) != 0 ||
        strcmp(checkpoint_sha, D3_EXPECTED_CHECKPOINT_SHA256) != 0 ||
        strcmp(tokenizer_sha, D3_EXPECTED_TOKENIZER_SHA256) != 0 ||
        strcmp(d1_sha, D3_EXPECTED_D1_REPORT_SHA256) != 0 ||
        strcmp(d2_sha, D3_EXPECTED_D2_REPORT_SHA256) != 0) {
        fprintf(stderr, "error: one or more pinned D3 input identities mismatch\n");
        failed = 1;
        goto cleanup;
    }
    if (!d2_float_format_supported() ||
        !stage4_file_size_bytes(cache_path, &cache_size) ||
        cache_size != D2_CACHE_BYTES) {
        fprintf(stderr, "error: A4 cache size or float format mismatch\n");
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
            (row_index > 0U && rows[row_index].record_index < rows[row_index - 1U].record_index) ||
            (row_index > 0U && rows[row_index].record_index == rows[row_index - 1U].record_index &&
                (rows[row_index].split_id != rows[row_index - 1U].split_id ||
                 rows[row_index].input_position <= rows[row_index - 1U].input_position))) {
            fprintf(stderr, "error: invalid cache row %u\n", row_index);
            failed = 1;
            goto cleanup;
        }
        split_rows[rows[row_index].split_id]++;
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
    if (split_rows[D2_SPLIT_TRAIN] != 111U ||
        split_rows[D2_SPLIT_VALIDATION] != 33U ||
        split_rows[D2_SPLIT_TEST] != 21U ||
        split_rows[D2_SPLIT_REGRESSION] != 0U) {
        fprintf(stderr, "error: cache split counts mismatch\n");
        failed = 1;
        goto cleanup;
    }

    unload_adapter_runtime();
    load_transformer(&transformer, checkpoint_path);
    transformer_ready = 1;
    if (transformer.config.dim != (int)D2_CACHE_DIM ||
        transformer.config.vocab_size != (int)D2_CACHE_VOCAB) {
        fprintf(stderr, "error: checkpoint dimensions mismatch\n");
        failed = 1;
        goto cleanup;
    }
    if (!stage4_adapter_init(&adapter, transformer.config.dim,
        transformer.config.vocab_size, D2_RANK, D2_SCALE)) {
        fprintf(stderr, "error: rank-8 adapter init failed\n");
        failed = 1;
        goto cleanup;
    }
    adapter_ready = 1;
    stage4_adapter_fill_a_deterministic(&adapter, 1U, 0.01f);
    stage4_adapter_zero_b(&adapter);
    stage4_adapter_zero_moments(&adapter);
    if (adapter.rank != D2_RANK || adapter.scale != D2_SCALE ||
        adapter.a_count != 2304U || adapter.b_count != 256000U) {
        fprintf(stderr, "error: adapter shape mismatch\n");
        failed = 1;
        goto cleanup;
    }
    {
        size_t i;
        for (i = 0U; i < adapter.b_count; i++) {
            if (d2_float_bits(adapter.b[i]) != 0U) {
                fprintf(stderr, "error: initial B not zero\n");
                failed = 1;
                goto cleanup;
            }
        }
    }
    initial_a = malloc(adapter.a_count * sizeof(float));
    initial_b = malloc(adapter.b_count * sizeof(float));
    step1_a = malloc(adapter.a_count * sizeof(float));
    step1_b = malloc(adapter.b_count * sizeof(float));
    base_logits = calloc(D2_CACHE_VOCAB, sizeof(float));
    final_logits = calloc(D2_CACHE_VOCAB, sizeof(float));
    if (initial_a == NULL || initial_b == NULL || step1_a == NULL ||
        step1_b == NULL || base_logits == NULL || final_logits == NULL) {
        fprintf(stderr, "error: D3 buffer allocation failed\n");
        failed = 1;
        goto cleanup;
    }
    memcpy(initial_a, adapter.a, adapter.a_count * sizeof(float));
    memcpy(initial_b, adapter.b, adapter.b_count * sizeof(float));

    if (!d2_evaluate(rows, &adapter, &transformer, &initial_metrics, NULL, NULL)) {
        fprintf(stderr, "error: initial evaluation failed\n");
        failed = 1;
        goto cleanup;
    }
    initial_d1_match = d2_preupdate_matches(&initial_metrics);
    if (!initial_d1_match) {
        fprintf(stderr, "error: initialization metrics do not reproduce D1\n");
        failed = 1;
        goto cleanup;
    }

    optimizer_steps_before = 0U;
    if (!d3_accumulate_and_clip(rows, &adapter, &transformer,
        base_logits, final_logits, 1, &step1)) {
        fprintf(stderr, "error: step-1 gradient pass failed; step not performed\n");
        failed = 1;
        goto cleanup;
    }
    update_norm = stage4_adapter_adam_step(&adapter,
        (float)D2_LEARNING_RATE, (float)D2_BETA1, (float)D2_BETA2,
        (float)D2_EPSILON, (float)D2_WEIGHT_DECAY, 1U);
    optimizer_steps_after = 1U;
    if (!isfinite(update_norm) || !(update_norm > 0.0) ||
        !stage4_adapter_parameters_are_finite(&adapter)) {
        fprintf(stderr, "error: D3 step-1 Adam update failed\n");
        failed = 1;
        goto cleanup;
    }
    step1.adapter_a_changed = memcmp(initial_a, adapter.a,
        adapter.a_count * sizeof(float)) != 0;
    step1.adapter_b_changed = memcmp(initial_b, adapter.b,
        adapter.b_count * sizeof(float)) != 0;
    d2_parameter_deltas(&adapter, initial_a, initial_b,
        &step1_a_delta, &step1_b_delta);
    d2_moment_nonzero_counts(&adapter, &final_m1_a, &final_m2_a,
        &final_m1_b, &final_m2_b);
    step1.m1_a_nonzero = final_m1_a;
    step1.m2_a_nonzero = final_m2_a;
    step1.m1_b_nonzero = final_m1_b;
    step1.m2_b_nonzero = final_m2_b;
    if (!d2_evaluate(rows, &adapter, &transformer, &step1.evaluation,
        NULL, NULL)) {
        fprintf(stderr, "error: step-1 evaluation failed\n");
        failed = 1;
        goto cleanup;
    }
    step1_d2_match = d3_step1_matches_d2(&step1, &adapter,
        &step1.evaluation, step1.adapter_a_changed, step1.adapter_b_changed,
        step1_a_delta, step1_b_delta, final_m1_a, final_m2_a,
        final_m1_b, final_m2_b);
    if (!step1_d2_match) {
        fprintf(stderr, "error: step 1 does not reproduce the accepted D2 report\n");
        failed = 1;
        goto cleanup;
    }
    optimizer_steps_after = 1U;
    memcpy(step1_a, adapter.a, adapter.a_count * sizeof(float));
    memcpy(step1_b, adapter.b, adapter.b_count * sizeof(float));

    if (!d3_accumulate_and_clip(rows, &adapter, &transformer,
        base_logits, final_logits, 2, &step2)) {
        fprintf(stderr, "error: step-2 gradient pass failed; second update not performed\n");
        failed = 1;
        goto cleanup;
    }
    update_norm = stage4_adapter_adam_step(&adapter,
        (float)D2_LEARNING_RATE, (float)D2_BETA1, (float)D2_BETA2,
        (float)D2_EPSILON, (float)D2_WEIGHT_DECAY, 2U);
    optimizer_steps_after = 2U;
    if (!isfinite(update_norm) || !(update_norm > 0.0) ||
        !stage4_adapter_parameters_are_finite(&adapter)) {
        fprintf(stderr, "error: D3 step-2 Adam update failed\n");
        failed = 1;
        goto cleanup;
    }
    step2.adapter_a_changed = memcmp(step1_a, adapter.a,
        adapter.a_count * sizeof(float)) != 0;
    step2.adapter_b_changed = memcmp(step1_b, adapter.b,
        adapter.b_count * sizeof(float)) != 0;
    d2_parameter_deltas(&adapter, step1_a, step1_b,
        &step2.adapter_a_max_abs_delta, &step2.adapter_b_max_abs_delta);
    initial_a_changed_final = memcmp(initial_a, adapter.a,
        adapter.a_count * sizeof(float)) != 0;
    initial_b_changed_final = memcmp(initial_b, adapter.b,
        adapter.b_count * sizeof(float)) != 0;
    d2_parameter_deltas(&adapter, initial_a, initial_b,
        &final_a_max_delta, &final_b_max_delta);
    d2_moment_nonzero_counts(&adapter, &final_m1_a, &final_m2_a,
        &final_m1_b, &final_m2_b);
    if (!d2_evaluate(rows, &adapter, &transformer, &step2.evaluation,
        &final_zero_correction_rows, &final_nonzero_correction_rows)) {
        fprintf(stderr, "error: step-2 evaluation failed\n");
        failed = 1;
        goto cleanup;
    }
    step2_vs_step1_delta =
        step2.evaluation.splits[D2_SPLIT_TRAIN].loss_sum /
            (double)step2.evaluation.splits[D2_SPLIT_TRAIN].target_count -
        step1.evaluation.splits[D2_SPLIT_TRAIN].loss_sum /
            (double)step1.evaluation.splits[D2_SPLIT_TRAIN].target_count;
    step2_vs_initial_delta =
        step2.evaluation.splits[D2_SPLIT_TRAIN].loss_sum /
            (double)step2.evaluation.splits[D2_SPLIT_TRAIN].target_count -
        initial_metrics.splits[D2_SPLIT_TRAIN].loss_sum /
            (double)initial_metrics.splits[D2_SPLIT_TRAIN].target_count;
    validation_acceptance_reached =
        step2.evaluation.splits[D2_SPLIT_VALIDATION].loss_sum /
            (double)step2.evaluation.splits[D2_SPLIT_VALIDATION].target_count <=
        8.650949130533;

    passed = optimizer_steps_before == 0U && optimizer_steps_after == 2U &&
        initial_d1_match && step1_d2_match &&
        step2.gradient_train_rows == 111U &&
        step2.gradient_validation_rows == 0U && step2.gradient_test_rows == 0U &&
        step2.gradient.a_nonzero > 0U && step2.gradient.a_max_abs > 0.0 &&
        step2.gradient.b_nonzero > 0U && step2.gradient.b_max_abs > 0.0 &&
        step2.adapter_a_changed && step2.adapter_b_changed &&
        step2.adapter_a_max_abs_delta > 0.0 && step2.adapter_b_max_abs_delta > 0.0 &&
        initial_a_changed_final && initial_b_changed_final &&
        final_m1_a > 0U && final_m2_a > 0U && final_m1_b > 0U && final_m2_b > 0U &&
        step2.evaluation.splits[D2_SPLIT_TRAIN].loss_sum /
            (double)step2.evaluation.splits[D2_SPLIT_TRAIN].target_count <
        initial_metrics.splits[D2_SPLIT_TRAIN].loss_sum /
            (double)initial_metrics.splits[D2_SPLIT_TRAIN].target_count &&
        final_nonzero_correction_rows > 0U &&
        step1.clip_consistency && step2.clip_consistency;

    if (!d3_write_report(report_path, dataset_sha, cache_sha, checkpoint_sha,
        tokenizer_sha, d1_sha, d2_sha, &adapter, &initial_metrics, &step1, &step2,
        step1.adapter_a_changed, step1.adapter_b_changed, step1_a_delta,
        step1_b_delta, initial_a_changed_final, initial_b_changed_final,
        final_a_max_delta, final_b_max_delta, optimizer_steps_before,
        optimizer_steps_after, final_m1_a, final_m2_a, final_m1_b, final_m2_b,
        step2_vs_step1_delta, step2_vs_initial_delta,
        validation_acceptance_reached, final_zero_correction_rows,
        final_nonzero_correction_rows, initial_d1_match, step1_d2_match, passed)) {
        fprintf(stderr, "error: cannot write D3 report '%s'\n", report_path);
        failed = 1;
        goto cleanup;
    }

    printf("base_checkpoint_identity=PASS\n");
    printf("tokenizer_identity=PASS\n");
    printf("dataset_identity=PASS\n");
    printf("cache_identity=PASS\n");
    printf("d1_report_identity=PASS\n");
    printf("d2_report_identity=PASS\n");
    printf("rank=%d\n", adapter.rank);
    printf("scale=%.1f\n", (double)adapter.scale);
    printf("adapter_parameter_count=%lu\n",
        (unsigned long)(adapter.a_count + adapter.b_count));
    printf("step1_d2_metrics_match=%s\n", step1_d2_match ? "yes" : "no");
    printf("step1_grad_A_nonzero_elements=%u\n", step1.gradient.a_nonzero);
    printf("step1_grad_B_nonzero_elements=%u\n", step1.gradient.b_nonzero);
    printf("step2_gradient_train_rows=%u\n", step2.gradient_train_rows);
    printf("step2_gradient_validation_rows=%u\n", step2.gradient_validation_rows);
    printf("step2_gradient_test_rows=%u\n", step2.gradient_test_rows);
    printf("step2_grad_A_nonzero_elements=%u\n", step2.gradient.a_nonzero);
    printf("step2_grad_A_max_abs=%.12g\n", step2.gradient.a_max_abs);
    printf("step2_grad_B_nonzero_elements=%u\n", step2.gradient.b_nonzero);
    printf("step2_grad_B_max_abs=%.12g\n", step2.gradient.b_max_abs);
    printf("step2_gradient_norm_before_clip=%.12g\n", step2.norm_before_clip);
    printf("step2_gradient_norm_after_clip=%.12g\n", step2.norm_after_clip);
    printf("step2_gradient_clip_applied=%s\n",
        step2.clip_applied ? "yes" : "no");
    printf("step2_adapter_A_changed=%s\n", step2.adapter_a_changed ? "yes" : "no");
    printf("step2_adapter_B_changed=%s\n", step2.adapter_b_changed ? "yes" : "no");
    printf("optimizer_steps_before=%llu\n",
        (unsigned long long)optimizer_steps_before);
    printf("optimizer_steps_after=%llu\n",
        (unsigned long long)optimizer_steps_after);
    printf("final_m1_A_nonzero_elements=%u\n", final_m1_a);
    printf("final_m2_A_nonzero_elements=%u\n", final_m2_a);
    printf("final_m1_B_nonzero_elements=%u\n", final_m1_b);
    printf("final_m2_B_nonzero_elements=%u\n", final_m2_b);
    printf("initial_train_average_loss=%.12f\n",
        initial_metrics.splits[D2_SPLIT_TRAIN].loss_sum /
            (double)initial_metrics.splits[D2_SPLIT_TRAIN].target_count);
    printf("step1_train_average_loss=%.12f\n",
        step1.evaluation.splits[D2_SPLIT_TRAIN].loss_sum /
            (double)step1.evaluation.splits[D2_SPLIT_TRAIN].target_count);
    printf("step2_train_average_loss=%.12f\n",
        step2.evaluation.splits[D2_SPLIT_TRAIN].loss_sum /
            (double)step2.evaluation.splits[D2_SPLIT_TRAIN].target_count);
    printf("initial_validation_average_loss=%.12f\n",
        initial_metrics.splits[D2_SPLIT_VALIDATION].loss_sum /
            (double)initial_metrics.splits[D2_SPLIT_VALIDATION].target_count);
    printf("step1_validation_average_loss=%.12f\n",
        step1.evaluation.splits[D2_SPLIT_VALIDATION].loss_sum /
            (double)step1.evaluation.splits[D2_SPLIT_VALIDATION].target_count);
    printf("step2_validation_average_loss=%.12f\n",
        step2.evaluation.splits[D2_SPLIT_VALIDATION].loss_sum /
            (double)step2.evaluation.splits[D2_SPLIT_VALIDATION].target_count);
    printf("step2_validation_acceptance_reached=%s\n",
        validation_acceptance_reached ? "yes" : "no");
    printf("final_nonzero_correction_rows=%u\n", final_nonzero_correction_rows);
    printf("Stage4E-D3-A4=%s\n", passed ? "PASS" : "FAIL");
    if (!passed)
        failed = 1;

cleanup:
    if (cache_file != NULL)
        fclose(cache_file);
    free(rows);
    free(base_logits);
    free(final_logits);
    free(initial_a);
    free(initial_b);
    free(step1_a);
    free(step1_b);
    if (adapter_ready)
        stage4_adapter_free(&adapter);
    if (transformer_ready)
        free_transformer(&transformer);
    return failed || !passed ? EXIT_FAILURE : EXIT_SUCCESS;
}
