#ifndef STAGE4E_D4_A4_TRAJECTORY_SOURCE
#define STAGE4E_D4_A4_TRAJECTORY_SOURCE

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

#ifndef STAGE4E_D4_CORE_ONLY
#define D4_PROVENANCE "ten-step-trajectory-2026-10-01-a"
#endif
#define D4_HEADER_SIZE 172U
#define D4_ENTRY_SIZE 1168U
#define D4_DIM 288U
#define D4_VOCAB 32000U
#define D4_RECORDS 30U
#define D4_ROWS 165U
#define D4_CACHE_SIZE 192892U
#define D4_SPLITS 4U
#define D4_TRAIN 0U
#define D4_VALIDATION 1U
#define D4_TEST 2U
#define D4_REGRESSION 3U
#define D4_STEPS 10U
#define D4_RANK 8
#define D4_SCALE 1.0f
#define D4_LR 0.001
#define D4_BETA1 0.9
#define D4_BETA2 0.999
#define D4_EPSILON 1.0e-8
#define D4_WEIGHT_DECAY 0.0
#define D4_CLIP 1.0

static const unsigned char d4_magic[8] = { 'S','4','T','C','A','C','H','1' };
#ifndef STAGE4E_D4_CORE_ONLY
static const char d4_dataset_sha[] = "71744cea4de049e2e192cb2a4494c8a4a68e769b3a7f3b8ed4880433e4583b84";
static const char d4_cache_sha[] = "d6288ad3d128c7182c20088aa57a1bca8bcaa1e13314fb0cdbd60eba14278def";
static const char d4_checkpoint_sha[] = "cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a";
static const char d4_tokenizer_sha[] = "50a52ef822ee9e83de5ce9d0be0a025a773d019437f58b5ff9dcafb063ece361";
static const char d4_d1_sha[] = "45ebaa750f17d02d80feed332b16c9c29c3782a544aed7544da72a62161c76a6";
static const char d4_d2_sha[] = "c27aad7474df093f5a3ff781bb3619b71b3e284edda46290f5814c350e99d97e";
static const char d4_d3_sha[] = "f418305f3fb9263480b8a80f34b02a966ebbb94f305f2aa5897d5c2f4356b94e";
#endif
static const char *d4_split_names[2] = { "train", "validation" };
static const uint32_t d4_expected_split_records[D4_SPLITS] = {20U, 5U, 5U, 0U};
static const uint32_t d4_expected_split_rows[D4_SPLITS] = {111U, 33U, 21U, 0U};
static const uint32_t d4_step1_top1[2] = {0U, 2U};
static const char *d4_step1_total[2] = {"1158.085676086679", "300.451596646300"};
static const char *d4_step1_average[2] = {"10.433204289069", "9.104593837767"};
static const char *d4_step2_total[2] = {"1156.021496625882", "300.274985419655"};
static const char *d4_step2_average[2] = {"10.414608077711", "9.099241982414"};
static const uint32_t d4_step2_top1[2] = {0U, 2U};
#ifndef STAGE4E_D4_CORE_ONLY
static const char *d4_validation_baseline = "9.106262242666";
static const char *d4_validation_max = "8.650949130533";
#endif

typedef struct {
    uint32_t version, header_size, entry_size, float_format, byte_order;
    uint32_t dim, vocab, records, rows;
    uint32_t split_records[D4_SPLITS];
    uint32_t split_rows[D4_SPLITS];
    unsigned char dataset[32], checkpoint[32], tokenizer[32];
} d4_header_t;

typedef struct {
    uint32_t record, position, target, split;
    float hidden[D4_DIM];
} d4_row_t;

typedef struct {
    uint32_t count;
    uint64_t top1;
    double loss;
} d4_split_metric_t;

typedef struct {
    uint32_t evaluated_rows;
    uint32_t evaluation_train_rows;
    uint32_t evaluation_validation_rows;
    uint32_t evaluation_test_rows;
    d4_split_metric_t split[2];
} d4_metrics_t;

typedef struct {
    uint32_t grad_train_rows, grad_validation_rows, grad_test_rows;
    uint32_t grad_a_nonzero, grad_b_nonzero;
    double grad_a_max, grad_b_max;
    double norm_before, norm_after;
    int clip_applied, clip_consistency;
    d4_metrics_t metrics;
} d4_point_t;

static uint32_t d4_u32(const unsigned char b[4])
{
    return (uint32_t)b[0] | ((uint32_t)b[1] << 8) |
        ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
}

static int d4_read_u32(FILE *f, uint32_t *out)
{
    unsigned char b[4];
    if (fread(b, 1, 4, f) != 4)
        return 0;
    *out = d4_u32(b);
    return 1;
}

static int d4_read_header(FILE *f, d4_header_t *h)
{
    unsigned char magic[8];
    uint32_t i;
    memset(h, 0, sizeof(*h));
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, d4_magic, 8) != 0)
        return 0;
    if (!d4_read_u32(f, &h->version) || !d4_read_u32(f, &h->header_size) ||
        !d4_read_u32(f, &h->entry_size) || !d4_read_u32(f, &h->float_format) ||
        !d4_read_u32(f, &h->byte_order) || !d4_read_u32(f, &h->dim) ||
        !d4_read_u32(f, &h->vocab) || !d4_read_u32(f, &h->records) ||
        !d4_read_u32(f, &h->rows))
        return 0;
    for (i = 0; i < D4_SPLITS; i++)
        if (!d4_read_u32(f, &h->split_records[i])) return 0;
    for (i = 0; i < D4_SPLITS; i++)
        if (!d4_read_u32(f, &h->split_rows[i])) return 0;
    return fread(h->dataset, 1, 32, f) == 32 &&
        fread(h->checkpoint, 1, 32, f) == 32 &&
        fread(h->tokenizer, 1, 32, f) == 32;
}

static int d4_read_row(FILE *f, d4_row_t *row)
{
    uint32_t i, bits;
    if (!d4_read_u32(f, &row->record) || !d4_read_u32(f, &row->position) ||
        !d4_read_u32(f, &row->target) || !d4_read_u32(f, &row->split))
        return 0;
    for (i = 0; i < D4_DIM; i++) {
        if (!d4_read_u32(f, &bits)) return 0;
        memcpy(&row->hidden[i], &bits, sizeof(bits));
        if (!isfinite(row->hidden[i])) return 0;
    }
    return 1;
}

static int d4_hex_raw(const char *hex, unsigned char out[32])
{
    uint32_t i;
    if (strlen(hex) != 64U) return 0;
    for (i = 0; i < 32U; i++) {
        int hi, lo;
        char a = hex[2U * i], b = hex[2U * i + 1U];
        hi = a >= '0' && a <= '9' ? a - '0' :
            (a >= 'a' && a <= 'f' ? a - 'a' + 10 : -1);
        lo = b >= '0' && b <= '9' ? b - '0' :
            (b >= 'a' && b <= 'f' ? b - 'a' + 10 : -1);
        if (hi < 0 || lo < 0) return 0;
        out[i] = (unsigned char)((hi << 4) | lo);
    }
    return 1;
}

static int d4_float_ok(void)
{
    const uint16_t endian = 1U;
    float one = 1.0f;
    uint32_t bits = 0U;
    if (sizeof(float) != 4U || sizeof(uint32_t) != 4U || FLT_RADIX != 2 ||
        FLT_MANT_DIG != 24 || FLT_MAX_EXP != 128 || FLT_MIN_EXP != -125 ||
        *((const unsigned char *)&endian) != 1U) return 0;
    memcpy(&bits, &one, 4U);
    return bits == UINT32_C(0x3f800000);
}

static int d4_header_matches(const d4_header_t *h,
    const unsigned char dataset[32], const unsigned char checkpoint[32],
    const unsigned char tokenizer[32])
{
    uint32_t i;
    if (h->version != 1U || h->header_size != D4_HEADER_SIZE ||
        h->entry_size != D4_ENTRY_SIZE || h->float_format != 1U ||
        h->byte_order != 1U || h->dim != D4_DIM || h->vocab != D4_VOCAB ||
        h->records != D4_RECORDS || h->rows != D4_ROWS ||
        memcmp(h->dataset, dataset, 32U) != 0 ||
        memcmp(h->checkpoint, checkpoint, 32U) != 0 ||
        memcmp(h->tokenizer, tokenizer, 32U) != 0) return 0;
    for (i = 0; i < D4_SPLITS; i++)
        if (h->split_records[i] != d4_expected_split_records[i] ||
            h->split_rows[i] != d4_expected_split_rows[i]) return 0;
    return 1;
}

static int d4_loss(const float *logits, uint32_t target, double *out)
{
    double max_logit, sum = 0.0, logprob;
    int i;
    if (target >= D4_VOCAB || !isfinite(logits[0])) return 0;
    max_logit = logits[0];
    for (i = 1; i < (int)D4_VOCAB; i++) {
        if (!isfinite(logits[i])) return 0;
        if (logits[i] > max_logit) max_logit = logits[i];
    }
    for (i = 0; i < (int)D4_VOCAB; i++)
        sum += exp((double)logits[i] - max_logit);
    if (!(sum > 0.0) || !isfinite(sum)) return 0;
    logprob = (double)logits[target] - max_logit - log(sum);
    *out = -logprob;
    return isfinite(*out);
}

static int d4_argmax(const float *logits)
{
    int best = 0, i;
    float best_value = logits[0];
    for (i = 1; i < (int)D4_VOCAB; i++)
        if (logits[i] > best_value) { best_value = logits[i]; best = i; }
    return best;
}

static void d4_flush_record(d4_metrics_t *m, uint32_t split,
    uint32_t count, double loss, uint64_t top1)
{
    m->split[split].count += count;
    m->split[split].loss += loss;
    m->split[split].top1 += top1;
}

static int d4_evaluate(const d4_row_t *rows, stage4_adapter_t *ad,
    Transformer *tr, d4_metrics_t *m, int *losses_finite)
{
    float *base = calloc(D4_VOCAB, sizeof(float));
    float *final = calloc(D4_VOCAB, sizeof(float));
    uint32_t i, current_record = 0U, current_split = 0U, record_rows = 0U;
    uint64_t record_top1 = 0U;
    double record_loss = 0.0;
    int have_record = 0;
    if (base == NULL || final == NULL) { free(base); free(final); return 0; }
    memset(m, 0, sizeof(*m));
    for (i = 0; i < D4_ROWS; i++) {
        const d4_row_t *row = &rows[i];
        double loss;
        int prediction;
        if (row->split == D4_TEST) continue;
        if (row->split > D4_VALIDATION) { free(base); free(final); return 0; }
        if (!have_record || row->record != current_record) {
            if (have_record) d4_flush_record(m, current_split, record_rows,
                record_loss, record_top1);
            current_record = row->record;
            current_split = row->split;
            record_rows = 0U;
            record_loss = 0.0;
            record_top1 = 0U;
            have_record = 1;
        }
        if (row->split != current_split) { free(base); free(final); return 0; }
        matmul(base, row->hidden, tr->weights.wcls, (int)D4_DIM, (int)D4_VOCAB);
        stage4_adapter_forward_logits(ad, row->hidden, base, final);
        if (!d4_loss(final, row->target, &loss)) {
            *losses_finite = 0;
            free(base); free(final); return 0;
        }
        prediction = d4_argmax(final);
        record_loss += loss;
        record_rows++;
        m->evaluated_rows++;
        if (row->split == D4_TRAIN) m->evaluation_train_rows++;
        else if (row->split == D4_VALIDATION) m->evaluation_validation_rows++;
        if (prediction == (int)row->target) record_top1++;
    }
    if (have_record) d4_flush_record(m, current_split, record_rows,
        record_loss, record_top1);
    free(base); free(final);
    return m->evaluation_train_rows == 111U &&
        m->evaluation_validation_rows == 33U && m->evaluation_test_rows == 0U;
}

static void d4_gradient_stats(const stage4_adapter_t *ad, d4_point_t *p)
{
    size_t i;
    p->grad_a_nonzero = p->grad_b_nonzero = 0U;
    p->grad_a_max = p->grad_b_max = 0.0;
    for (i = 0; i < ad->a_count; i++) {
        double x = fabs((double)ad->grad_a[i]);
        if (ad->grad_a[i] != 0.0f) p->grad_a_nonzero++;
        if (x > p->grad_a_max) p->grad_a_max = x;
    }
    for (i = 0; i < ad->b_count; i++) {
        double x = fabs((double)ad->grad_b[i]);
        if (ad->grad_b[i] != 0.0f) p->grad_b_nonzero++;
        if (x > p->grad_b_max) p->grad_b_max = x;
    }
}

static int d4_step(const d4_row_t *rows, stage4_adapter_t *ad,
    Transformer *tr, float *base, float *final, int step_index, d4_point_t *p,
    int *gradients_finite, int *losses_finite)
{
    uint32_t i;
    stage4_adapter_zero_grad(ad);
    p->grad_train_rows = p->grad_validation_rows = p->grad_test_rows = 0U;
    for (i = 0; i < D4_ROWS; i++) {
        const d4_row_t *row = &rows[i];
        double loss;
        if (row->split == D4_VALIDATION || row->split == D4_TEST) continue;
        if (row->split != D4_TRAIN) return 0;
        matmul(base, row->hidden, tr->weights.wcls, (int)D4_DIM, (int)D4_VOCAB);
        stage4_adapter_forward_logits(ad, row->hidden, base, final);
        if (!stage4_softmax_loss_and_dlogits(final, ad->vocab,
            (int)row->target, &loss, ad->dlogits)) { *losses_finite = 0; return 0; }
        stage4_adapter_backward(ad, row->hidden, ad->dlogits);
        p->grad_train_rows++;
    }
    if (p->grad_train_rows != 111U || p->grad_validation_rows != 0U ||
        p->grad_test_rows != 0U) return 0;
    stage4_adapter_scale_gradients(ad, 1.0f / (float)p->grad_train_rows);
    *gradients_finite = stage4_adapter_gradient_is_finite(ad);
    if (!*gradients_finite) return 0;
    d4_gradient_stats(ad, p);
    p->norm_before = stage4_adapter_gradient_norm(ad);
    if (!isfinite(p->norm_before) || !(p->norm_before > 0.0) ||
        (step_index == 1 && p->grad_a_nonzero != 0U) ||
        (step_index > 1 && p->grad_a_nonzero == 0U) ||
        p->grad_b_nonzero == 0U) return 0;
    p->clip_applied = p->norm_before > D4_CLIP;
    stage4_adapter_clip_gradients(ad, D4_CLIP);
    p->norm_after = stage4_adapter_gradient_norm(ad);
    if (p->clip_applied)
        p->clip_consistency = isfinite(p->norm_after) &&
            fabs(p->norm_after - D4_CLIP) <= 0.000001;
    else
        p->clip_consistency = memcmp(&p->norm_after, &p->norm_before,
            sizeof(p->norm_after)) == 0;
    return p->clip_consistency;
}

static int d4_fmt12(double x, const char *expected)
{
    char b[64];
    if (snprintf(b, sizeof(b), "%.12f", x) < 0) return 0;
    return strcmp(b, expected) == 0;
}

static int d4_fmtg(double x, const char *expected)
{
    char b[64];
    if (snprintf(b, sizeof(b), "%.12g", x) < 0) return 0;
    return strcmp(b, expected) == 0;
}

static void d4_moment_counts(const stage4_adapter_t *ad, uint32_t *m1a,
    uint32_t *m2a, uint32_t *m1b, uint32_t *m2b)
{
    size_t i;
    *m1a=*m2a=*m1b=*m2b=0U;
    for (i=0;i<ad->a_count;i++) { if(ad->m1_a[i]!=0.0f)(*m1a)++; if(ad->m2_a[i]!=0.0f)(*m2a)++; }
    for (i=0;i<ad->b_count;i++) { if(ad->m1_b[i]!=0.0f)(*m1b)++; if(ad->m2_b[i]!=0.0f)(*m2b)++; }
}

static void d4_state_finite(const stage4_adapter_t *ad,
    int *parameters_finite, int *moments_finite)
{
    size_t i;
    *parameters_finite = 1;
    *moments_finite = 1;
    for (i = 0U; i < ad->a_count; i++) {
        if (!isfinite(ad->a[i])) *parameters_finite = 0;
        if (!isfinite(ad->m1_a[i]) || !isfinite(ad->m2_a[i]))
            *moments_finite = 0;
    }
    for (i = 0U; i < ad->b_count; i++) {
        if (!isfinite(ad->b[i])) *parameters_finite = 0;
        if (!isfinite(ad->m1_b[i]) || !isfinite(ad->m2_b[i]))
            *moments_finite = 0;
    }
}

#ifndef STAGE4E_D4_CORE_ONLY
static int d4_write_outputs(const char *report_path, const char *tsv_path,
    const char *dataset_sha, const char *cache_sha, const char *checkpoint_sha,
    const char *tokenizer_sha, const char *d1_sha, const char *d2_sha,
    const char *d3_sha, const d4_point_t points[D4_STEPS+1],
    int step0_match, int step1_match, int step2_match,
    int final_a_changed, int final_b_changed, double initial_a_delta,
    double initial_b_delta, uint32_t m1a, uint32_t m2a, uint32_t m1b,
    uint32_t m2b, uint32_t train_improved, uint32_t train_same,
    uint32_t train_worse, uint32_t val_improved, uint32_t val_same,
    uint32_t val_worse, int all_grad_finite, int all_param_finite,
    int all_moments_finite, int all_losses_finite, int passed,
    uint64_t optimizer_steps_before, uint64_t optimizer_steps_after,
    uint32_t test_gradient_rows, uint32_t test_evaluation_rows)
{
    FILE *r = fopen(report_path, "wb");
    FILE *t = fopen(tsv_path, "wb");
    uint32_t i;
    int ok = 1;
    if (r == NULL || t == NULL) { if (r) fclose(r); if (t) fclose(t); return 0; }
    fprintf(t, "step\ttrain_average_loss\ttrain_top1_correct\tvalidation_average_loss\tvalidation_top1_correct\tvalidation_acceptance_reached\tgradient_norm_before_clip\tgradient_norm_after_clip\tgradient_clip_applied\n");
    for (i=0;i<=D4_STEPS;i++) {
        double ta=points[i].metrics.split[D4_TRAIN].loss/111.0;
        double va=points[i].metrics.split[D4_VALIDATION].loss/33.0;
        int accepted=va<=8.650949130533;
        if (i==0) fprintf(t, "%u\t%.12f\t%llu\t%.12f\t%llu\t%s\tNA\tNA\tNA\n", i, ta, (unsigned long long)points[i].metrics.split[D4_TRAIN].top1, va, (unsigned long long)points[i].metrics.split[D4_VALIDATION].top1, accepted?"yes":"no");
        else fprintf(t, "%u\t%.12f\t%llu\t%.12f\t%llu\t%s\t%.12g\t%.12g\t%s\n", i, ta, (unsigned long long)points[i].metrics.split[D4_TRAIN].top1, va, (unsigned long long)points[i].metrics.split[D4_VALIDATION].top1, accepted?"yes":"no", points[i].norm_before, points[i].norm_after, points[i].clip_applied?"yes":"no");
    }
    fprintf(r,
        "dataset_sha256=%s\ncache_sha256=%s\nbase_checkpoint_sha256=%s\ntokenizer_sha256=%s\nd1_report_sha256=%s\nd2_report_sha256=%s\nd3_report_sha256=%s\n"
        "rank=%d\nscale=%.1f\nadapter_parameter_count=%lu\n"
        "learning_rate=%.12g\nbeta1=%.12g\nbeta2=%.12g\nepsilon=%.12g\nweight_decay=%.12g\ngradient_clip_threshold=%.1f\n"
        "step0_d1_metrics_match=%s\nstep1_d2_metrics_match=%s\nstep2_d3_metrics_match=%s\n"
        "test_gradient_rows_seen=%u\ntest_evaluation_rows_seen=%u\noptimizer_steps_before=%llu\noptimizer_steps_after=%llu\n"
        "final_adapter_A_changed_from_initial=%s\nfinal_adapter_B_changed_from_initial=%s\n"
        "final_m1_A_nonzero_elements=%u\nfinal_m2_A_nonzero_elements=%u\nfinal_m1_B_nonzero_elements=%u\nfinal_m2_B_nonzero_elements=%u\n"
        "final_adapter_A_max_abs_delta=%.12g\nfinal_adapter_B_max_abs_delta=%.12g\n"
        "initial_train_average_loss=%.12f\nfinal_train_average_loss=%.12f\nfinal_vs_initial_train_loss_delta=%.12f\n"
        "initial_validation_average_loss=%.12f\nfinal_validation_average_loss=%.12f\nfinal_vs_initial_validation_loss_delta=%.12f\n"
        "train_steps_improved_from_previous=%u\ntrain_steps_unchanged_from_previous=%u\ntrain_steps_worse_than_previous=%u\n"
        "validation_steps_improved_from_previous=%u\nvalidation_steps_unchanged_from_previous=%u\nvalidation_steps_worse_than_previous=%u\n"
        "validation_baseline_loss=%s\nvalidation_required_relative_improvement=0.05\nvalidation_acceptance_loss_max=%s\nfinal_validation_acceptance_reached=%s\n"
        "all_gradients_finite=%s\nall_parameters_finite=%s\nall_optimizer_moments_finite=%s\nall_losses_finite=%s\nd4_trajectory_pass=%s\n",
        dataset_sha,cache_sha,checkpoint_sha,tokenizer_sha,d1_sha,d2_sha,d3_sha,
        D4_RANK,(double)D4_SCALE,(unsigned long)258304,D4_LR,D4_BETA1,D4_BETA2,D4_EPSILON,D4_WEIGHT_DECAY,D4_CLIP,
        step0_match?"yes":"no",step1_match?"yes":"no",step2_match?"yes":"no",
        test_gradient_rows,test_evaluation_rows,
        (unsigned long long)optimizer_steps_before,
        (unsigned long long)optimizer_steps_after,
        final_a_changed?"yes":"no",final_b_changed?"yes":"no",m1a,m2a,m1b,m2b,
        initial_a_delta,initial_b_delta,points[0].metrics.split[D4_TRAIN].loss/111.0,
        points[D4_STEPS].metrics.split[D4_TRAIN].loss/111.0,
        points[D4_STEPS].metrics.split[D4_TRAIN].loss/111.0-points[0].metrics.split[D4_TRAIN].loss/111.0,
        points[0].metrics.split[D4_VALIDATION].loss/33.0,
        points[D4_STEPS].metrics.split[D4_VALIDATION].loss/33.0,
        points[D4_STEPS].metrics.split[D4_VALIDATION].loss/33.0-points[0].metrics.split[D4_VALIDATION].loss/33.0,
        train_improved,train_same,train_worse,val_improved,val_same,val_worse,
        d4_validation_baseline,d4_validation_max,
        points[D4_STEPS].metrics.split[D4_VALIDATION].loss/33.0<=8.650949130533?"yes":"no",
        all_grad_finite?"yes":"no",all_param_finite?"yes":"no",all_moments_finite?"yes":"no",all_losses_finite?"yes":"no",passed?"yes":"no");
    for (i=0;i<=D4_STEPS;i++) {
        int s;
        if (i>0) {
            fprintf(r,"step%u_gradient_train_rows=%u\nstep%u_gradient_validation_rows=%u\nstep%u_gradient_test_rows=%u\n",i,points[i].grad_train_rows,i,points[i].grad_validation_rows,i,points[i].grad_test_rows);
            fprintf(r,"step%u_grad_A_nonzero_elements=%u\nstep%u_grad_A_max_abs=%.12g\nstep%u_grad_B_nonzero_elements=%u\nstep%u_grad_B_max_abs=%.12g\nstep%u_gradient_norm_before_clip=%.17g\nstep%u_gradient_norm_after_clip=%.17g\nstep%u_gradient_clip_applied=%s\nstep%u_gradient_clip_consistency=%s\n",i,points[i].grad_a_nonzero,i,points[i].grad_a_max,i,points[i].grad_b_nonzero,i,points[i].grad_b_max,i,points[i].norm_before,i,points[i].norm_after,i,points[i].clip_applied?"yes":"no",i,points[i].clip_consistency?"yes":"no");
        }
        for (s=0;s<2;s++) fprintf(r,"step%u_%s_total_loss=%.12f\nstep%u_%s_average_loss=%.12f\nstep%u_%s_top1_correct=%llu\n",i,d4_split_names[s],points[i].metrics.split[s].loss,i,d4_split_names[s],points[i].metrics.split[s].loss/(double)points[i].metrics.split[s].count,i,d4_split_names[s],(unsigned long long)points[i].metrics.split[s].top1);
        fprintf(r,"step%u_evaluation_train_rows=%u\nstep%u_evaluation_validation_rows=%u\nstep%u_evaluation_test_rows=%u\n",i,points[i].metrics.evaluation_train_rows,i,points[i].metrics.evaluation_validation_rows,i,points[i].metrics.evaluation_test_rows);
        fprintf(r,"step%u_validation_acceptance_reached=%s\n",i,points[i].metrics.split[D4_VALIDATION].loss/33.0<=8.650949130533?"yes":"no");
    }
    if (fclose(r)!=0) ok=0;
    if (fclose(t)!=0) ok=0;
    return ok;
}
#endif

static int d4_anchor_metrics(const d4_metrics_t *m, uint32_t point)
{
    if (m->evaluation_train_rows!=111U || m->evaluation_validation_rows!=33U || m->evaluation_test_rows!=0U) return 0;
    if (point==0U)
        return d4_fmt12(m->split[D4_TRAIN].loss,"1158.750195944566") && d4_fmt12(m->split[D4_TRAIN].loss/111.0,"10.439190954456") && m->split[D4_TRAIN].top1==0U && d4_fmt12(m->split[D4_VALIDATION].loss,"300.506654007981") && d4_fmt12(m->split[D4_VALIDATION].loss/33.0,"9.106262242666") && m->split[D4_VALIDATION].top1==2U;
    if (point==1U)
        return d4_fmt12(m->split[D4_TRAIN].loss,d4_step1_total[D4_TRAIN]) && d4_fmt12(m->split[D4_TRAIN].loss/111.0,d4_step1_average[D4_TRAIN]) && m->split[D4_TRAIN].top1==d4_step1_top1[D4_TRAIN] && d4_fmt12(m->split[D4_VALIDATION].loss,d4_step1_total[D4_VALIDATION]) && d4_fmt12(m->split[D4_VALIDATION].loss/33.0,d4_step1_average[D4_VALIDATION]) && m->split[D4_VALIDATION].top1==d4_step1_top1[D4_VALIDATION];
    if (point==2U)
        return d4_fmt12(m->split[D4_TRAIN].loss,d4_step2_total[D4_TRAIN]) && d4_fmt12(m->split[D4_TRAIN].loss/111.0,d4_step2_average[D4_TRAIN]) && m->split[D4_TRAIN].top1==d4_step2_top1[D4_TRAIN] && d4_fmt12(m->split[D4_VALIDATION].loss,d4_step2_total[D4_VALIDATION]) && d4_fmt12(m->split[D4_VALIDATION].loss/33.0,d4_step2_average[D4_VALIDATION]) && m->split[D4_VALIDATION].top1==d4_step2_top1[D4_VALIDATION];
    return 1;
}

static int d4_gradient_anchor(const d4_point_t *p, uint32_t index)
{
    if (index==1U)
        return p->grad_a_nonzero==0U && p->grad_a_max==0.0 && p->grad_b_nonzero==256000U && d4_fmtg(p->grad_b_max,"0.145234793425") && d4_fmtg(p->norm_before,"0.381956843381") && d4_fmtg(p->norm_after,"0.381956843381") && !p->clip_applied;
    if (index==2U)
        return p->grad_a_nonzero==2304U && d4_fmtg(p->grad_a_max,"0.0729372277856") && p->grad_b_nonzero==256000U && d4_fmtg(p->grad_b_max,"0.145245939493") && d4_fmtg(p->norm_before,"0.487589763461") && d4_fmtg(p->norm_after,"0.487589763461") && !p->clip_applied;
    return 1;
}

#ifndef STAGE4E_D4_CORE_ONLY
int main(int argc, char **argv)
{
    const char *checkpoint_path = NULL, *tokenizer_path = NULL, *dataset_path = NULL;
    const char *cache_path = NULL, *d1_path = NULL, *d2_path = NULL, *d3_path = NULL;
    const char *report_path = "stage4e-d4-a4-trajectory-report.txt";
    const char *tsv_path = "stage4e-d4-a4-trajectory.tsv";
    char ds_sha[65], ca_sha[65], ck_sha[65], tk_sha[65], d1_sha[65], d2_sha[65], d3_sha[65];
    unsigned char ds_raw[32], ck_raw[32], tk_raw[32], expected_raw[32];
    d4_header_t header;
    d4_row_t *rows = NULL;
    d4_point_t points[D4_STEPS+1];
    stage4_adapter_t ad;
    Transformer tr;
    FILE *cache = NULL;
    float *base = NULL, *final = NULL, *initial_a = NULL, *initial_b = NULL;
    uint32_t counts[D4_SPLITS]={0U,0U,0U,0U}, i;
    uint32_t test_gradient_rows=0U, test_evaluation_rows=0U;
    uint32_t train_improved=0U, train_same=0U, train_worse=0U;
    uint32_t val_improved=0U, val_same=0U, val_worse=0U;
    uint32_t m1a=0U,m2a=0U,m1b=0U,m2b=0U;
    uint64_t cache_bytes=0U, optimizer_steps_before=0U, optimizer_steps_after=0U;
    int argi, tr_ready=0, ad_ready=0, all_grad_finite=1;
    int all_param_finite=1, all_moments_finite=1, all_losses_finite=1;
    int step0_match=0,step1_match=0,step2_match=0,final_a_changed=0,final_b_changed=0;
    int passed=0, failed=0;
    double initial_a_delta=0.0,initial_b_delta=0.0;

    printf("d4_build_provenance=%s\n",D4_PROVENANCE);
    memset(&header,0,sizeof(header)); memset(points,0,sizeof(points));
    memset(&ad,0,sizeof(ad)); memset(&tr,0,sizeof(tr));
    if (argc<2) { fprintf(stderr,"usage: %s checkpoint -z tokenizer -d dataset -c cache -1 d1 -2 d2 -3 d3\n",argv[0]); return 1; }
    checkpoint_path=argv[1];
    for (argi=2;argi<argc;argi++) {
        if(!strcmp(argv[argi],"-z")&&argi+1<argc) tokenizer_path=argv[++argi];
        else if(!strcmp(argv[argi],"-d")&&argi+1<argc) dataset_path=argv[++argi];
        else if(!strcmp(argv[argi],"-c")&&argi+1<argc) cache_path=argv[++argi];
        else if(!strcmp(argv[argi],"-1")&&argi+1<argc) d1_path=argv[++argi];
        else if(!strcmp(argv[argi],"-2")&&argi+1<argc) d2_path=argv[++argi];
        else if(!strcmp(argv[argi],"-3")&&argi+1<argc) d3_path=argv[++argi];
        else { fprintf(stderr,"error: invalid D4 option\n"); return 1; }
    }
    if(!tokenizer_path||!dataset_path||!cache_path||!d1_path||!d2_path||!d3_path) { fprintf(stderr,"error: missing D4 input path\n"); return 1; }
    if(!stage4_hash_file_sha256_hex(dataset_path,ds_sha)||!stage4_hash_file_sha256_hex(cache_path,ca_sha)||!stage4_hash_file_sha256_hex(checkpoint_path,ck_sha)||!stage4_hash_file_sha256_hex(tokenizer_path,tk_sha)||!stage4_hash_file_sha256_hex(d1_path,d1_sha)||!stage4_hash_file_sha256_hex(d2_path,d2_sha)||!stage4_hash_file_sha256_hex(d3_path,d3_sha)) { fprintf(stderr,"error: input hash failure\n"); failed=1; goto done; }
    if(strcmp(ds_sha,d4_dataset_sha)||strcmp(ca_sha,d4_cache_sha)||strcmp(ck_sha,d4_checkpoint_sha)||strcmp(tk_sha,d4_tokenizer_sha)||strcmp(d1_sha,d4_d1_sha)||strcmp(d2_sha,d4_d2_sha)||strcmp(d3_sha,d4_d3_sha)) { fprintf(stderr,"error: D4 identity mismatch\n"); failed=1; goto done; }
    if(!d4_hex_raw(ds_sha,ds_raw)||!d4_hex_raw(ck_sha,ck_raw)||!d4_hex_raw(tk_sha,tk_raw)||!d4_hex_raw(d4_dataset_sha,expected_raw)) { fprintf(stderr,"error: identity decode failure\n"); failed=1; goto done; }
    if(!d4_float_ok()||!stage4_file_size_bytes(cache_path,&cache_bytes)||cache_bytes!=D4_CACHE_SIZE) { fprintf(stderr,"error: cache format/size mismatch\n"); failed=1; goto done; }
    cache=fopen(cache_path,"rb");
    if(!cache||!d4_read_header(cache,&header)||!d4_header_matches(&header,ds_raw,ck_raw,tk_raw)) { fprintf(stderr,"error: cache header mismatch\n"); failed=1; goto done; }
    rows=calloc(D4_ROWS,sizeof(*rows)); if(!rows) { fprintf(stderr,"error: cache row allocation failed\n"); failed=1; goto done; }
    for(i=0;i<D4_ROWS;i++) {
        if(!d4_read_row(cache,&rows[i])||rows[i].record>=D4_RECORDS||rows[i].target>=D4_VOCAB||rows[i].split>=D4_SPLITS||rows[i].split==D4_REGRESSION) { fprintf(stderr,"error: invalid cache row %u\n",i); failed=1; goto done; }
        if(i && (rows[i].record<rows[i-1].record||(rows[i].record==rows[i-1].record&&(rows[i].split!=rows[i-1].split||rows[i].position<=rows[i-1].position)))) { fprintf(stderr,"error: cache order invalid at row %u\n",i); failed=1; goto done; }
        counts[rows[i].split]++;
    }
    if(fgetc(cache)!=EOF||ferror(cache)||fclose(cache)!=0) { cache=NULL; fprintf(stderr,"error: cache trailing/read/close failure\n"); failed=1; goto done; }
    cache=NULL;
    if(counts[D4_TRAIN]!=111U||counts[D4_VALIDATION]!=33U||counts[D4_TEST]!=21U) { fprintf(stderr,"error: cache split counts mismatch\n"); failed=1; goto done; }
    unload_adapter_runtime(); load_transformer(&tr,checkpoint_path); tr_ready=1;
    if(tr.config.dim!=(int)D4_DIM||tr.config.vocab_size!=(int)D4_VOCAB||!stage4_adapter_init(&ad,tr.config.dim,tr.config.vocab_size,D4_RANK,D4_SCALE)) { fprintf(stderr,"error: model/adapter init failure\n"); failed=1; goto done; }
    ad_ready=1; stage4_adapter_fill_a_deterministic(&ad,1U,0.01f); stage4_adapter_zero_b(&ad); stage4_adapter_zero_moments(&ad);
    initial_a=malloc(ad.a_count*sizeof(float)); initial_b=malloc(ad.b_count*sizeof(float));
    base=calloc(D4_VOCAB,sizeof(float)); final=calloc(D4_VOCAB,sizeof(float));
    if(!initial_a||!initial_b||!base||!final) { fprintf(stderr,"error: D4 work allocation failure\n"); failed=1; goto done; }
    memcpy(initial_a,ad.a,ad.a_count*sizeof(float)); memcpy(initial_b,ad.b,ad.b_count*sizeof(float));
    optimizer_steps_before=0U;
    if(!d4_evaluate(rows,&ad,&tr,&points[0].metrics,&all_losses_finite)) { fprintf(stderr,"error: step-0 train/validation evaluation failed\n"); failed=1; goto done; }
    step0_match=d4_anchor_metrics(&points[0].metrics,0U);
    if(!step0_match) { fprintf(stderr,"error: D1 step-0 anchor failed\n"); failed=1; goto done; }

    for(i=1U;i<=D4_STEPS;i++) {
        d4_point_t *p=&points[i];
        double update_norm;
        int grad_ok=1, loss_ok=1;
        if(!d4_step(rows,&ad,&tr,base,final,(int)i,p,&grad_ok,&loss_ok)) { fprintf(stderr,"error: D4 gradient/clipping step %u failed\n",i); failed=1; goto done; }
        if(!grad_ok) all_grad_finite=0;
        if(!loss_ok) all_losses_finite=0;
        int params_ok, moments_ok;
        update_norm=stage4_adapter_adam_step(&ad,(float)D4_LR,(float)D4_BETA1,(float)D4_BETA2,(float)D4_EPSILON,(float)D4_WEIGHT_DECAY,(uint64_t)i);
        optimizer_steps_after++;
        d4_state_finite(&ad,&params_ok,&moments_ok);
        if(!isfinite(update_norm)||!params_ok) all_param_finite=0;
        if(!moments_ok) all_moments_finite=0;
        if(!d4_evaluate(rows,&ad,&tr,&p->metrics,&loss_ok)) { all_losses_finite=0; fprintf(stderr,"error: D4 train/validation eval failed at step %u\n",i); failed=1; goto done; }
        if(!loss_ok) all_losses_finite=0;
        if(!d4_anchor_metrics(&p->metrics,i)||!d4_gradient_anchor(p,i)) { fprintf(stderr,"error: deterministic D%d anchor failed\n",i); failed=1; goto done; }
        if(i==1U) step1_match=1; if(i==2U) step2_match=1;
        if(i>0U) {
            double oldt=points[i-1U].metrics.split[D4_TRAIN].loss/111.0, newt=p->metrics.split[D4_TRAIN].loss/111.0;
            double oldv=points[i-1U].metrics.split[D4_VALIDATION].loss/33.0, newv=p->metrics.split[D4_VALIDATION].loss/33.0;
            if(newt<oldt)train_improved++; else if(newt==oldt)train_same++; else train_worse++;
            if(newv<oldv)val_improved++; else if(newv==oldv)val_same++; else val_worse++;
        }
        printf("step=%u gradient_train_rows=%u gradient_validation_rows=%u gradient_test_rows=%u grad_A_nonzero_elements=%u grad_A_max_abs=%.12g grad_B_nonzero_elements=%u grad_B_max_abs=%.12g gradient_norm_before_clip=%.17g gradient_norm_after_clip=%.17g gradient_clip_applied=%s gradient_clip_consistency=%s train_average_loss=%.12f validation_average_loss=%.12f\n",i,p->grad_train_rows,p->grad_validation_rows,p->grad_test_rows,p->grad_a_nonzero,p->grad_a_max,p->grad_b_nonzero,p->grad_b_max,p->norm_before,p->norm_after,p->clip_applied?"yes":"no",p->clip_consistency?"yes":"no",p->metrics.split[D4_TRAIN].loss/111.0,p->metrics.split[D4_VALIDATION].loss/33.0);
    }
    {
        uint32_t point;
        double amax=0.0,bmax=0.0;
        size_t j;
        for(j=0;j<ad.a_count;j++){double x=fabs((double)ad.a[j]-(double)initial_a[j]);if(x>amax)amax=x;}
        for(j=0;j<ad.b_count;j++){double x=fabs((double)ad.b[j]-(double)initial_b[j]);if(x>bmax)bmax=x;}
        initial_a_delta=amax; initial_b_delta=bmax; final_a_changed=memcmp(initial_a,ad.a,ad.a_count*sizeof(float))!=0; final_b_changed=memcmp(initial_b,ad.b,ad.b_count*sizeof(float))!=0;
        d4_moment_counts(&ad,&m1a,&m2a,&m1b,&m2b);
        for(point=0;point<=D4_STEPS;point++) if(!isfinite(points[point].metrics.split[D4_TRAIN].loss)||!isfinite(points[point].metrics.split[D4_VALIDATION].loss))all_losses_finite=0;
    }
    passed=step0_match&&step1_match&&step2_match&&optimizer_steps_before==0U&&optimizer_steps_after==D4_STEPS&&
        ad.rank==D4_RANK&&ad.scale==D4_SCALE&&ad.a_count+ad.b_count==258304U&&
        final_a_changed&&final_b_changed&&initial_a_delta>0.0&&initial_b_delta>0.0&&
        m1a>0U&&m2a>0U&&m1b>0U&&m2b>0U&&all_grad_finite&&all_param_finite&&all_moments_finite&&all_losses_finite&&
        train_improved+train_same+train_worse==D4_STEPS&&val_improved+val_same+val_worse==D4_STEPS&&
        test_gradient_rows==0U&&test_evaluation_rows==0U&&
        points[D4_STEPS].metrics.split[D4_TRAIN].loss/111.0<points[0].metrics.split[D4_TRAIN].loss/111.0;
    if(!d4_write_outputs(report_path,tsv_path,ds_sha,ca_sha,ck_sha,tk_sha,d1_sha,d2_sha,d3_sha,points,step0_match,step1_match,step2_match,final_a_changed,final_b_changed,initial_a_delta,initial_b_delta,m1a,m2a,m1b,m2b,train_improved,train_same,train_worse,val_improved,val_same,val_worse,all_grad_finite,all_param_finite,all_moments_finite,all_losses_finite,passed,optimizer_steps_before,optimizer_steps_after,test_gradient_rows,test_evaluation_rows)){fprintf(stderr,"error: D4 report/trajectory write failed\n");failed=1;goto done;}
    printf("base_checkpoint_identity=PASS\ntokenizer_identity=PASS\ndataset_identity=PASS\ncache_identity=PASS\nd1_report_identity=PASS\nd2_report_identity=PASS\nd3_report_identity=PASS\nrank=8\nscale=1.0\nadapter_parameter_count=258304\nstep0_d1_metrics_match=%s\nstep1_d2_metrics_match=%s\nstep2_d3_metrics_match=%s\noptimizer_steps_before=%llu\noptimizer_steps_after=%llu\ntest_gradient_rows_seen=0\ntest_evaluation_rows_seen=0\ninitial_train_average_loss=%.12f\nfinal_train_average_loss=%.12f\nfinal_vs_initial_train_loss_delta=%.12f\ninitial_validation_average_loss=%.12f\nfinal_validation_average_loss=%.12f\nfinal_vs_initial_validation_loss_delta=%.12f\ntrain_steps_improved_from_previous=%u\ntrain_steps_unchanged_from_previous=%u\ntrain_steps_worse_than_previous=%u\nvalidation_steps_improved_from_previous=%u\nvalidation_steps_unchanged_from_previous=%u\nvalidation_steps_worse_than_previous=%u\nfinal_validation_acceptance_reached=%s\nall_gradients_finite=%s\nall_parameters_finite=%s\nall_optimizer_moments_finite=%s\nall_losses_finite=%s\nStage4E-D4-A4=%s\n",
        step0_match?"yes":"no",step1_match?"yes":"no",step2_match?"yes":"no",
        (unsigned long long)optimizer_steps_before,(unsigned long long)optimizer_steps_after,
        points[0].metrics.split[D4_TRAIN].loss/111.0,points[D4_STEPS].metrics.split[D4_TRAIN].loss/111.0,
        points[D4_STEPS].metrics.split[D4_TRAIN].loss/111.0-points[0].metrics.split[D4_TRAIN].loss/111.0,
        points[0].metrics.split[D4_VALIDATION].loss/33.0,points[D4_STEPS].metrics.split[D4_VALIDATION].loss/33.0,
        points[D4_STEPS].metrics.split[D4_VALIDATION].loss/33.0-points[0].metrics.split[D4_VALIDATION].loss/33.0,
        train_improved,train_same,train_worse,val_improved,val_same,val_worse,
        points[D4_STEPS].metrics.split[D4_VALIDATION].loss/33.0<=8.650949130533?"yes":"no",
        all_grad_finite?"yes":"no",all_param_finite?"yes":"no",all_moments_finite?"yes":"no",all_losses_finite?"yes":"no",passed?"PASS":"FAIL");
    if(!passed)failed=1;
done:
    if (cache != NULL)
        fclose(cache);
    free(rows);
    free(base);
    free(final);
    free(initial_a);
    free(initial_b);
    if (ad_ready)
        stage4_adapter_free(&ad);
    if (tr_ready)
        free_transformer(&tr);
    return failed||!passed?EXIT_FAILURE:EXIT_SUCCESS;
}
#endif
#endif
