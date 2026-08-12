#include "stage4_adapter_train.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int gradient_check_one_parameter(stage4_adapter_t *ad,
	const float *hidden,
	const float *base_logits,
	int target,
	float *param,
	double analytic,
	double epsilon,
	double rel_tol,
	double abs_tol,
	const char *tensor_name,
	size_t index)
{
	float *final_logits;
	double loss_plus;
	double loss_minus;
	double numerical;
	double abs_err;
	double rel_err;
	float original;
	int ok;

	final_logits = calloc((size_t)ad->vocab, sizeof(float));
	if (final_logits == NULL)
		return 0;

	original = *param;
	*param = original + (float)epsilon;
	stage4_adapter_forward_logits(ad, hidden, base_logits, final_logits);
	if (!stage4_softmax_loss_and_dlogits(final_logits, ad->vocab, target,
	    &loss_plus, ad->dlogits)) {
		free(final_logits);
		return 0;
	}

	*param = original - (float)epsilon;
	stage4_adapter_forward_logits(ad, hidden, base_logits, final_logits);
	if (!stage4_softmax_loss_and_dlogits(final_logits, ad->vocab, target,
	    &loss_minus, ad->dlogits)) {
		free(final_logits);
		return 0;
	}

	*param = original;
	free(final_logits);

	numerical = (loss_plus - loss_minus) / (2.0 * epsilon);
	abs_err = fabs(analytic - numerical);
	rel_err = abs_err / fmax(1e-12, fmax(fabs(analytic), fabs(numerical)));

	printf("grad_check tensor=%s index=%lu analytic=%.12e numerical=%.12e abs_err=%.12e rel_err=%.12e\n",
	    tensor_name, (unsigned long)index,
	    analytic, numerical, abs_err, rel_err);

	ok = (abs_err <= abs_tol) || (rel_err <= rel_tol);
	return ok;
}

int main(void)
{
	stage4_adapter_t ad;
	stage4_adapter_t reload;
	stage4_base_identity_t base_id;
	const int dim = 3;
	const int rank = 2;
	const int vocab = 5;
	const int target = 3;
	float hidden[3] = {0.20f, -0.10f, 0.70f};
	float base_logits[5] = {0.30f, -0.20f, 0.10f, 0.50f, -0.40f};
	float *final_logits;
	double loss_before;
	double loss_after;
	double loss_reload;
	double grad_norm_before_clip;
	double grad_norm_after_clip;
	double update_norm;
	double eps = 1e-4;
	double rel_tol = 7e-3;
	double abs_tol = 4e-5;
	double zero_tol = 1e-8;
	double loss_tol = 1e-7;
	double no_op_clip_threshold = 0.5;
	double active_clip_threshold = 0.005;
	int failures = 0;
	size_t i;
	char adapter_hash[65];
	stage4_storage_accounting_t mem;
	float *grad_a_before_clip;
	float *grad_b_before_clip;

	memset(&ad, 0, sizeof(ad));
	memset(&reload, 0, sizeof(reload));
	strcpy(base_id.base_sha256,
	    "cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a");
	base_id.base_checkpoint_bytes = 60816028ULL;

	if (!stage4_adapter_init(&ad, dim, vocab, rank, 1.0f)) {
		fprintf(stderr, "error: stage4_adapter_init failed\n");
		return 1;
	}

	if (!stage4_adapter_storage_accounting(&ad, &mem)) {
		fprintf(stderr, "error: storage accounting failed\n");
		stage4_adapter_free(&ad);
		return 1;
	}

	printf("synthetic_memory_accounting adapter_weight_bytes=%llu\n",
	    (unsigned long long)mem.adapter_weight_bytes);
	printf("synthetic_memory_accounting gradient_bytes=%llu\n",
	    (unsigned long long)mem.gradient_bytes);
	printf("synthetic_memory_accounting adam_first_moment_bytes=%llu\n",
	    (unsigned long long)mem.adam_first_moment_bytes);
	printf("synthetic_memory_accounting adam_second_moment_bytes=%llu\n",
	    (unsigned long long)mem.adam_second_moment_bytes);
	printf("synthetic_memory_accounting total_trainable_storage_bytes=%llu\n",
	    (unsigned long long)mem.total_trainable_storage_bytes);

	/* Deterministic nonzero A, nonzero B for gradient check quality. */
	stage4_adapter_fill_a_deterministic(&ad, 12345U, 0.05f);
	for (i = 0; i < ad.b_count; i++)
		ad.b[i] = (float)((int)(i % 7) - 3) * 0.01f;

	final_logits = calloc((size_t)vocab, sizeof(float));
	if (final_logits == NULL) {
		stage4_adapter_free(&ad);
		return 1;
	}
	grad_a_before_clip = calloc(ad.a_count, sizeof(float));
	grad_b_before_clip = calloc(ad.b_count, sizeof(float));
	if (grad_a_before_clip == NULL || grad_b_before_clip == NULL) {
		free(final_logits);
		free(grad_a_before_clip);
		free(grad_b_before_clip);
		stage4_adapter_free(&ad);
		return 1;
	}

	stage4_adapter_zero_grad(&ad);
	stage4_adapter_forward_logits(&ad, hidden, base_logits, final_logits);
	if (!stage4_softmax_loss_and_dlogits(final_logits, vocab, target,
	    &loss_before, ad.dlogits)) {
		fprintf(stderr, "error: softmax/loss failed\n");
		free(final_logits);
		stage4_adapter_free(&ad);
		return 1;
	}
	stage4_adapter_backward(&ad, hidden, ad.dlogits);

	for (i = 0; i < ad.a_count; i++) {
		if (!gradient_check_one_parameter(&ad, hidden, base_logits, target,
		    &ad.a[i], (double)ad.grad_a[i], eps, rel_tol, abs_tol, "A", i))
			failures++;
	}
	for (i = 0; i < ad.b_count; i++) {
		if (!gradient_check_one_parameter(&ad, hidden, base_logits, target,
		    &ad.b[i], (double)ad.grad_b[i], eps, rel_tol, abs_tol, "B", i))
			failures++;
	}

	printf("gradient_check epsilon=%.3e abs_tol=%.3e rel_tol=%.3e failures=%d\n",
	    eps, abs_tol, rel_tol, failures);
	if (failures != 0) {
		free(final_logits);
		stage4_adapter_free(&ad);
		return 2;
	}

	/* Zero-B behavior check. */
	stage4_adapter_fill_a_deterministic(&ad, 999U, 0.01f);
	stage4_adapter_zero_b(&ad);
	stage4_adapter_zero_grad(&ad);
	stage4_adapter_forward_logits(&ad, hidden, base_logits, final_logits);
	printf("zero_b max_adapter_logit_correction=%.12e\n",
	    stage4_adapter_max_logit_correction(base_logits, final_logits, vocab));
	if (!stage4_softmax_loss_and_dlogits(final_logits, vocab, target,
	    &loss_before, ad.dlogits)) {
		free(final_logits);
		stage4_adapter_free(&ad);
		return 3;
	}
	stage4_adapter_backward(&ad, hidden, ad.dlogits);

	if (!stage4_adapter_gradient_is_finite(&ad)) {
		fprintf(stderr, "error: non-finite gradients for zero-B check\n");
		free(final_logits);
		stage4_adapter_free(&ad);
		return 3;
	}

	{
		double max_abs_grad_a = 0.0;
		double max_abs_grad_b = 0.0;
		for (i = 0; i < ad.a_count; i++) {
			double v = fabs((double)ad.grad_a[i]);
			if (v > max_abs_grad_a)
				max_abs_grad_a = v;
		}
		for (i = 0; i < ad.b_count; i++) {
			double v = fabs((double)ad.grad_b[i]);
			if (v > max_abs_grad_b)
				max_abs_grad_b = v;
		}
		printf("zero_b max_abs_grad_A=%.12e max_abs_grad_B=%.12e\n",
		    max_abs_grad_a, max_abs_grad_b);
		if (max_abs_grad_b <= zero_tol || max_abs_grad_a > zero_tol) {
			fprintf(stderr,
			    "error: zero-B gradient behavior failed (expected grad_B nonzero and grad_A near zero)\n");
			free(final_logits);
			stage4_adapter_free(&ad);
			return 4;
		}
	}

	/* One synthetic SGD step should reduce loss. */
	stage4_adapter_fill_a_deterministic(&ad, 54321U, 0.03f);
	for (i = 0; i < ad.b_count; i++)
		ad.b[i] = (float)((int)(i % 5) - 2) * 0.02f;
	stage4_adapter_zero_grad(&ad);
	stage4_adapter_forward_logits(&ad, hidden, base_logits, final_logits);
	if (!stage4_softmax_loss_and_dlogits(final_logits, vocab, target,
	    &loss_before, ad.dlogits)) {
		free(final_logits);
		stage4_adapter_free(&ad);
		return 5;
	}
	stage4_adapter_backward(&ad, hidden, ad.dlogits);
	grad_norm_before_clip = stage4_adapter_gradient_norm(&ad);
	memcpy(grad_a_before_clip, ad.grad_a, ad.a_count * sizeof(float));
	memcpy(grad_b_before_clip, ad.grad_b, ad.b_count * sizeof(float));

	stage4_adapter_clip_gradients(&ad, no_op_clip_threshold);
	grad_norm_after_clip = stage4_adapter_gradient_norm(&ad);
	printf("clip_test_noop grad_norm_before=%.12e grad_norm_after=%.12e clip_threshold=%.12e\n",
	    grad_norm_before_clip, grad_norm_after_clip, no_op_clip_threshold);
	if (fabs(grad_norm_after_clip - grad_norm_before_clip) > 1e-12) {
		fprintf(stderr, "error: no-op clipping changed gradient norm\n");
		free(final_logits);
		free(grad_a_before_clip);
		free(grad_b_before_clip);
		stage4_adapter_free(&ad);
		return 6;
	}

	memcpy(ad.grad_a, grad_a_before_clip, ad.a_count * sizeof(float));
	memcpy(ad.grad_b, grad_b_before_clip, ad.b_count * sizeof(float));
	stage4_adapter_clip_gradients(&ad, active_clip_threshold);
	grad_norm_after_clip = stage4_adapter_gradient_norm(&ad);
	printf("clip_test_active grad_norm_before=%.12e grad_norm_after=%.12e clip_threshold=%.12e\n",
	    grad_norm_before_clip, grad_norm_after_clip, active_clip_threshold);
	if (!(grad_norm_before_clip > active_clip_threshold)) {
		fprintf(stderr, "error: active clipping precondition failed\n");
		free(final_logits);
		free(grad_a_before_clip);
		free(grad_b_before_clip);
		stage4_adapter_free(&ad);
		return 6;
	}
	if (fabs(grad_norm_after_clip - active_clip_threshold) > 1e-6) {
		fprintf(stderr, "error: active clipping norm mismatch\n");
		free(final_logits);
		free(grad_a_before_clip);
		free(grad_b_before_clip);
		stage4_adapter_free(&ad);
		return 6;
	}
	if (!stage4_adapter_gradient_is_finite(&ad)) {
		fprintf(stderr, "error: active clipping produced non-finite gradients\n");
		free(final_logits);
		free(grad_a_before_clip);
		free(grad_b_before_clip);
		stage4_adapter_free(&ad);
		return 6;
	}
	{
		double dot = 0.0;
		double norm_before_sq = 0.0;
		double norm_after_sq = 0.0;
		double cosine;
		for (i = 0; i < ad.a_count; i++) {
			double gb = (double)grad_a_before_clip[i];
			double ga = (double)ad.grad_a[i];
			dot += gb * ga;
			norm_before_sq += gb * gb;
			norm_after_sq += ga * ga;
		}
		for (i = 0; i < ad.b_count; i++) {
			double gb = (double)grad_b_before_clip[i];
			double ga = (double)ad.grad_b[i];
			dot += gb * ga;
			norm_before_sq += gb * gb;
			norm_after_sq += ga * ga;
		}
		cosine = dot / (sqrt(norm_before_sq) * sqrt(norm_after_sq));
		printf("clip_test_active direction_cosine=%.12e\n", cosine);
		if (!isfinite(cosine) || cosine < 0.999999) {
			fprintf(stderr, "error: active clipping changed gradient direction\n");
			free(final_logits);
			free(grad_a_before_clip);
			free(grad_b_before_clip);
			stage4_adapter_free(&ad);
			return 6;
		}
	}

	update_norm = stage4_adapter_sgd_step(&ad, 0.05f);
	stage4_adapter_forward_logits(&ad, hidden, base_logits, final_logits);
	if (!stage4_softmax_loss_and_dlogits(final_logits, vocab, target,
	    &loss_after, ad.dlogits)) {
		free(final_logits);
		free(grad_a_before_clip);
		free(grad_b_before_clip);
		stage4_adapter_free(&ad);
		return 7;
	}
	printf("sgd_test loss_before=%.12f loss_after=%.12f update_norm=%.12e\n",
	    loss_before, loss_after, update_norm);
	if (!(loss_after < loss_before)) {
		fprintf(stderr, "error: SGD did not lower synthetic loss\n");
		free(final_logits);
		free(grad_a_before_clip);
		free(grad_b_before_clip);
		stage4_adapter_free(&ad);
		return 7;
	}

	if (!stage4_adapter_save_file("stories15M.adapter-gradient-test.bin",
	    &ad, &base_id)) {
		fprintf(stderr, "error: failed to save gradient test adapter\n");
		free(final_logits);
		free(grad_a_before_clip);
		free(grad_b_before_clip);
		stage4_adapter_free(&ad);
		return 8;
	}
	if (!stage4_hash_file_sha256_hex("stories15M.adapter-gradient-test.bin",
	    adapter_hash)) {
		fprintf(stderr, "error: failed to hash saved adapter\n");
		free(final_logits);
		free(grad_a_before_clip);
		free(grad_b_before_clip);
		stage4_adapter_free(&ad);
		return 8;
	}
	printf("saved_adapter path=stories15M.adapter-gradient-test.bin sha256=%s\n",
	    adapter_hash);

	if (!stage4_adapter_init(&reload, dim, vocab, rank, 1.0f)) {
		free(final_logits);
		free(grad_a_before_clip);
		free(grad_b_before_clip);
		stage4_adapter_free(&ad);
		return 8;
	}
	if (!stage4_adapter_load_file("stories15M.adapter-gradient-test.bin",
	    &reload, &base_id, 0)) {
		fprintf(stderr, "error: reload validation failed\n");
		free(final_logits);
		free(grad_a_before_clip);
		free(grad_b_before_clip);
		stage4_adapter_free(&ad);
		stage4_adapter_free(&reload);
		return 8;
	}
	stage4_adapter_forward_logits(&reload, hidden, base_logits, final_logits);
	if (!stage4_softmax_loss_and_dlogits(final_logits, vocab, target,
	    &loss_reload, reload.dlogits)) {
		free(final_logits);
		free(grad_a_before_clip);
		free(grad_b_before_clip);
		stage4_adapter_free(&ad);
		stage4_adapter_free(&reload);
		return 8;
	}
	printf("reload_test loss_before_save=%.12f loss_after_reload=%.12f abs_diff=%.12e\n",
	    loss_after, loss_reload, fabs(loss_after - loss_reload));
	if (fabs(loss_after - loss_reload) > loss_tol) {
		fprintf(stderr, "error: reload changed loss beyond tolerance\n");
		free(final_logits);
		free(grad_a_before_clip);
		free(grad_b_before_clip);
		stage4_adapter_free(&ad);
		stage4_adapter_free(&reload);
		return 8;
	}

	free(final_logits);
	free(grad_a_before_clip);
	free(grad_b_before_clip);
	stage4_adapter_free(&ad);
	stage4_adapter_free(&reload);
	printf("PASS: stage4_gradient_check\n");
	return 0;
}
