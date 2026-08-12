#include "stage4_adapter_train.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int minix_llama_embedded_main(int argc, char **argv);

#define main minix_llama_embedded_main
#include "../minix_llama.c"
#undef main

static int evaluate_one_pair(Transformer *t,
	Tokenizer *tok,
	stage4_adapter_t *ad,
	const char *record,
	int *target_out,
	double *loss_out,
	float *final_logits)
{
	int *tokens = NULL;
	int token_count = 0;
	const float *base_logits;
	int target;
	double loss;

	if (!encode_text(tok, record, 1, 0, &tokens, &token_count))
		return 0;
	if (token_count < 2 || token_count > t->config.seq_len) {
		free(tokens);
		return 0;
	}

	clear_run_state(t);
	base_logits = forward(t, tokens[0], 0);
	target = tokens[1];

	if (target < 0 || target >= t->config.vocab_size) {
		free(tokens);
		return 0;
	}

	stage4_adapter_forward_logits(ad, t->state.x, base_logits, final_logits);
	if (!stage4_softmax_loss_and_dlogits(final_logits, ad->vocab,
	    target, &loss, ad->dlogits)) {
		free(tokens);
		return 0;
	}

	*target_out = target;
	*loss_out = loss;
	free(tokens);
	return 1;
}

int main(int argc, char **argv)
{
	const char *checkpoint_path;
	const char *tokenizer_path = "/usr/src/minix_ai_stage3_stories15m/tokenizer.bin";
	const char *record = "Once upon a time there was a tiny fox in the garden.";
	const char *gradient_test_path = "stories15M.adapter-gradient-test.bin";
	Transformer transformer;
	Tokenizer tokenizer;
	stage4_adapter_t ad;
	stage4_adapter_t reload;
	stage4_base_identity_t base_id;
	char checkpoint_hash_before[65];
	char checkpoint_hash_after[65];
	char adapter_hash[65];
	uint64_t checkpoint_bytes;
	float *final_logits;
	double loss_before;
	double loss_after;
	double loss_reload;
	double grad_norm;
	double update_norm;
	double max_correction;
	int target_token;
	size_t i;
	int argi;

	if (argc < 2) {
		fprintf(stderr,
		    "Usage: %s <checkpoint.bin> [options]\n"
		    "Options:\n"
		    "  -z <tokenizer.bin>\n"
		    "  -r <record text>\n",
		    argv[0]);
		return 1;
	}
	checkpoint_path = argv[1];

	for (argi = 2; argi < argc; argi++) {
		if (strcmp(argv[argi], "-z") == 0 && argi + 1 < argc) {
			tokenizer_path = argv[++argi];
		} else if (strcmp(argv[argi], "-r") == 0 && argi + 1 < argc) {
			record = argv[++argi];
		} else {
			fprintf(stderr, "error: unknown option '%s'\n", argv[argi]);
			return 1;
		}
	}

	if (!stage4_hash_file_sha256_hex(checkpoint_path, checkpoint_hash_before)) {
		fprintf(stderr, "error: cannot hash checkpoint before\n");
		return 1;
	}
	if (!stage4_file_size_bytes(checkpoint_path, &checkpoint_bytes)) {
		fprintf(stderr, "error: cannot stat checkpoint\n");
		return 1;
	}
	printf("base_checkpoint_sha256_before=%s\n", checkpoint_hash_before);

	load_transformer(&transformer, checkpoint_path);
	load_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

	if (!stage4_adapter_init(&ad, transformer.config.dim,
	    transformer.config.vocab_size, 8, 1.0f)) {
		fprintf(stderr, "error: adapter init failed\n");
		free_tokenizer(&tokenizer);
		free_transformer(&transformer);
		return 1;
	}

	stage4_adapter_fill_a_deterministic(&ad, 1U, 0.01f);
	stage4_adapter_zero_b(&ad);
	stage4_adapter_zero_grad(&ad);

	final_logits = calloc((size_t)ad.vocab, sizeof(float));
	if (final_logits == NULL) {
		stage4_adapter_free(&ad);
		free_tokenizer(&tokenizer);
		free_transformer(&transformer);
		return 1;
	}

	/* Zero-B behavior in real-model context. */
	if (!evaluate_one_pair(&transformer, &tokenizer, &ad, record,
	    &target_token, &loss_before, final_logits)) {
		fprintf(stderr, "error: real-model pair evaluation failed\n");
		free(final_logits);
		stage4_adapter_free(&ad);
		free_tokenizer(&tokenizer);
		free_transformer(&transformer);
		return 1;
	}

	max_correction = stage4_adapter_max_logit_correction(transformer.state.logits,
	    final_logits, ad.vocab);
	stage4_adapter_backward(&ad, transformer.state.x, ad.dlogits);
	if (!stage4_adapter_gradient_is_finite(&ad)) {
		fprintf(stderr, "error: non-finite gradients\n");
		free(final_logits);
		stage4_adapter_free(&ad);
		free_tokenizer(&tokenizer);
		free_transformer(&transformer);
		return 1;
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
		printf("zero_b_real max_adapter_logit_correction=%.12e max_abs_grad_A=%.12e max_abs_grad_B=%.12e\n",
		    max_correction, max_abs_grad_a, max_abs_grad_b);
		if (max_correction > 1e-9 || max_abs_grad_b <= 1e-12 || max_abs_grad_a > 1e-9) {
			fprintf(stderr, "error: zero-B behavior check failed\n");
			free(final_logits);
			stage4_adapter_free(&ad);
			free_tokenizer(&tokenizer);
			free_transformer(&transformer);
			return 1;
		}
	}

	/* One deterministic SGD step. */
	grad_norm = stage4_adapter_gradient_norm(&ad);
	update_norm = stage4_adapter_sgd_step(&ad, 0.05f);

	if (!evaluate_one_pair(&transformer, &tokenizer, &ad, record,
	    &target_token, &loss_after, final_logits)) {
		fprintf(stderr, "error: real-model post-update evaluation failed\n");
		free(final_logits);
		stage4_adapter_free(&ad);
		free_tokenizer(&tokenizer);
		free_transformer(&transformer);
		return 1;
	}

	printf("one_token_train target_token=%d loss_before=%.12f loss_after=%.12f gradient_norm=%.12e update_norm=%.12e\n",
	    target_token, loss_before, loss_after, grad_norm, update_norm);
	if (!(isfinite(loss_before) && isfinite(loss_after) && loss_after < loss_before)) {
		fprintf(stderr, "error: one-token loss did not decrease\n");
		free(final_logits);
		stage4_adapter_free(&ad);
		free_tokenizer(&tokenizer);
		free_transformer(&transformer);
		return 1;
	}

	strcpy(base_id.base_sha256, checkpoint_hash_before);
	base_id.base_checkpoint_bytes = checkpoint_bytes;
	if (!stage4_adapter_save_file(gradient_test_path, &ad, &base_id)) {
		fprintf(stderr, "error: save gradient-test adapter failed\n");
		free(final_logits);
		stage4_adapter_free(&ad);
		free_tokenizer(&tokenizer);
		free_transformer(&transformer);
		return 1;
	}
	if (!stage4_hash_file_sha256_hex(gradient_test_path, adapter_hash)) {
		fprintf(stderr, "error: hash gradient-test adapter failed\n");
		free(final_logits);
		stage4_adapter_free(&ad);
		free_tokenizer(&tokenizer);
		free_transformer(&transformer);
		return 1;
	}
	printf("gradient_test_adapter path=%s sha256=%s\n", gradient_test_path, adapter_hash);

	if (!stage4_adapter_init(&reload, transformer.config.dim,
	    transformer.config.vocab_size, 8, 1.0f)) {
		free(final_logits);
		stage4_adapter_free(&ad);
		free_tokenizer(&tokenizer);
		free_transformer(&transformer);
		return 1;
	}
	if (!stage4_adapter_load_file(gradient_test_path, &reload, &base_id, 0)) {
		fprintf(stderr, "error: gradient-test adapter reload validation failed\n");
		free(final_logits);
		stage4_adapter_free(&reload);
		stage4_adapter_free(&ad);
		free_tokenizer(&tokenizer);
		free_transformer(&transformer);
		return 1;
	}
	if (!evaluate_one_pair(&transformer, &tokenizer, &reload, record,
	    &target_token, &loss_reload, final_logits)) {
		fprintf(stderr, "error: reload loss evaluation failed\n");
		free(final_logits);
		stage4_adapter_free(&reload);
		stage4_adapter_free(&ad);
		free_tokenizer(&tokenizer);
		free_transformer(&transformer);
		return 1;
	}
	printf("reload_loss_compare before_save=%.12f after_reload=%.12f abs_diff=%.12e\n",
	    loss_after, loss_reload, fabs(loss_after - loss_reload));
	if (fabs(loss_after - loss_reload) > 1e-7) {
		fprintf(stderr, "error: reload loss mismatch exceeds tolerance\n");
		free(final_logits);
		stage4_adapter_free(&reload);
		stage4_adapter_free(&ad);
		free_tokenizer(&tokenizer);
		free_transformer(&transformer);
		return 1;
	}

	if (!stage4_hash_file_sha256_hex(checkpoint_path, checkpoint_hash_after)) {
		fprintf(stderr, "error: cannot hash checkpoint after\n");
		free(final_logits);
		stage4_adapter_free(&reload);
		stage4_adapter_free(&ad);
		free_tokenizer(&tokenizer);
		free_transformer(&transformer);
		return 1;
	}
	printf("base_checkpoint_sha256_after=%s\n", checkpoint_hash_after);
	if (strcmp(checkpoint_hash_before, checkpoint_hash_after) != 0) {
		fprintf(stderr, "error: checkpoint hash changed\n");
		free(final_logits);
		stage4_adapter_free(&reload);
		stage4_adapter_free(&ad);
		free_tokenizer(&tokenizer);
		free_transformer(&transformer);
		return 1;
	}

	free(final_logits);
	stage4_adapter_free(&reload);
	stage4_adapter_free(&ad);
	free_tokenizer(&tokenizer);
	free_transformer(&transformer);
	printf("PASS: stage4_one_token_train\n");
	return 0;
}
