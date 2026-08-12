#include "stage4_adapter_train.h"

#include <ctype.h>
#include <errno.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int minix_llama_embedded_main(int argc, char **argv);

#define main minix_llama_embedded_main
#include "../minix_llama.c"
#undef main

#define STAGE4D_EXPECTED_BASE_SHA256 "cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a"
#define STAGE4D_CKPT_MAGIC "S4TK"
#define STAGE4D_CKPT_VERSION 1U
#define STAGE4D_MAX_TEXT 2048
#define STAGE4D_MAX_PATH 256
#define STAGE4D_MAX_GEN_TOKENS 64

typedef enum {
	STAGE4D_OPT_SGD = 0,
	STAGE4D_OPT_ADAM = 1
} stage4d_optimizer_t;

typedef struct {
	char id[128];
	char task[64];
	char split[64];
	double weight;
	char prompt[STAGE4D_MAX_TEXT];
	char target[STAGE4D_MAX_TEXT];
	char canonical_sequence[STAGE4D_MAX_TEXT * 2];
	char canonical_inference_prompt[STAGE4D_MAX_TEXT * 2];
	int *sequence_tokens;
	int sequence_count;
	int *prompt_tokens;
	int prompt_count;
	int first_target_index;
	int target_token_count;
} stage4d_record_t;

typedef struct {
	double target_loss_sum;
	double target_average_loss;
	double target_perplexity;
	int total_sequence_predictions;
	int masked_prompt_predictions;
	int target_prediction_count;
} stage4d_loss_stats_t;

typedef struct {
	char magic[4];
	uint32_t version;
	uint32_t optimizer;
	uint32_t rank;
	uint32_t dim;
	uint32_t vocab;
	uint32_t reserved0;
	uint64_t optimizer_step;
	double learning_rate;
	double beta1;
	double beta2;
	double epsilon;
	double weight_decay;
	double gradient_clip;
	double adapter_scale;
	double best_loss;
	double current_loss;
	uint64_t tokens_processed;
	uint64_t rng_state;
	uint64_t a_count;
	uint64_t b_count;
	uint64_t base_checkpoint_bytes;
	char base_checkpoint_sha256[65];
	char tokenizer_identity[65];
	char dataset_identity[65];
	uint64_t payload_checksum;
} stage4d_ckpt_header_t;

typedef struct {
	uint64_t optimizer_step;
	uint64_t tokens_processed;
	uint64_t rng_state;
	double best_loss;
	double current_loss;
} stage4d_runtime_state_t;

typedef struct {
	const char *checkpoint_path;
	const char *tokenizer_path;
	const char *adapter_path;
	const char *dataset_path;
	const char *train_ckpt_path;
	const char *resume_ckpt_path;
	int rank;
	float adapter_scale;
	stage4d_optimizer_t optimizer;
	double learning_rate;
	double beta1;
	double beta2;
	double epsilon;
	double weight_decay;
	double gradient_clip;
	int steps;
	int eval_every;
	uint64_t seed;
	double target_loss_threshold;
	int stop_on_exact_match;
	int max_gen_tokens;
	int run_resume_check;
} stage4d_options_t;

static void stage4d_usage(const char *prog)
{
	fprintf(stderr,
	    "Usage: %s <checkpoint.bin> [options]\n"
	    "Options:\n"
	    "  -z <tokenizer.bin>\n"
	    "  -a <adapter-overfit.bin>\n"
	    "  -d <records file>\n"
	    "  -r <rank>\n"
	    "  -y <adapter scale>\n"
	    "  --optimizer <adam|sgd>\n"
	    "  --learning-rate <value>\n"
	    "  --beta1 <value>\n"
	    "  --beta2 <value>\n"
	    "  --epsilon <value>\n"
	    "  --weight-decay <value>\n"
	    "  --gradient-clip <value>\n"
	    "  --steps <count>\n"
	    "  --eval-every <count>\n"
	    "  --seed <value>\n"
	    "  --target-loss-threshold <value>\n"
	    "  --stop-on-exact-match <0|1>\n"
	    "  --max-gen-tokens <count>\n"
	    "  --train-checkpoint <path>\n"
	    "  --resume-checkpoint <path>\n"
	    "  --resume-check <0|1>\n",
	    prog);
}

static int stage4d_streq(const char *a, const char *b)
{
	return strcmp(a, b) == 0;
}

static int stage4d_parse_u64(const char *text, uint64_t *out)
{
	char *end = NULL;
	unsigned long long v;

	errno = 0;
	v = strtoull(text, &end, 10);
	if (errno != 0 || end == text || *end != '\0')
		return 0;
	*out = (uint64_t)v;
	return 1;
}

static int stage4d_parse_int(const char *text, int *out)
{
	char *end = NULL;
	long v;

	errno = 0;
	v = strtol(text, &end, 10);
	if (errno != 0 || end == text || *end != '\0')
		return 0;
	if (v < INT_MIN || v > INT_MAX)
		return 0;
	*out = (int)v;
	return 1;
}

static int stage4d_parse_double(const char *text, double *out)
{
	char *end = NULL;
	double v;

	errno = 0;
	v = strtod(text, &end);
	if (errno != 0 || end == text || *end != '\0' || !isfinite(v))
		return 0;
	*out = v;
	return 1;
}

static int stage4d_parse_bool01(const char *text, int *out)
{
	if (strcmp(text, "0") == 0) {
		*out = 0;
		return 1;
	}
	if (strcmp(text, "1") == 0) {
		*out = 1;
		return 1;
	}
	return 0;
}

static int stage4d_read_file_text(const char *path, char **out_text)
{
	FILE *f;
	long sz;
	char *buf;

	f = fopen(path, "rb");
	if (f == NULL)
		return 0;
	if (fseek(f, 0, SEEK_END) != 0) {
		fclose(f);
		return 0;
	}
	sz = ftell(f);
	if (sz < 0) {
		fclose(f);
		return 0;
	}
	if (fseek(f, 0, SEEK_SET) != 0) {
		fclose(f);
		return 0;
	}
	buf = calloc((size_t)sz + 1U, 1);
	if (buf == NULL) {
		fclose(f);
		return 0;
	}
	if (sz > 0 && fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
		free(buf);
		fclose(f);
		return 0;
	}
	if (fclose(f) != 0) {
		free(buf);
		return 0;
	}
	buf[(size_t)sz] = '\0';
	*out_text = buf;
	return 1;
}

static void stage4d_trim_trailing_whitespace(char *text)
{
	size_t n;

	if (text == NULL)
		return;
	n = strlen(text);
	while (n > 0) {
		unsigned char c = (unsigned char)text[n - 1];
		if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
			text[n - 1] = '\0';
			n--;
		} else {
			break;
		}
	}
}

static int stage4d_extract_section(const char *src,
	const char *begin_marker,
	const char *end_marker,
	char *out,
	size_t out_cap)
{
	const char *begin;
	const char *end;
	size_t len;

	begin = strstr(src, begin_marker);
	if (begin == NULL)
		return 0;
	begin += strlen(begin_marker);
	if (*begin == '\r')
		begin++;
	if (*begin == '\n')
		begin++;
	end = strstr(begin, end_marker);
	if (end == NULL)
		return 0;
	len = (size_t)(end - begin);
	if (len + 1U > out_cap)
		return 0;
	memcpy(out, begin, len);
	out[len] = '\0';
	stage4d_trim_trailing_whitespace(out);
	return 1;
}

static int stage4d_extract_key_value(const char *src,
	const char *key,
	char *out,
	size_t out_cap)
{
	const char *p;
	size_t key_len;
	const char *line_end;
	size_t len;

	key_len = strlen(key);
	p = src;
	while (*p != '\0') {
		if (strncmp(p, key, key_len) == 0 && p[key_len] == '=') {
			const char *value = p + key_len + 1;
			line_end = strchr(value, '\n');
			if (line_end == NULL)
				line_end = value + strlen(value);
			len = (size_t)(line_end - value);
			if (len + 1U > out_cap)
				return 0;
			memcpy(out, value, len);
			out[len] = '\0';
			stage4d_trim_trailing_whitespace(out);
			return 1;
		}
		p = strchr(p, '\n');
		if (p == NULL)
			break;
		p++;
	}
	return 0;
}

static int stage4d_load_record(const char *path, stage4d_record_t *rec)
{
	char *text = NULL;
	char weight_buf[64];
	double weight;

	memset(rec, 0, sizeof(*rec));
	if (!stage4d_read_file_text(path, &text)) {
		fprintf(stderr, "error: cannot read records file '%s'\n", path);
		return 0;
	}

	if (strstr(text, "%%BEGIN") == NULL || strstr(text, "%%END") == NULL) {
		fprintf(stderr, "error: malformed record markers\n");
		free(text);
		return 0;
	}
	if (!stage4d_extract_key_value(text, "id", rec->id, sizeof(rec->id)) ||
	    !stage4d_extract_key_value(text, "task", rec->task, sizeof(rec->task)) ||
	    !stage4d_extract_key_value(text, "split", rec->split, sizeof(rec->split)) ||
	    !stage4d_extract_key_value(text, "weight", weight_buf, sizeof(weight_buf)) ||
	    !stage4d_extract_section(text, "%%PROMPT", "%%TARGET",
	    rec->prompt, sizeof(rec->prompt)) ||
	    !stage4d_extract_section(text, "%%TARGET", "%%END",
	    rec->target, sizeof(rec->target))) {
		fprintf(stderr, "error: malformed prompt/target record\n");
		free(text);
		return 0;
	}
	if (!stage4d_parse_double(weight_buf, &weight) || !(weight > 0.0)) {
		fprintf(stderr, "error: invalid record weight\n");
		free(text);
		return 0;
	}
	rec->weight = weight;
	if (rec->prompt[0] == '\0' || rec->target[0] == '\0') {
		fprintf(stderr, "error: empty target or prompt\n");
		free(text);
		return 0;
	}

	if (snprintf(rec->canonical_sequence, sizeof(rec->canonical_sequence),
	    "User: %s\nAssistant: %s", rec->prompt, rec->target) >=
	    (int)sizeof(rec->canonical_sequence)) {
		fprintf(stderr, "error: canonical sequence too long\n");
		free(text);
		return 0;
	}
	if (snprintf(rec->canonical_inference_prompt,
	    sizeof(rec->canonical_inference_prompt),
	    "User: %s\nAssistant:", rec->prompt) >=
	    (int)sizeof(rec->canonical_inference_prompt)) {
		fprintf(stderr, "error: canonical prompt too long\n");
		free(text);
		return 0;
	}

	if (strcmp(rec->canonical_sequence,
	    "User: MINIX service status command:\nAssistant: service status") != 0 ||
	    strcmp(rec->canonical_inference_prompt,
	    "User: MINIX service status command:\nAssistant:") != 0) {
		fprintf(stderr,
		    "error: canonical instruction representation mismatch\n"
		    "expected sequence: User: MINIX service status command:\\nAssistant: service status\n"
		    "expected prompt:   User: MINIX service status command:\\nAssistant:\n");
		free(text);
		return 0;
	}

	free(text);
	return 1;
}

static int stage4d_prepare_tokens(Tokenizer *tok,
	Transformer *tr,
	stage4d_record_t *rec)
{
	int i;

	if (!encode_text(tok, rec->canonical_sequence, 1, 0,
	    &rec->sequence_tokens, &rec->sequence_count)) {
		fprintf(stderr, "error: cannot tokenize canonical sequence\n");
		return 0;
	}
	if (!encode_text(tok, rec->canonical_inference_prompt, 1, 0,
	    &rec->prompt_tokens, &rec->prompt_count)) {
		fprintf(stderr, "error: cannot tokenize canonical prompt\n");
		return 0;
	}

	if (rec->sequence_count <= 1) {
		fprintf(stderr, "error: empty target sequence\n");
		return 0;
	}
	if (rec->sequence_count > tr->config.seq_len) {
		fprintf(stderr, "error: context overflow sequence_count=%d seq_len=%d\n",
		    rec->sequence_count, tr->config.seq_len);
		return 0;
	}

	rec->first_target_index = rec->prompt_count;
	if (rec->first_target_index <= 0 ||
	    rec->first_target_index >= rec->sequence_count) {
		fprintf(stderr, "error: target boundary outside sequence\n");
		return 0;
	}
	rec->target_token_count = rec->sequence_count - rec->first_target_index;
	if (rec->target_token_count <= 0) {
		fprintf(stderr, "error: no trainable prediction positions\n");
		return 0;
	}

	printf("canonical_sequence=%s\n", rec->canonical_sequence);
	printf("canonical_inference_prompt=%s\n", rec->canonical_inference_prompt);
	printf("canonical_prompt_bytes_hex:");
	for (i = 0; rec->canonical_inference_prompt[i] != '\0'; i++)
		printf(" %02x", (unsigned char)rec->canonical_inference_prompt[i]);
	printf("\n");

	printf("sequence_token_count=%d\n", rec->sequence_count);
	printf("prompt_token_count=%d\n", rec->prompt_count);
	printf("target_token_count=%d\n", rec->target_token_count);
	printf("first_target_token_index=%d\n", rec->first_target_index);

	printf("sequence_token_ids:");
	for (i = 0; i < rec->sequence_count; i++)
		printf(" %d", rec->sequence_tokens[i]);
	printf("\n");
	printf("prompt_token_ids:");
	for (i = 0; i < rec->prompt_count; i++)
		printf(" %d", rec->prompt_tokens[i]);
	printf("\n");
	printf("target_prediction_positions:");
	for (i = rec->first_target_index; i < rec->sequence_count; i++)
		printf(" %d(input=%d->target=%d)", i, i - 1, i);
	printf("\n");
	return 1;
}

static void stage4d_free_record(stage4d_record_t *rec)
{
	free(rec->sequence_tokens);
	free(rec->prompt_tokens);
	memset(rec, 0, sizeof(*rec));
}

static int stage4d_eval_record(Transformer *tr,
	stage4_adapter_t *ad,
	const stage4d_record_t *rec,
	int accumulate_grad,
	stage4d_loss_stats_t *stats)
{
	float *final_logits;
	int k;
	int target_preds = 0;
	double loss_sum = 0.0;

	memset(stats, 0, sizeof(*stats));
	stats->total_sequence_predictions = rec->sequence_count - 1;
	stats->masked_prompt_predictions = rec->first_target_index - 1;

	final_logits = calloc((size_t)tr->config.vocab_size, sizeof(float));
	if (final_logits == NULL)
		return 0;

	clear_run_state(tr);
	if (accumulate_grad)
		stage4_adapter_zero_grad(ad);

	for (k = 1; k < rec->sequence_count; k++) {
		const float *base_logits = forward(tr, rec->sequence_tokens[k - 1], k - 1);
		int target = rec->sequence_tokens[k];
		double loss;

		if (k < rec->first_target_index)
			continue;

		stage4_adapter_forward_logits(ad, tr->state.x, base_logits, final_logits);
		if (!stage4_softmax_loss_and_dlogits(final_logits, ad->vocab, target,
		    &loss, ad->dlogits)) {
			free(final_logits);
			return 0;
		}
		if (!isfinite(loss)) {
			free(final_logits);
			return 0;
		}
		if (accumulate_grad)
			stage4_adapter_backward(ad, tr->state.x, ad->dlogits);
		loss_sum += loss;
		target_preds++;
	}

	free(final_logits);
	if (target_preds <= 0)
		return 0;

	if (accumulate_grad)
		stage4_adapter_scale_gradients(ad, 1.0f / (float)target_preds);
	stats->target_loss_sum = loss_sum;
	stats->target_prediction_count = target_preds;
	stats->target_average_loss = loss_sum / (double)target_preds;
	stats->target_perplexity = exp(stats->target_average_loss);
	return isfinite(stats->target_average_loss) && isfinite(stats->target_perplexity);
}

static void stage4d_print_masking_stats(const stage4d_loss_stats_t *stats)
{
	printf("total_sequence_predictions=%d\n", stats->total_sequence_predictions);
	printf("masked_prompt_predictions=%d\n", stats->masked_prompt_predictions);
	printf("trained_target_predictions=%d\n", stats->target_prediction_count);
}

static int stage4d_generate_greedy(Transformer *tr,
	Tokenizer *tok,
	const char *prompt,
	int max_new_tokens,
	const stage4_adapter_t *ad,
	int use_adapter,
	int *out_ids,
	int *out_count,
	char *out_text,
	size_t out_text_cap)
{
	int *prompt_tokens = NULL;
	int prompt_count = 0;
	int token;
	int next = 0;
	int position = 0;
	int generated = 0;
	float *final_logits;
	size_t text_len = 0;

	if (!encode_text(tok, prompt, 1, 0, &prompt_tokens, &prompt_count))
		return 0;
	if (prompt_count < 1 || prompt_count > tr->config.seq_len) {
		free(prompt_tokens);
		return 0;
	}

	final_logits = calloc((size_t)tr->config.vocab_size, sizeof(float));
	if (final_logits == NULL) {
		free(prompt_tokens);
		return 0;
	}

	clear_run_state(tr);
	token = prompt_tokens[0];
	out_text[0] = '\0';

	while (position < tr->config.seq_len && generated < max_new_tokens) {
		const float *logits = forward(tr, token, position);
		if (position < prompt_count - 1) {
			next = prompt_tokens[position + 1];
		} else {
			if (use_adapter) {
				stage4_adapter_forward_logits(ad, tr->state.x, logits, final_logits);
				next = sample_argmax(final_logits, tr->config.vocab_size);
			} else {
				next = sample_argmax(logits, tr->config.vocab_size);
			}
			if (out_ids != NULL)
				out_ids[generated] = next;
			generated++;
			if (next == 1)
				break;
			{
				char *piece = decode_piece(tok, token, next);
				size_t piece_len = strlen(piece);
				if (text_len + piece_len + 1U < out_text_cap) {
					memcpy(out_text + text_len, piece, piece_len);
					text_len += piece_len;
					out_text[text_len] = '\0';
				}
			}
		}
		position++;
		token = next;
	}

	*out_count = generated;
	free(final_logits);
	free(prompt_tokens);
	return 1;
}

static int stage4d_exact_target_token_match(const stage4d_record_t *rec,
	const int *generated_ids,
	int generated_count)
{
	int i;

	if (rec == NULL || generated_ids == NULL)
		return 0;
	if (generated_count < rec->target_token_count)
		return 0;
	for (i = 0; i < rec->target_token_count; i++) {
		int expected = rec->sequence_tokens[rec->first_target_index + i];
		if (generated_ids[i] != expected)
			return 0;
	}
	return 1;
}

static uint64_t stage4d_checksum_u64_update(uint64_t h, const void *data, size_t n)
{
	const unsigned char *p = (const unsigned char *)data;
	size_t i;
	for (i = 0; i < n; i++) {
		h ^= (uint64_t)p[i];
		h *= UINT64_C(1099511628211);
	}
	return h;
}

static uint64_t stage4d_checkpoint_payload_checksum(const stage4_adapter_t *ad)
{
	uint64_t h = UINT64_C(1469598103934665603);
	h = stage4d_checksum_u64_update(h, ad->a, ad->a_count * sizeof(float));
	h = stage4d_checksum_u64_update(h, ad->b, ad->b_count * sizeof(float));
	h = stage4d_checksum_u64_update(h, ad->m1_a, ad->a_count * sizeof(float));
	h = stage4d_checksum_u64_update(h, ad->m1_b, ad->b_count * sizeof(float));
	h = stage4d_checksum_u64_update(h, ad->m2_a, ad->a_count * sizeof(float));
	h = stage4d_checksum_u64_update(h, ad->m2_b, ad->b_count * sizeof(float));
	return h;
}

static int stage4d_validate_checkpoint_file(const char *path,
	const stage4_adapter_t *shape_like,
	const char *base_sha256,
	const char *tokenizer_identity,
	const char *dataset_identity,
	stage4d_ckpt_header_t *header_out)
{
	FILE *f;
	stage4d_ckpt_header_t hdr;
	stage4_adapter_t tmp;
	uint64_t checksum;
	size_t i;

	memset(&tmp, 0, sizeof(tmp));
	f = fopen(path, "rb");
	if (f == NULL)
		return 0;
	if (fread(&hdr, sizeof(hdr), 1, f) != 1) {
		fclose(f);
		return 0;
	}
	if (memcmp(hdr.magic, STAGE4D_CKPT_MAGIC, 4) != 0 ||
	    hdr.version != STAGE4D_CKPT_VERSION) {
		fclose(f);
		return 0;
	}
	if (strcmp(hdr.base_checkpoint_sha256, base_sha256) != 0 ||
	    strcmp(hdr.tokenizer_identity, tokenizer_identity) != 0 ||
	    strcmp(hdr.dataset_identity, dataset_identity) != 0) {
		fclose(f);
		return 0;
	}
	if (hdr.a_count != (uint64_t)shape_like->a_count ||
	    hdr.b_count != (uint64_t)shape_like->b_count ||
	    hdr.rank != (uint32_t)shape_like->rank ||
	    hdr.dim != (uint32_t)shape_like->dim ||
	    hdr.vocab != (uint32_t)shape_like->vocab) {
		fclose(f);
		return 0;
	}

	if (!stage4_adapter_init(&tmp, shape_like->dim, shape_like->vocab,
	    shape_like->rank, shape_like->scale)) {
		fclose(f);
		return 0;
	}
	if (fread(tmp.a, sizeof(float), tmp.a_count, f) != tmp.a_count ||
	    fread(tmp.b, sizeof(float), tmp.b_count, f) != tmp.b_count ||
	    fread(tmp.m1_a, sizeof(float), tmp.a_count, f) != tmp.a_count ||
	    fread(tmp.m1_b, sizeof(float), tmp.b_count, f) != tmp.b_count ||
	    fread(tmp.m2_a, sizeof(float), tmp.a_count, f) != tmp.a_count ||
	    fread(tmp.m2_b, sizeof(float), tmp.b_count, f) != tmp.b_count) {
		stage4_adapter_free(&tmp);
		fclose(f);
		return 0;
	}
	if (fgetc(f) != EOF) {
		stage4_adapter_free(&tmp);
		fclose(f);
		return 0;
	}
	for (i = 0; i < tmp.a_count; i++) {
		if (!isfinite(tmp.a[i]) || !isfinite(tmp.m1_a[i]) || !isfinite(tmp.m2_a[i])) {
			stage4_adapter_free(&tmp);
			fclose(f);
			return 0;
		}
	}
	for (i = 0; i < tmp.b_count; i++) {
		if (!isfinite(tmp.b[i]) || !isfinite(tmp.m1_b[i]) || !isfinite(tmp.m2_b[i])) {
			stage4_adapter_free(&tmp);
			fclose(f);
			return 0;
		}
	}
	checksum = stage4d_checkpoint_payload_checksum(&tmp);
	stage4_adapter_free(&tmp);
	if (checksum != hdr.payload_checksum) {
		fclose(f);
		return 0;
	}
	if (fclose(f) != 0)
		return 0;
	if (header_out != NULL)
		*header_out = hdr;
	return 1;
}

static int stage4d_save_training_checkpoint_atomic(const char *path,
	const stage4_adapter_t *ad,
	const stage4d_options_t *opt,
	const stage4d_runtime_state_t *state,
	const char *base_sha256,
	uint64_t base_bytes,
	const char *tokenizer_identity,
	const char *dataset_identity)
{
	char tmp_path[STAGE4D_MAX_PATH * 2];
	FILE *f;
	stage4d_ckpt_header_t hdr;

	if (snprintf(tmp_path, sizeof(tmp_path), "%s.tmp", path) >= (int)sizeof(tmp_path))
		return 0;

	memset(&hdr, 0, sizeof(hdr));
	memcpy(hdr.magic, STAGE4D_CKPT_MAGIC, 4);
	hdr.version = STAGE4D_CKPT_VERSION;
	hdr.optimizer = (uint32_t)opt->optimizer;
	hdr.rank = (uint32_t)ad->rank;
	hdr.dim = (uint32_t)ad->dim;
	hdr.vocab = (uint32_t)ad->vocab;
	hdr.optimizer_step = state->optimizer_step;
	hdr.learning_rate = opt->learning_rate;
	hdr.beta1 = opt->beta1;
	hdr.beta2 = opt->beta2;
	hdr.epsilon = opt->epsilon;
	hdr.weight_decay = opt->weight_decay;
	hdr.gradient_clip = opt->gradient_clip;
	hdr.adapter_scale = opt->adapter_scale;
	hdr.best_loss = state->best_loss;
	hdr.current_loss = state->current_loss;
	hdr.tokens_processed = state->tokens_processed;
	hdr.rng_state = state->rng_state;
	hdr.a_count = (uint64_t)ad->a_count;
	hdr.b_count = (uint64_t)ad->b_count;
	hdr.base_checkpoint_bytes = base_bytes;
	strncpy(hdr.base_checkpoint_sha256, base_sha256,
	    sizeof(hdr.base_checkpoint_sha256) - 1);
	strncpy(hdr.tokenizer_identity, tokenizer_identity,
	    sizeof(hdr.tokenizer_identity) - 1);
	strncpy(hdr.dataset_identity, dataset_identity,
	    sizeof(hdr.dataset_identity) - 1);
	hdr.payload_checksum = stage4d_checkpoint_payload_checksum(ad);

	f = fopen(tmp_path, "wb");
	if (f == NULL)
		return 0;
	if (fwrite(&hdr, sizeof(hdr), 1, f) != 1 ||
	    fwrite(ad->a, sizeof(float), ad->a_count, f) != ad->a_count ||
	    fwrite(ad->b, sizeof(float), ad->b_count, f) != ad->b_count ||
	    fwrite(ad->m1_a, sizeof(float), ad->a_count, f) != ad->a_count ||
	    fwrite(ad->m1_b, sizeof(float), ad->b_count, f) != ad->b_count ||
	    fwrite(ad->m2_a, sizeof(float), ad->a_count, f) != ad->a_count ||
	    fwrite(ad->m2_b, sizeof(float), ad->b_count, f) != ad->b_count) {
		fclose(f);
		remove(tmp_path);
		return 0;
	}
	if (fflush(f) != 0 || fclose(f) != 0) {
		remove(tmp_path);
		return 0;
	}

	if (!stage4d_validate_checkpoint_file(tmp_path, ad, base_sha256,
	    tokenizer_identity, dataset_identity, NULL)) {
		remove(tmp_path);
		return 0;
	}

	if (rename(tmp_path, path) != 0) {
		remove(tmp_path);
		return 0;
	}
	return 1;
}

static int stage4d_load_training_checkpoint(const char *path,
	stage4_adapter_t *ad,
	const stage4d_options_t *opt,
	stage4d_runtime_state_t *state,
	const char *base_sha256,
	const char *tokenizer_identity,
	const char *dataset_identity)
{
	FILE *f;
	stage4d_ckpt_header_t hdr;

	if (!stage4d_validate_checkpoint_file(path, ad, base_sha256,
	    tokenizer_identity, dataset_identity, &hdr))
		return 0;

	if ((stage4d_optimizer_t)hdr.optimizer != opt->optimizer) {
		fprintf(stderr, "error: checkpoint optimizer mismatch\n");
		return 0;
	}

	f = fopen(path, "rb");
	if (f == NULL)
		return 0;
	if (fread(&hdr, sizeof(hdr), 1, f) != 1 ||
	    fread(ad->a, sizeof(float), ad->a_count, f) != ad->a_count ||
	    fread(ad->b, sizeof(float), ad->b_count, f) != ad->b_count ||
	    fread(ad->m1_a, sizeof(float), ad->a_count, f) != ad->a_count ||
	    fread(ad->m1_b, sizeof(float), ad->b_count, f) != ad->b_count ||
	    fread(ad->m2_a, sizeof(float), ad->a_count, f) != ad->a_count ||
	    fread(ad->m2_b, sizeof(float), ad->b_count, f) != ad->b_count) {
		fclose(f);
		return 0;
	}
	if (fclose(f) != 0)
		return 0;

	state->optimizer_step = hdr.optimizer_step;
	state->tokens_processed = hdr.tokens_processed;
	state->rng_state = hdr.rng_state;
	state->best_loss = hdr.best_loss;
	state->current_loss = hdr.current_loss;
	return 1;
}

static double stage4d_max_abs_diff(const float *a, const float *b, size_t n)
{
	double m = 0.0;
	size_t i;
	for (i = 0; i < n; i++) {
		double d = fabs((double)a[i] - (double)b[i]);
		if (d > m)
			m = d;
	}
	return m;
}

static int stage4d_train_steps(Transformer *tr,
	Tokenizer *tok,
	stage4_adapter_t *ad,
	const stage4d_record_t *rec,
	const stage4d_options_t *opt,
	stage4d_runtime_state_t *state,
	int start_step,
	int end_step,
	double *start_loss_out,
	double *final_loss_out,
	double *best_loss_out,
	int *exact_match_out)
{
	double start_wall = monotonic_seconds();
	double last_loss = NAN;
	int step;
	char generation_text[2048];
	int generation_ids[STAGE4D_MAX_GEN_TOKENS];
	int generation_count = 0;
	int exact_match = 0;

	for (step = start_step; step <= end_step; step++) {
		stage4d_loss_stats_t train_stats;
		double grad_norm_before;
		double grad_norm_after;
		double update_norm;
		double elapsed;

		if (!stage4d_eval_record(tr, ad, rec, 1, &train_stats)) {
			fprintf(stderr, "error: training forward/backward failed at step %d\n", step);
			return 0;
		}
		if (step == start_step)
			*start_loss_out = train_stats.target_average_loss;
		if (!isfinite(train_stats.target_average_loss)) {
			fprintf(stderr, "error: non-finite loss at step %d\n", step);
			return 0;
		}

		grad_norm_before = stage4_adapter_gradient_norm(ad);
		if (opt->gradient_clip > 0.0)
			stage4_adapter_clip_gradients(ad, opt->gradient_clip);
		grad_norm_after = stage4_adapter_gradient_norm(ad);

		state->optimizer_step = (uint64_t)step;
		if (opt->optimizer == STAGE4D_OPT_ADAM) {
			update_norm = stage4_adapter_adam_step(ad,
			    (float)opt->learning_rate,
			    (float)opt->beta1,
			    (float)opt->beta2,
			    (float)opt->epsilon,
			    (float)opt->weight_decay,
			    state->optimizer_step);
		} else {
			update_norm = stage4_adapter_sgd_step(ad, (float)opt->learning_rate);
		}
		if (!stage4_adapter_parameters_are_finite(ad)) {
			fprintf(stderr, "error: non-finite parameter at step %d\n", step);
			return 0;
		}

		state->tokens_processed += (uint64_t)train_stats.target_prediction_count;
		state->current_loss = train_stats.target_average_loss;
		if (step == start_step || state->current_loss < state->best_loss)
			state->best_loss = state->current_loss;
		last_loss = state->current_loss;

		if (step == start_step || step == end_step ||
		    (opt->eval_every > 0 && (step % opt->eval_every) == 0)) {
			stage4d_loss_stats_t eval_stats;
			if (!stage4d_eval_record(tr, ad, rec, 0, &eval_stats)) {
				fprintf(stderr, "error: eval failed at step %d\n", step);
				return 0;
			}
			elapsed = monotonic_seconds() - start_wall;
			printf("step=%d target_loss_sum=%.12f target_prediction_count=%d target_average_loss=%.12f target_perplexity=%.12f gradient_norm_before_clip=%.12e gradient_norm_after_clip=%.12e update_norm=%.12e elapsed_seconds=%.6f\n",
			    step,
			    eval_stats.target_loss_sum,
			    eval_stats.target_prediction_count,
			    eval_stats.target_average_loss,
			    eval_stats.target_perplexity,
			    grad_norm_before,
			    grad_norm_after,
			    update_norm,
			    elapsed);
		}

		if (opt->target_loss_threshold > 0.0 &&
		    state->current_loss <= opt->target_loss_threshold) {
			printf("stop_condition=target_loss_threshold reached step=%d value=%.12f\n",
			    step, state->current_loss);
			break;
		}

		if (opt->stop_on_exact_match) {
			if (!stage4d_generate_greedy(tr, tok,
			    rec->canonical_inference_prompt,
			    rec->target_token_count,
			    ad, 1,
			    generation_ids,
			    &generation_count,
			    generation_text,
			    sizeof(generation_text))) {
				fprintf(stderr, "error: generation failed for exact-match check\n");
				return 0;
			}
			exact_match = stage4d_exact_target_token_match(rec,
			    generation_ids, generation_count);
			if (exact_match) {
				printf("stop_condition=exact_target_match step=%d\n", step);
				break;
			}
		}
	}

	*final_loss_out = last_loss;
	*best_loss_out = state->best_loss;
	*exact_match_out = exact_match;
	return 1;
}

static int stage4d_run_resume_determinism_check(Transformer *tr,
	Tokenizer *tok,
	const stage4d_record_t *rec,
	const stage4d_options_t *opt,
	const char *base_sha,
	uint64_t base_bytes,
	const char *tok_sha,
	const char *data_sha)
{
	stage4_adapter_t ad_cont;
	stage4_adapter_t ad_split_a;
	stage4_adapter_t ad_split_b;
	stage4d_runtime_state_t st_cont;
	stage4d_runtime_state_t st_split;
	double cont_start = NAN;
	double cont_final = NAN;
	double cont_best = NAN;
	double split_start = NAN;
	double split_final = NAN;
	double split_best = NAN;
	int cont_exact = 0;
	int split_exact = 0;
	int compare_steps;
	int split_step;
	double diff_a;
	double diff_b;
	double diff_m1a;
	double diff_m1b;
	double diff_m2a;
	double diff_m2b;
	double diff_loss;
	double max_diff;
	int ids_cont[STAGE4D_MAX_GEN_TOKENS];
	int ids_split[STAGE4D_MAX_GEN_TOKENS];
	int n_cont = 0;
	int n_split = 0;
	char text_cont[2048];
	char text_split[2048];
	int token_match = 0;

	memset(&ad_cont, 0, sizeof(ad_cont));
	memset(&ad_split_a, 0, sizeof(ad_split_a));
	memset(&ad_split_b, 0, sizeof(ad_split_b));
	memset(&st_cont, 0, sizeof(st_cont));
	memset(&st_split, 0, sizeof(st_split));

	compare_steps = opt->steps >= 100 ? 100 : opt->steps;
	if (compare_steps < 2)
		return 1;
	split_step = compare_steps / 2;

	if (!stage4_adapter_init(&ad_cont, tr->config.dim, tr->config.vocab_size,
	    opt->rank, opt->adapter_scale) ||
	    !stage4_adapter_init(&ad_split_a, tr->config.dim, tr->config.vocab_size,
	    opt->rank, opt->adapter_scale) ||
	    !stage4_adapter_init(&ad_split_b, tr->config.dim, tr->config.vocab_size,
	    opt->rank, opt->adapter_scale)) {
		fprintf(stderr, "error: resume-check adapter init failed\n");
		goto fail;
	}
	stage4_adapter_fill_a_deterministic(&ad_cont, opt->seed, 0.01f);
	stage4_adapter_zero_b(&ad_cont);
	stage4_adapter_zero_moments(&ad_cont);

	memcpy(ad_split_a.a, ad_cont.a, ad_cont.a_count * sizeof(float));
	memcpy(ad_split_a.b, ad_cont.b, ad_cont.b_count * sizeof(float));
	stage4_adapter_zero_moments(&ad_split_a);

	st_cont.best_loss = DBL_MAX;
	st_split.best_loss = DBL_MAX;

	if (!stage4d_train_steps(tr, tok, &ad_cont, rec, opt, &st_cont,
	    1, compare_steps, &cont_start, &cont_final, &cont_best, &cont_exact))
		goto fail;

	if (!stage4d_train_steps(tr, tok, &ad_split_a, rec, opt, &st_split,
	    1, split_step, &split_start, &split_final, &split_best, &split_exact))
		goto fail;

	if (!stage4d_save_training_checkpoint_atomic(opt->resume_ckpt_path,
	    &ad_split_a, opt, &st_split, base_sha, base_bytes, tok_sha, data_sha)) {
		fprintf(stderr, "error: resume-check save checkpoint failed\n");
		goto fail;
	}

	memcpy(ad_split_b.a, ad_split_a.a, ad_split_a.a_count * sizeof(float));
	memcpy(ad_split_b.b, ad_split_a.b, ad_split_a.b_count * sizeof(float));
	memcpy(ad_split_b.m1_a, ad_split_a.m1_a, ad_split_a.a_count * sizeof(float));
	memcpy(ad_split_b.m1_b, ad_split_a.m1_b, ad_split_a.b_count * sizeof(float));
	memcpy(ad_split_b.m2_a, ad_split_a.m2_a, ad_split_a.a_count * sizeof(float));
	memcpy(ad_split_b.m2_b, ad_split_a.m2_b, ad_split_a.b_count * sizeof(float));

	if (!stage4d_load_training_checkpoint(opt->resume_ckpt_path,
	    &ad_split_b, opt, &st_split, base_sha, tok_sha, data_sha)) {
		fprintf(stderr, "error: resume-check load checkpoint failed\n");
		goto fail;
	}

	if (!stage4d_train_steps(tr, tok, &ad_split_b, rec, opt, &st_split,
	    split_step + 1, compare_steps,
	    &split_start, &split_final, &split_best, &split_exact))
		goto fail;

	diff_a = stage4d_max_abs_diff(ad_cont.a, ad_split_b.a, ad_cont.a_count);
	diff_b = stage4d_max_abs_diff(ad_cont.b, ad_split_b.b, ad_cont.b_count);
	diff_m1a = stage4d_max_abs_diff(ad_cont.m1_a, ad_split_b.m1_a, ad_cont.a_count);
	diff_m1b = stage4d_max_abs_diff(ad_cont.m1_b, ad_split_b.m1_b, ad_cont.b_count);
	diff_m2a = stage4d_max_abs_diff(ad_cont.m2_a, ad_split_b.m2_a, ad_cont.a_count);
	diff_m2b = stage4d_max_abs_diff(ad_cont.m2_b, ad_split_b.m2_b, ad_cont.b_count);
	diff_loss = fabs(cont_final - split_final);
	max_diff = diff_a;
	if (diff_b > max_diff)
		max_diff = diff_b;
	if (diff_m1a > max_diff)
		max_diff = diff_m1a;
	if (diff_m1b > max_diff)
		max_diff = diff_m1b;
	if (diff_m2a > max_diff)
		max_diff = diff_m2a;
	if (diff_m2b > max_diff)
		max_diff = diff_m2b;
	if (diff_loss > max_diff)
		max_diff = diff_loss;

	if (!stage4d_generate_greedy(tr, tok, rec->canonical_inference_prompt,
	    opt->max_gen_tokens, &ad_cont, 1, ids_cont, &n_cont,
	    text_cont, sizeof(text_cont)) ||
	    !stage4d_generate_greedy(tr, tok, rec->canonical_inference_prompt,
	    opt->max_gen_tokens, &ad_split_b, 1, ids_split, &n_split,
	    text_split, sizeof(text_split))) {
		fprintf(stderr, "error: resume-check generation failed\n");
		goto fail;
	}
	token_match = (n_cont == n_split && memcmp(ids_cont, ids_split,
	    (size_t)n_cont * sizeof(int)) == 0);

	printf("resume_check steps=%d split_step=%d cont_final_loss=%.12f resumed_final_loss=%.12f diff_loss=%.12e\n",
	    compare_steps, split_step, cont_final, split_final, diff_loss);
	printf("resume_check max_abs_diff_A=%.12e max_abs_diff_B=%.12e max_abs_diff_m1_A=%.12e max_abs_diff_m1_B=%.12e max_abs_diff_m2_A=%.12e max_abs_diff_m2_B=%.12e\n",
	    diff_a, diff_b, diff_m1a, diff_m1b, diff_m2a, diff_m2b);
	printf("resume_check optimizer_step_cont=%llu optimizer_step_resumed=%llu token_ids_match=%s\n",
	    (unsigned long long)st_cont.optimizer_step,
	    (unsigned long long)st_split.optimizer_step,
	    token_match ? "yes" : "no");
	printf("resume_check tolerance=1.0e-6 max_observed_diff=%.12e\n", max_diff);
	if (max_diff > 1e-6 || !token_match ||
	    st_cont.optimizer_step != st_split.optimizer_step) {
		fprintf(stderr, "error: deterministic resume check failed\n");
		goto fail;
	}

	stage4_adapter_free(&ad_cont);
	stage4_adapter_free(&ad_split_a);
	stage4_adapter_free(&ad_split_b);
	return 1;

fail:
	stage4_adapter_free(&ad_cont);
	stage4_adapter_free(&ad_split_a);
	stage4_adapter_free(&ad_split_b);
	return 0;
}

int main(int argc, char **argv)
{
	stage4d_options_t opt;
	Transformer tr;
	Tokenizer tok;
	stage4d_record_t rec;
	stage4_adapter_t ad;
	stage4_adapter_t ad_reload;
	stage4_base_identity_t base_id;
	stage4d_runtime_state_t state;
	stage4_storage_accounting_t mem;
	float *initial_a = NULL;
	float *initial_b = NULL;
	char checkpoint_sha_before[65];
	char checkpoint_sha_after[65];
	char tokenizer_sha[65];
	char dataset_sha[65];
	char adapter_sha[65];
	char train_ckpt_sha[65];
	uint64_t checkpoint_bytes;
	uint64_t adapter_bytes;
	uint64_t ckpt_bytes;
	double start_loss = NAN;
	double final_loss = NAN;
	double best_loss = NAN;
	double post_update_loss = NAN;
	int exact_match = 0;
	int base_before_ids[STAGE4D_MAX_GEN_TOKENS];
	int base_after_ids[STAGE4D_MAX_GEN_TOKENS];
	int train_after_ids[STAGE4D_MAX_GEN_TOKENS];
	int n_base_before = 0;
	int n_base_after = 0;
	int n_train_after = 0;
	char text_base_before[2048];
	char text_base_after[2048];
	char text_train_after[2048];
	int i;
	int argi;

	memset(&tr, 0, sizeof(tr));
	memset(&tok, 0, sizeof(tok));
	memset(&rec, 0, sizeof(rec));
	memset(&ad, 0, sizeof(ad));
	memset(&ad_reload, 0, sizeof(ad_reload));
	memset(&state, 0, sizeof(state));

	opt.checkpoint_path = NULL;
	opt.tokenizer_path = "/usr/src/minix_ai_stage3_stories15m/tokenizer.bin";
	opt.adapter_path = "stories15M.adapter-overfit.bin";
	opt.dataset_path = "overfit-one.records";
	opt.train_ckpt_path = "stories15M.adapter-overfit.train.ckpt";
	opt.resume_ckpt_path = "stories15M.adapter-overfit.train.resume.ckpt";
	opt.rank = 8;
	opt.adapter_scale = 1.0f;
	opt.optimizer = STAGE4D_OPT_ADAM;
	opt.learning_rate = 0.001;
	opt.beta1 = 0.9;
	opt.beta2 = 0.999;
	opt.epsilon = 1e-8;
	opt.weight_decay = 0.0;
	opt.gradient_clip = 1.0;
	opt.steps = 500;
	opt.eval_every = 10;
	opt.seed = 1;
	opt.target_loss_threshold = -1.0;
	opt.stop_on_exact_match = 0;
	opt.max_gen_tokens = STAGE4D_MAX_GEN_TOKENS;
	opt.run_resume_check = 1;

	if (argc < 2) {
		stage4d_usage(argv[0]);
		return 1;
	}
	opt.checkpoint_path = argv[1];

	for (argi = 2; argi < argc; argi++) {
		const char *arg = argv[argi];
		const char *val = (argi + 1 < argc) ? argv[argi + 1] : NULL;
		int b;
		int iv;
		double dv;
		uint64_t uv;

		if (strcmp(arg, "-z") == 0 && val != NULL) {
			opt.tokenizer_path = val;
			argi++;
		} else if (strcmp(arg, "-a") == 0 && val != NULL) {
			opt.adapter_path = val;
			argi++;
		} else if (strcmp(arg, "-d") == 0 && val != NULL) {
			opt.dataset_path = val;
			argi++;
		} else if (strcmp(arg, "-r") == 0 && val != NULL) {
			if (!stage4d_parse_int(val, &iv) || iv <= 0) {
				fprintf(stderr, "error: invalid rank\n");
				return 1;
			}
			opt.rank = iv;
			argi++;
		} else if (strcmp(arg, "-y") == 0 && val != NULL) {
			if (!stage4d_parse_double(val, &dv) || !(dv > 0.0) || dv > FLT_MAX) {
				fprintf(stderr, "error: invalid adapter scale\n");
				return 1;
			}
			opt.adapter_scale = (float)dv;
			argi++;
		} else if (strcmp(arg, "--optimizer") == 0 && val != NULL) {
			if (stage4d_streq(val, "adam"))
				opt.optimizer = STAGE4D_OPT_ADAM;
			else if (stage4d_streq(val, "sgd"))
				opt.optimizer = STAGE4D_OPT_SGD;
			else {
				fprintf(stderr, "error: optimizer must be adam or sgd\n");
				return 1;
			}
			argi++;
		} else if (strcmp(arg, "--learning-rate") == 0 && val != NULL) {
			if (!stage4d_parse_double(val, &dv) || !(dv > 0.0))
				return 1;
			opt.learning_rate = dv;
			argi++;
		} else if (strcmp(arg, "--beta1") == 0 && val != NULL) {
			if (!stage4d_parse_double(val, &dv) || !(dv > 0.0 && dv < 1.0))
				return 1;
			opt.beta1 = dv;
			argi++;
		} else if (strcmp(arg, "--beta2") == 0 && val != NULL) {
			if (!stage4d_parse_double(val, &dv) || !(dv > 0.0 && dv < 1.0))
				return 1;
			opt.beta2 = dv;
			argi++;
		} else if (strcmp(arg, "--epsilon") == 0 && val != NULL) {
			if (!stage4d_parse_double(val, &dv) || !(dv > 0.0))
				return 1;
			opt.epsilon = dv;
			argi++;
		} else if (strcmp(arg, "--weight-decay") == 0 && val != NULL) {
			if (!stage4d_parse_double(val, &dv) || dv < 0.0)
				return 1;
			opt.weight_decay = dv;
			argi++;
		} else if (strcmp(arg, "--gradient-clip") == 0 && val != NULL) {
			if (!stage4d_parse_double(val, &dv) || dv < 0.0)
				return 1;
			opt.gradient_clip = dv;
			argi++;
		} else if (strcmp(arg, "--steps") == 0 && val != NULL) {
			if (!stage4d_parse_int(val, &iv) || iv <= 0)
				return 1;
			opt.steps = iv;
			argi++;
		} else if (strcmp(arg, "--eval-every") == 0 && val != NULL) {
			if (!stage4d_parse_int(val, &iv) || iv <= 0)
				return 1;
			opt.eval_every = iv;
			argi++;
		} else if (strcmp(arg, "--seed") == 0 && val != NULL) {
			if (!stage4d_parse_u64(val, &uv) || uv == 0)
				return 1;
			opt.seed = uv;
			argi++;
		} else if (strcmp(arg, "--target-loss-threshold") == 0 && val != NULL) {
			if (!stage4d_parse_double(val, &dv) || !(dv > 0.0))
				return 1;
			opt.target_loss_threshold = dv;
			argi++;
		} else if (strcmp(arg, "--stop-on-exact-match") == 0 && val != NULL) {
			if (!stage4d_parse_bool01(val, &b))
				return 1;
			opt.stop_on_exact_match = b;
			argi++;
		} else if (strcmp(arg, "--max-gen-tokens") == 0 && val != NULL) {
			if (!stage4d_parse_int(val, &iv) || iv <= 0 || iv > STAGE4D_MAX_GEN_TOKENS)
				return 1;
			opt.max_gen_tokens = iv;
			argi++;
		} else if (strcmp(arg, "--train-checkpoint") == 0 && val != NULL) {
			opt.train_ckpt_path = val;
			argi++;
		} else if (strcmp(arg, "--resume-checkpoint") == 0 && val != NULL) {
			opt.resume_ckpt_path = val;
			argi++;
		} else if (strcmp(arg, "--resume-check") == 0 && val != NULL) {
			if (!stage4d_parse_bool01(val, &b))
				return 1;
			opt.run_resume_check = b;
			argi++;
		} else {
			stage4d_usage(argv[0]);
			return 1;
		}
	}

	if (!stage4_hash_file_sha256_hex(opt.checkpoint_path, checkpoint_sha_before) ||
	    !stage4_file_size_bytes(opt.checkpoint_path, &checkpoint_bytes)) {
		fprintf(stderr, "error: cannot hash/stat checkpoint\n");
		return 1;
	}
	if (strcmp(checkpoint_sha_before, STAGE4D_EXPECTED_BASE_SHA256) != 0) {
		fprintf(stderr,
		    "error: unexpected base checkpoint hash\n"
		    "  got:      %s\n"
		    "  expected: %s\n",
		    checkpoint_sha_before, STAGE4D_EXPECTED_BASE_SHA256);
		return 1;
	}
	if (!stage4_hash_file_sha256_hex(opt.tokenizer_path, tokenizer_sha) ||
	    !stage4_hash_file_sha256_hex(opt.dataset_path, dataset_sha)) {
		fprintf(stderr, "error: cannot hash tokenizer or dataset\n");
		return 1;
	}

	printf("base_checkpoint_sha256=%s\n", checkpoint_sha_before);
	printf("tokenizer_identity_sha256=%s\n", tokenizer_sha);
	printf("dataset_sha256=%s\n", dataset_sha);
	printf("adapter_rank=%d\n", opt.rank);
	printf("adapter_scale=%.9g\n", (double)opt.adapter_scale);
	printf("optimizer=%s learning_rate=%.9g beta1=%.9g beta2=%.9g epsilon=%.9g weight_decay=%.9g gradient_clip=%.9g\n",
	    opt.optimizer == STAGE4D_OPT_ADAM ? "adam" : "sgd",
	    opt.learning_rate,
	    opt.beta1,
	    opt.beta2,
	    opt.epsilon,
	    opt.weight_decay,
	    opt.gradient_clip);
	printf("seed=%llu\n", (unsigned long long)opt.seed);

	load_transformer(&tr, opt.checkpoint_path);
	load_tokenizer(&tok, opt.tokenizer_path, tr.config.vocab_size);

	if (!stage4d_load_record(opt.dataset_path, &rec) ||
	    !stage4d_prepare_tokens(&tok, &tr, &rec)) {
		free_tokenizer(&tok);
		free_transformer(&tr);
		stage4d_free_record(&rec);
		return 1;
	}

	if (!stage4_adapter_init(&ad, tr.config.dim, tr.config.vocab_size,
	    opt.rank, opt.adapter_scale)) {
		fprintf(stderr, "error: adapter init failed\n");
		free_tokenizer(&tok);
		free_transformer(&tr);
		stage4d_free_record(&rec);
		return 1;
	}
	stage4_adapter_fill_a_deterministic(&ad, opt.seed, 0.01f);
	stage4_adapter_zero_b(&ad);
	stage4_adapter_zero_moments(&ad);
	initial_a = calloc(ad.a_count, sizeof(float));
	initial_b = calloc(ad.b_count, sizeof(float));
	if (initial_a == NULL || initial_b == NULL) {
		fprintf(stderr, "error: initial parameter snapshot allocation failed\n");
		free(initial_a);
		free(initial_b);
		stage4_adapter_free(&ad);
		free_tokenizer(&tok);
		free_transformer(&tr);
		stage4d_free_record(&rec);
		return 1;
	}
	memcpy(initial_a, ad.a, ad.a_count * sizeof(float));
	memcpy(initial_b, ad.b, ad.b_count * sizeof(float));

	if (!stage4_adapter_storage_accounting(&ad, &mem)) {
		fprintf(stderr, "error: storage accounting failed\n");
		stage4_adapter_free(&ad);
		free_tokenizer(&tok);
		free_transformer(&tr);
		stage4d_free_record(&rec);
		return 1;
	}
	printf("memory_accounting adapter_weight_bytes=%llu\n",
	    (unsigned long long)mem.adapter_weight_bytes);
	printf("memory_accounting gradient_bytes=%llu\n",
	    (unsigned long long)mem.gradient_bytes);
	printf("memory_accounting adam_first_moment_bytes=%llu\n",
	    (unsigned long long)mem.adam_first_moment_bytes);
	printf("memory_accounting adam_second_moment_bytes=%llu\n",
	    (unsigned long long)mem.adam_second_moment_bytes);
	printf("memory_accounting total_trainable_storage_bytes=%llu\n",
	    (unsigned long long)mem.total_trainable_storage_bytes);

	if (!stage4d_generate_greedy(&tr, &tok, rec.canonical_inference_prompt,
	    opt.max_gen_tokens, &ad, 0,
	    base_before_ids, &n_base_before,
	    text_base_before, sizeof(text_base_before))) {
		fprintf(stderr, "error: initial base generation failed\n");
		goto fail;
	}
	if (!stage4d_generate_greedy(&tr, &tok, rec.canonical_inference_prompt,
	    opt.max_gen_tokens, &ad, 1,
	    train_after_ids, &n_train_after,
	    text_train_after, sizeof(text_train_after))) {
		fprintf(stderr, "error: initial adapter generation failed\n");
		goto fail;
	}
	printf("initial_greedy_base_token_ids:");
	for (i = 0; i < n_base_before; i++)
		printf(" %d", base_before_ids[i]);
	printf("\n");
	printf("initial_greedy_base_text=%s\n", text_base_before);
	printf("initial_greedy_adapter_token_ids:");
	for (i = 0; i < n_train_after; i++)
		printf(" %d", train_after_ids[i]);
	printf("\n");
	printf("initial_greedy_adapter_text=%s\n", text_train_after);

	{
		stage4d_loss_stats_t init_stats;
		if (!stage4d_eval_record(&tr, &ad, &rec, 0, &init_stats)) {
			fprintf(stderr, "error: initial eval failed\n");
			goto fail;
		}
		printf("initial_target_loss_sum=%.12f\n", init_stats.target_loss_sum);
		printf("initial_target_average_loss=%.12f\n", init_stats.target_average_loss);
		printf("initial_target_perplexity=%.12f\n", init_stats.target_perplexity);
		stage4d_print_masking_stats(&init_stats);
	}

	state.optimizer_step = 0;
	state.tokens_processed = 0;
	state.rng_state = opt.seed;
	state.best_loss = DBL_MAX;
	state.current_loss = NAN;

	if (!stage4d_train_steps(&tr, &tok, &ad, &rec, &opt, &state,
	    1, opt.steps, &start_loss, &final_loss, &best_loss, &exact_match)) {
		goto fail;
	}

	strncpy(base_id.base_sha256, checkpoint_sha_before, sizeof(base_id.base_sha256) - 1);
	base_id.base_checkpoint_bytes = checkpoint_bytes;
	if (!stage4_adapter_save_file(opt.adapter_path, &ad, &base_id)) {
		fprintf(stderr, "error: saving adapter failed\n");
		goto fail;
	}
	if (!stage4_hash_file_sha256_hex(opt.adapter_path, adapter_sha) ||
	    !stage4_file_size_bytes(opt.adapter_path, &adapter_bytes)) {
		fprintf(stderr, "error: hashing saved adapter failed\n");
		goto fail;
	}

	if (!stage4_adapter_init(&ad_reload, tr.config.dim, tr.config.vocab_size,
	    opt.rank, opt.adapter_scale) ||
	    !stage4_adapter_load_file(opt.adapter_path, &ad_reload, &base_id, 0)) {
		fprintf(stderr, "error: adapter reload validation failed\n");
		goto fail;
	}
	{
		stage4d_loss_stats_t final_stats;
		if (!stage4d_eval_record(&tr, &ad, &rec, 0, &final_stats)) {
			fprintf(stderr, "error: final post-update eval failed\n");
			goto fail;
		}
		post_update_loss = final_stats.target_average_loss;

		stage4d_loss_stats_t re_stats;
		if (!stage4d_eval_record(&tr, &ad_reload, &rec, 0, &re_stats)) {
			fprintf(stderr, "error: reload eval failed\n");
			goto fail;
		}
		printf("reload_loss_compare before_save=%.12f after_reload=%.12f abs_diff=%.12e\n",
		    post_update_loss, re_stats.target_average_loss,
		    fabs(post_update_loss - re_stats.target_average_loss));
		if (fabs(post_update_loss - re_stats.target_average_loss) > 1e-7) {
			fprintf(stderr, "error: adapter reload changed loss\n");
			goto fail;
		}
	}

	if (!stage4d_save_training_checkpoint_atomic(opt.train_ckpt_path,
	    &ad, &opt, &state, checkpoint_sha_before, checkpoint_bytes,
	    tokenizer_sha, dataset_sha)) {
		fprintf(stderr, "error: saving training checkpoint failed\n");
		goto fail;
	}
	if (!stage4_hash_file_sha256_hex(opt.train_ckpt_path, train_ckpt_sha) ||
	    !stage4_file_size_bytes(opt.train_ckpt_path, &ckpt_bytes)) {
		fprintf(stderr, "error: hashing training checkpoint failed\n");
		goto fail;
	}

	if (!stage4d_generate_greedy(&tr, &tok, rec.canonical_inference_prompt,
	    opt.max_gen_tokens, &ad, 1,
	    train_after_ids, &n_train_after,
	    text_train_after, sizeof(text_train_after)) ||
	    !stage4d_generate_greedy(&tr, &tok, rec.canonical_inference_prompt,
	    opt.max_gen_tokens, &ad, 0,
	    base_after_ids, &n_base_after,
	    text_base_after, sizeof(text_base_after))) {
		fprintf(stderr, "error: final generation failed\n");
		goto fail;
	}

	if (!stage4_hash_file_sha256_hex(opt.checkpoint_path, checkpoint_sha_after)) {
		fprintf(stderr, "error: cannot hash checkpoint after training\n");
		goto fail;
	}
	if (strcmp(checkpoint_sha_before, checkpoint_sha_after) != 0) {
		fprintf(stderr, "error: base checkpoint hash changed\n");
		goto fail;
	}

	if (!(n_base_before == n_base_after &&
	    memcmp(base_before_ids, base_after_ids,
	    (size_t)n_base_before * sizeof(int)) == 0)) {
		fprintf(stderr, "error: base-only generation changed unexpectedly\n");
		goto fail;
	}

	if (opt.run_resume_check) {
		if (!stage4d_run_resume_determinism_check(&tr, &tok, &rec, &opt,
		    checkpoint_sha_before, checkpoint_bytes,
		    tokenizer_sha, dataset_sha)) {
			goto fail;
		}
	}

	{
		double max_delta_a = stage4d_max_abs_diff(initial_a, ad.a, ad.a_count);
		double max_delta_b = stage4d_max_abs_diff(initial_b, ad.b, ad.b_count);
		printf("parameter_change max_abs_delta_A=%.12e max_abs_delta_B=%.12e\n",
		    max_delta_a, max_delta_b);
		if (max_delta_a <= 0.0 && max_delta_b <= 0.0) {
			fprintf(stderr, "error: adapter parameters did not change\n");
			goto fail;
		}
	}
	free(initial_a);
	free(initial_b);

	exact_match = stage4d_exact_target_token_match(&rec,
	    train_after_ids, n_train_after);
	printf("training_complete starting_loss=%.12f final_loss=%.12f best_loss=%.12f steps_completed=%llu target_tokens_processed=%llu\n",
	    start_loss, final_loss, best_loss,
	    (unsigned long long)state.optimizer_step,
	    (unsigned long long)state.tokens_processed);
	printf("adapter_output path=%s size_bytes=%llu sha256=%s\n",
	    opt.adapter_path,
	    (unsigned long long)adapter_bytes,
	    adapter_sha);
	printf("train_checkpoint path=%s size_bytes=%llu sha256=%s\n",
	    opt.train_ckpt_path,
	    (unsigned long long)ckpt_bytes,
	    train_ckpt_sha);
	printf("generation_before_base text=%s\n", text_base_before);
	printf("generation_after_adapter text=%s\n", text_train_after);
	printf("generation_after_base text=%s\n", text_base_after);
	printf("exact_target_match=%s expected_target=%s\n",
	    exact_match ? "yes" : "no", rec.target);
	printf("exact_target_window_token_ids_expected:");
	for (i = 0; i < rec.target_token_count; i++)
		printf(" %d", rec.sequence_tokens[rec.first_target_index + i]);
	printf("\n");
	printf("exact_target_window_token_ids_generated:");
	for (i = 0; i < rec.target_token_count && i < n_train_after; i++)
		printf(" %d", train_after_ids[i]);
	printf("\n");
	printf("base_checkpoint_sha256_before=%s\n", checkpoint_sha_before);
	printf("base_checkpoint_sha256_after=%s\n", checkpoint_sha_after);
	printf("PASS: stage4_train\n");

	stage4_adapter_free(&ad_reload);
	stage4_adapter_free(&ad);
	stage4d_free_record(&rec);
	free_tokenizer(&tok);
	free_transformer(&tr);
	return 0;

fail:
	free(initial_a);
	free(initial_b);
	stage4_adapter_free(&ad_reload);
	stage4_adapter_free(&ad);
	stage4d_free_record(&rec);
	free_tokenizer(&tok);
	free_transformer(&tr);
	return 1;
}
