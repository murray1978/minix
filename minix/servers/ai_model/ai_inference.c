#include "ai_inference.h"

#include <minix/ai_model.h>
#include <minix/const.h>

#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef OK
#define OK 0
#endif

#define main minix_llama_embedded_main
#include "../../../minix_ai_stage3_stories15m/minix_llama.c"
#undef main

static Transformer g_transformer;
static Tokenizer g_tokenizer;
static int g_ready;
static uint64_t g_load_count;
static char g_diag_path[256];
static uint32_t g_diag_positions;

static int append_piece(char *out, size_t cap, size_t *used, const char *piece)
{
	size_t piece_len;
	unsigned char value;

	if (piece == NULL || piece[0] == '\0')
		return OK;

	if (piece[1] == '\0') {
		value = (unsigned char)piece[0];
		if (!(isprint((int)value) || isspace((int)value)))
			return OK;
	}

	piece_len = strlen(piece);
	if (*used + piece_len + 1 > cap)
		return E2BIG;

	memcpy(out + *used, piece, piece_len);
	*used += piece_len;
	out[*used] = '\0';
	return OK;
}

static void normalize_options(struct ai_infer_options *dst,
	const struct ai_infer_options *src)
{
	dst->temperature_milli = AI_MODEL_DEFAULT_TEMP_MILLI;
	dst->topp_milli = AI_MODEL_DEFAULT_TOPP_MILLI;
	dst->steps = AI_MODEL_DEFAULT_STEPS;
	dst->seed = AI_MODEL_DEFAULT_SEED;

	if (src == NULL)
		return;

	if (src->temperature_milli <= 5000)
		dst->temperature_milli = src->temperature_milli;
	if (src->topp_milli <= 1000)
		dst->topp_milli = src->topp_milli;
	if (src->steps > 0 && src->steps <= AI_MODEL_MAX_STEPS)
		dst->steps = src->steps;
	if (src->seed != 0)
		dst->seed = src->seed;
}

int ai_infer_init(const char *checkpoint_path, const char *tokenizer_path)
{
	if (g_ready)
		return OK;

	if (checkpoint_path == NULL || tokenizer_path == NULL)
		return EINVAL;

	load_transformer(&g_transformer, checkpoint_path);
	load_tokenizer(&g_tokenizer, tokenizer_path, g_transformer.config.vocab_size);
	if (g_diag_path[0] != '\0') {
		if (minix_llama_set_diagnostic_output(g_diag_path,
		    (int)g_diag_positions) != 0)
			return EIO;
	}

	g_ready = TRUE;
	g_load_count++;
	return OK;
}

void ai_infer_shutdown(void)
{
	if (!g_ready)
		return;

	free_tokenizer(&g_tokenizer);
	free_transformer(&g_transformer);
	minix_llama_clear_diagnostic_output();
	g_ready = FALSE;
}

void ai_infer_reset(void)
{
	if (!g_ready)
		return;

	clear_run_state(&g_transformer);
}

int ai_infer_backend_state(void)
{
	return g_ready ? AI_BACKEND_READY : AI_BACKEND_UNAVAILABLE;
}

uint64_t ai_infer_load_count(void)
{
	return g_load_count;
}

int ai_infer_set_diagnostics(const char *path, uint32_t positions)
{
	if (path == NULL || path[0] == '\0') {
		g_diag_path[0] = '\0';
		g_diag_positions = 0;
		if (g_ready)
			minix_llama_clear_diagnostic_output();
		return OK;
	}

	if (strlen(path) >= sizeof(g_diag_path))
		return E2BIG;

	strlcpy(g_diag_path, path, sizeof(g_diag_path));
	g_diag_positions = (positions > 0) ? positions : 4;

	if (g_ready) {
		minix_llama_clear_diagnostic_output();
		if (minix_llama_set_diagnostic_output(g_diag_path,
		    (int)g_diag_positions) != 0)
			return EIO;
	}

	return OK;
}

int ai_infer_generate(const char *prompt, char *response, size_t response_cap,
	const struct ai_infer_options *opts, struct ai_infer_stats *stats)
{
	struct ai_infer_options options;
	Sampler sampler;
	int *prompt_tokens;
	int prompt_count;
	int token;
	int next;
	int position;
	double start;
	double end;
	unsigned int generated;
	size_t used;
	int r;

	if (!g_ready)
		return EAGAIN;
	if (prompt == NULL || response == NULL || response_cap == 0)
		return EINVAL;
	if (strlen(prompt) == 0 || strlen(prompt) > AI_MODEL_MAX_PROMPT)
		return EINVAL;

	normalize_options(&options, opts);

	if (!encode_text(&g_tokenizer, prompt, 1, 0, &prompt_tokens, &prompt_count))
		return EINVAL;

	if (prompt_count < 1 || prompt_count > g_transformer.config.seq_len) {
		free(prompt_tokens);
		return E2BIG;
	}

	build_sampler(&sampler, g_transformer.config.vocab_size,
		(float)options.temperature_milli / 1000.0f,
		(float)options.topp_milli / 1000.0f,
		options.seed);

	clear_run_state(&g_transformer);

	response[0] = '\0';
	used = 0;
	generated = 0;
	start = 0.0;
	end = 0.0;
	token = prompt_tokens[0];

	for (position = 0; position < (int)options.steps; position++) {
		float *logits = forward(&g_transformer, token, position);

		if (position < prompt_count - 1)
			next = prompt_tokens[position + 1];
		else {
			if (generated == 0)
				start = monotonic_seconds();
			next = sample_token(&sampler, logits);
			generated++;
		}

		if (next == 1)
			break;

		r = append_piece(response, response_cap, &used,
			decode_piece(&g_tokenizer, token, next));
		if (r != OK) {
			free_sampler(&sampler);
			free(prompt_tokens);
			return r;
		}

		token = next;
	}

	if (generated > 0)
		end = monotonic_seconds();

	if (stats != NULL) {
		stats->generated_tokens = generated;
		if (generated > 0 && end > start)
			stats->elapsed_ms = (uint32_t)((end - start) * 1000.0);
		else
			stats->elapsed_ms = 0;
	}

	free_sampler(&sampler);
	free(prompt_tokens);
	return OK;
}
