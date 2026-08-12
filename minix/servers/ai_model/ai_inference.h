#ifndef AI_MODEL_AI_INFERENCE_H
#define AI_MODEL_AI_INFERENCE_H

#include <stddef.h>
#include <stdint.h>

struct ai_infer_options {
	uint32_t temperature_milli;
	uint32_t topp_milli;
	uint32_t steps;
	uint64_t seed;
};

struct ai_infer_stats {
	uint32_t generated_tokens;
	uint32_t elapsed_ms;
};

int ai_infer_init(const char *checkpoint_path, const char *tokenizer_path);
void ai_infer_shutdown(void);
void ai_infer_reset(void);
int ai_infer_backend_state(void);
uint64_t ai_infer_load_count(void);
int ai_infer_set_diagnostics(const char *path, uint32_t positions);

int ai_infer_generate(const char *prompt, char *response, size_t response_cap,
	const struct ai_infer_options *opts, struct ai_infer_stats *stats);

#endif
