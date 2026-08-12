#ifndef AI_MODEL_AI_MODEL_H
#define AI_MODEL_AI_MODEL_H

#include <minix/ipc.h>

struct ai_model_state {
	int backend_state;
	uint64_t request_no;
	uint64_t load_count;
	uint32_t last_tokens;
	uint32_t last_elapsed_ms;
	uint32_t last_response_len;
	int last_result;
};

#endif
