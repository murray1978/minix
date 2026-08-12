#ifndef AI_DRIVER_AI_IPC_H
#define AI_DRIVER_AI_IPC_H

#include <minix/ipc.h>
#include <minix/type.h>

#define AI_IPC_PROTOCOL_VERSION 1

#define AI_MODEL_LABEL "ai_model"
#define AI_DRIVER_LABEL "ai_driver"

#define AI_MODEL_REQ_BASE 0x7A10
#define AI_MODEL_GENERATE (AI_MODEL_REQ_BASE + 1)
#define AI_MODEL_STATUS   (AI_MODEL_REQ_BASE + 2)
#define AI_MODEL_RESET    (AI_MODEL_REQ_BASE + 3)

#define AI_MODEL_REPLY    (AI_MODEL_REQ_BASE + 10)

#define AI_BACKEND_UNAVAILABLE 0
#define AI_BACKEND_READY       1
#define AI_BACKEND_ERROR       2

#define AI_MODEL_MAX_PROMPT   768
#define AI_MODEL_MAX_RESPONSE 4096
#define AI_MODEL_MAX_STEPS    256

#define AI_MODEL_DEFAULT_TEMP_MILLI 0
#define AI_MODEL_DEFAULT_TOPP_MILLI 900
#define AI_MODEL_DEFAULT_STEPS      64
#define AI_MODEL_DEFAULT_SEED       1ULL

struct ai_model_generate_req {
	uint32_t version;
	uint32_t operation;
	uint64_t request_no;
	uint64_t seed;
	uint32_t prompt_len;
	uint32_t response_cap;
	uint32_t temperature_milli;
	uint32_t topp_milli;
	uint32_t steps;
	cp_grant_id_t prompt_grant;
	cp_grant_id_t response_grant;
};

/* Message field helpers for message m9. */
#define AI_M_REQ_SIZE       m9_l1
#define AI_M_REQ_GRANT      m9_l2
#define AI_M_RESULT         m9_l1
#define AI_M_BACKEND_STATE  m9_l2
#define AI_M_GEN_TOKENS     m9_l3
#define AI_M_ELAPSED_MS     m9_l4
#define AI_M_RESPONSE_LEN   m9_l5
#define AI_M_REQUEST_NO     m9_ull1
#define AI_M_LOAD_COUNT     m9_ull2
#define AI_M_VERSION        m9_s1

#endif
