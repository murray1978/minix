#include "ai_model.h"
#include "ai_inference.h"

#include <minix/ai_model.h>
#include <minix/drivers.h>

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define AI_DEFAULT_CHECKPOINT "/usr/src/minix_ai_stage3_stories15m/stories15M.bin"
#define AI_DEFAULT_TOKENIZER  "/usr/src/minix_ai_stage3_stories15m/tokenizer.bin"

static struct ai_model_state g_state;

static const char *g_checkpoint_path = AI_DEFAULT_CHECKPOINT;
static const char *g_tokenizer_path = AI_DEFAULT_TOKENIZER;
static const char *g_diag_path;
static uint32_t g_diag_positions;

static void sef_local_startup(void);
static int sef_cb_init_fresh(int type, sef_init_info_t *info);

static int ai_model_handle_generate(endpoint_t caller, message *m_ptr);
static int ai_model_handle_status(message *m_ptr);
static int ai_model_handle_reset(message *m_ptr);
static void ai_model_parse_args(int argc, char **argv);

static void ai_model_parse_args(int argc, char **argv)
{
	int i;

	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
			g_checkpoint_path = argv[++i];
			continue;
		}
		if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
			g_tokenizer_path = argv[++i];
			continue;
		}
		if (strcmp(argv[i], "-D") == 0 && i + 1 < argc) {
			g_diag_path = argv[++i];
			continue;
		}
		if (strcmp(argv[i], "-X") == 0 && i + 1 < argc) {
			g_diag_positions = (uint32_t)strtoul(argv[++i], NULL, 10);
			continue;
		}
	}
}

static int ai_model_handle_generate(endpoint_t caller, message *m_ptr)
{
	struct ai_model_generate_req req;
	struct ai_infer_options opts;
	struct ai_infer_stats stats;
	char prompt[AI_MODEL_MAX_PROMPT + 1];
	char response[AI_MODEL_MAX_RESPONSE + 1];
	size_t response_len;
	int r;

	if (m_ptr->AI_M_VERSION != AI_IPC_PROTOCOL_VERSION)
		return EPROTO;
	if ((size_t)m_ptr->AI_M_REQ_SIZE != sizeof(req))
		return EINVAL;

	r = sys_safecopyfrom(caller, (cp_grant_id_t)m_ptr->AI_M_REQ_GRANT, 0,
		(vir_bytes)&req, sizeof(req));
	if (r != OK)
		return EFAULT;

	if (req.version != AI_IPC_PROTOCOL_VERSION)
		return EPROTO;
	if (req.operation != AI_MODEL_GENERATE)
		return EINVAL;
	if (req.prompt_len == 0 || req.prompt_len > AI_MODEL_MAX_PROMPT)
		return EINVAL;
	if (req.response_cap == 0 || req.response_cap > AI_MODEL_MAX_RESPONSE)
		return E2BIG;

	r = sys_safecopyfrom(caller, req.prompt_grant, 0,
		(vir_bytes)prompt, req.prompt_len);
	if (r != OK)
		return EFAULT;

	if (memchr(prompt, '\0', req.prompt_len) != NULL)
		return EINVAL;
	prompt[req.prompt_len] = '\0';

	opts.temperature_milli = req.temperature_milli;
	opts.topp_milli = req.topp_milli;
	opts.steps = req.steps;
	opts.seed = req.seed;

	r = ai_infer_generate(prompt, response, req.response_cap, &opts, &stats);
	if (r != OK)
		return r;

	response_len = strlen(response);
	r = sys_safecopyto(caller, req.response_grant, 0,
		(vir_bytes)response, response_len + 1);
	if (r != OK)
		return EFAULT;

	g_state.request_no = req.request_no;
	g_state.last_tokens = stats.generated_tokens;
	g_state.last_elapsed_ms = stats.elapsed_ms;
	g_state.last_response_len = (uint32_t)response_len;
	g_state.last_result = OK;

	m_ptr->AI_M_GEN_TOKENS = stats.generated_tokens;
	m_ptr->AI_M_ELAPSED_MS = stats.elapsed_ms;
	m_ptr->AI_M_RESPONSE_LEN = response_len;
	m_ptr->AI_M_REQUEST_NO = req.request_no;
	return OK;
}

static int ai_model_handle_status(message *m_ptr)
{
	m_ptr->AI_M_BACKEND_STATE = g_state.backend_state;
	m_ptr->AI_M_REQUEST_NO = g_state.request_no;
	m_ptr->AI_M_LOAD_COUNT = g_state.load_count;
	m_ptr->AI_M_GEN_TOKENS = g_state.last_tokens;
	m_ptr->AI_M_ELAPSED_MS = g_state.last_elapsed_ms;
	m_ptr->AI_M_RESPONSE_LEN = g_state.last_response_len;
	return OK;
}

static int ai_model_handle_reset(message *m_ptr)
{
	ai_infer_reset();
	g_state.request_no = 0;
	g_state.last_tokens = 0;
	g_state.last_elapsed_ms = 0;
	g_state.last_response_len = 0;
	g_state.last_result = OK;

	m_ptr->AI_M_BACKEND_STATE = g_state.backend_state;
	m_ptr->AI_M_LOAD_COUNT = g_state.load_count;
	return OK;
}

static void sef_local_startup(void)
{
	sef_setcb_init_fresh(sef_cb_init_fresh);
	sef_setcb_init_restart(sef_cb_init_fresh);

	sef_startup();
}

static int sef_cb_init_fresh(int UNUSED(type), sef_init_info_t *UNUSED(info))
{
	int r;

	memset(&g_state, 0, sizeof(g_state));
	g_state.backend_state = AI_BACKEND_UNAVAILABLE;
	g_state.last_result = EAGAIN;

	if (g_diag_path != NULL) {
		r = ai_infer_set_diagnostics(g_diag_path, g_diag_positions);
		if (r != OK) {
			g_state.backend_state = AI_BACKEND_ERROR;
			g_state.last_result = r;
			return OK;
		}
	}

	r = ai_infer_init(g_checkpoint_path, g_tokenizer_path);
	if (r == OK) {
		g_state.backend_state = AI_BACKEND_READY;
		g_state.load_count = ai_infer_load_count();
		g_state.last_result = OK;
	} else {
		g_state.backend_state = AI_BACKEND_ERROR;
		g_state.last_result = r;
	}

	return OK;
}

int main(int argc, char **argv)
{
	message m;
	int r;
	int result;
	endpoint_t caller;

	env_setargs(argc, argv);
	ai_model_parse_args(argc, argv);
	sef_local_startup();

	while (TRUE) {
		if ((r = sef_receive(ANY, &m)) != OK) {
			printf("ai_model: sef_receive failed: %d\n", r);
			continue;
		}

		caller = m.m_source;
		if (is_notify(m.m_type))
			continue;

		switch (m.m_type) {
		case AI_MODEL_GENERATE:
			result = ai_model_handle_generate(caller, &m);
			break;
		case AI_MODEL_STATUS:
			result = ai_model_handle_status(&m);
			break;
		case AI_MODEL_RESET:
			result = ai_model_handle_reset(&m);
			break;
		default:
			result = EINVAL;
			break;
		}

		g_state.last_result = result;

		m.m_type = AI_MODEL_REPLY;
		m.AI_M_RESULT = result;
		m.AI_M_BACKEND_STATE = g_state.backend_state;
		m.AI_M_LOAD_COUNT = g_state.load_count;
		m.AI_M_VERSION = AI_IPC_PROTOCOL_VERSION;

		if ((r = ipc_send(caller, &m)) != OK)
			printf("ai_model: ipc_send to %d failed: %d\n", caller, r);
	}

	return OK;
}
