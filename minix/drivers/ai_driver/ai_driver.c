#include <minix/drivers.h>
#include <minix/chardriver.h>
#include <minix/ai_model.h>
#include <minix/ds.h>
#include <minix/safecopies.h>

#include <assert.h>
#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define AI_MINOR_MAIN 0

#define AI_MAX_COMMAND_LEN 1024
#define AI_MAX_PROMPT_LEN 768
#define AI_MAX_RESPONSE_LEN 4096
#define AI_ERROR_TEXT_LEN 128
#define AI_GEN_RESPONSE_CAP (AI_MAX_RESPONSE_LEN - 96)

#define AI_STATE_IDLE 0
#define AI_STATE_DONE 1
#define AI_STATE_ERROR 2

struct ai_pending_read {
	int active;
	endpoint_t endpt;
	cp_grant_id_t grant;
	size_t size;
	cdev_id_t id;
	u64_t position;
};

struct ai_driver_state {
	int verbose;
	int state;
	int in_progress;
	int backend_state;
	endpoint_t model_endpt;
	int model_endpt_valid;
	char last_command[AI_MAX_COMMAND_LEN + 1];
	char last_prompt[AI_MAX_PROMPT_LEN + 1];
	char response[AI_MAX_RESPONSE_LEN + 1];
	size_t response_len;
	unsigned long request_no;
	uint64_t seed;
	uint32_t steps;
	uint32_t temperature_milli;
	uint32_t topp_milli;
	uint32_t last_generated_tokens;
	uint32_t last_elapsed_ms;
	int last_error_code;
	char last_error_text[AI_ERROR_TEXT_LEN];
	struct ai_pending_read pending_read;
};

static struct ai_driver_state ai_state;

static int ai_open(devminor_t minor, int access, endpoint_t user_endpt);
static int ai_close(devminor_t minor);
static ssize_t ai_read(devminor_t minor, u64_t position, endpoint_t endpt,
	cp_grant_id_t grant, size_t size, int flags, cdev_id_t id);
static ssize_t ai_write(devminor_t minor, u64_t position, endpoint_t endpt,
	cp_grant_id_t grant, size_t size, int flags, cdev_id_t id);
static int ai_ioctl(devminor_t minor, unsigned long request, endpoint_t endpt,
	cp_grant_id_t grant, int flags, endpoint_t user_endpt, cdev_id_t id);
static int ai_cancel(devminor_t minor, endpoint_t endpt, cdev_id_t id);

static void sef_local_startup(void);
static int sef_cb_init_fresh(int type, sef_init_info_t *info);

static int ai_state_name(int state, const char **name);
static int ai_set_response_fmt(const char *fmt, ...);
static void ai_clear_text_state(void);
static void ai_clear_errors(void);
static void ai_set_error(int code, const char *text);
static int ai_check_minor(devminor_t minor);
static int ai_handle_command(const char *line);
static int ai_parse_prompt(const char *src, char *dst, size_t dst_size);
static int ai_generate_via_model(const char *prompt);
static int ai_model_request(int type, struct ai_model_generate_req *req,
	char *response_buf, size_t response_cap, message *reply_out);
static int ai_parse_decimal_milli(const char *text, uint32_t max_milli,
	uint32_t *out);
static int ai_parse_u32(const char *text, uint32_t min, uint32_t max,
	uint32_t *out);
static int ai_parse_u64(const char *text, uint64_t *out);
static void ai_log(const char *fmt, ...);
static int ai_errno_normalize(int r);
static void ai_set_failure_state(const char *stage, int err_code);

static struct chardriver ai_tab = {
	.cdr_open = ai_open,
	.cdr_close = ai_close,
	.cdr_read = ai_read,
	.cdr_write = ai_write,
	.cdr_ioctl = ai_ioctl,
	.cdr_cancel = ai_cancel,
};

static int ai_state_name(int state, const char **name)
{
	switch (state) {
	case AI_STATE_IDLE:
		*name = "idle";
		return OK;
	case AI_STATE_DONE:
		*name = "done";
		return OK;
	case AI_STATE_ERROR:
		*name = "error";
		return OK;
	default:
		return EINVAL;
	}
}

static void ai_log(const char *fmt, ...)
{
	va_list ap;

	printf("ai_driver: ");
	va_start(ap, fmt);
	vprintf(fmt, ap);
	va_end(ap);
	printf("\n");
}

static int ai_errno_normalize(int r)
{
	if (r == OK)
		return OK;
	if (r > 0)
		return r;
	return EIO;
}

static void ai_set_failure_state(const char *stage, int err_code)
{
	const char *state_text;

	ai_state.backend_state = AI_BACKEND_ERROR;
	if (stage != NULL)
		ai_set_error(err_code, stage);
	else
		ai_set_error(err_code, "backend request failed");

	state_text = (ai_state.backend_state == AI_BACKEND_READY) ? "ready" :
		(ai_state.backend_state == AI_BACKEND_UNAVAILABLE) ? "unavailable" : "error";

	if (ai_set_response_fmt(
	    "state=error\n"
	    "request=%lu\n"
	    "backend_state=%s\n"
	    "last_error=%d\n"
	    "last_error_text=%s\n",
	    ai_state.request_no,
	    state_text,
	    ai_state.last_error_code,
	    ai_state.last_error_text[0] != '\0' ? ai_state.last_error_text : "none") != OK) {
		ai_state.response[0] = '\0';
		ai_state.response_len = 0;
	}
}

static int ai_set_response_fmt(const char *fmt, ...)
{
	va_list ap;
	int n;

	va_start(ap, fmt);
	n = vsnprintf(ai_state.response, sizeof(ai_state.response), fmt, ap);
	va_end(ap);

	if (n < 0)
		return EINVAL;
	if ((size_t)n >= sizeof(ai_state.response))
		return E2BIG;

	ai_state.response_len = (size_t)n;
	return OK;
}

static void ai_clear_text_state(void)
{
	ai_state.last_prompt[0] = '\0';
	ai_state.response[0] = '\0';
	ai_state.response_len = 0;
	ai_state.last_generated_tokens = 0;
	ai_state.last_elapsed_ms = 0;
	ai_state.state = AI_STATE_IDLE;
}

static void ai_clear_errors(void)
{
	ai_state.last_error_code = OK;
	ai_state.last_error_text[0] = '\0';
}

static void ai_set_error(int code, const char *text)
{
	ai_state.last_error_code = code;
	if (text != NULL) {
		strlcpy(ai_state.last_error_text, text, sizeof(ai_state.last_error_text));
	} else {
		ai_state.last_error_text[0] = '\0';
	}
	ai_state.state = AI_STATE_ERROR;
}

static int ai_check_minor(devminor_t minor)
{
	if (minor != AI_MINOR_MAIN)
		return ENXIO;

	return OK;
}

static int ai_parse_prompt(const char *src, char *dst, size_t dst_size)
{
	size_t len;

	len = strlen(src);
	if (len == 0)
		return EINVAL;

	if (src[0] == '"' || src[len - 1] == '"') {
		if (len < 2 || src[0] != '"' || src[len - 1] != '"')
			return EINVAL;
		src++;
		len -= 2;
	}

	if (len == 0)
		return EINVAL;
	if (len > AI_MAX_PROMPT_LEN)
		return E2BIG;
	if (len >= dst_size)
		return E2BIG;

	memcpy(dst, src, len);
	dst[len] = '\0';
	return OK;
}

static int ai_parse_decimal_milli(const char *text, uint32_t max_milli,
	uint32_t *out)
{
	const char *p;
	uint64_t int_part;
	uint32_t frac_part;
	uint32_t frac_scale;
	uint64_t milli;

	if (text == NULL || *text == '\0')
		return EINVAL;

	p = text;
	int_part = 0;
	while (*p >= '0' && *p <= '9') {
		int_part = int_part * 10 + (uint64_t)(*p - '0');
		if (int_part > ((uint64_t)max_milli / 1000U) + 1U)
			return EINVAL;
		p++;
	}

	frac_part = 0;
	frac_scale = 0;
	if (*p == '.') {
		p++;
		while (*p >= '0' && *p <= '9') {
			if (frac_scale < 3) {
				frac_part = frac_part * 10 + (uint32_t)(*p - '0');
				frac_scale++;
			}
			p++;
		}
	}

	if (*p != '\0')
		return EINVAL;

	if (frac_scale == 0)
		frac_part = 0;
	else if (frac_scale == 1)
		frac_part *= 100;
	else if (frac_scale == 2)
		frac_part *= 10;

	milli = int_part * 1000U + frac_part;
	if (milli > max_milli)
		return EINVAL;

	*out = (uint32_t)milli;
	return OK;
}

static int ai_parse_u32(const char *text, uint32_t min, uint32_t max,
	uint32_t *out)
{
	const char *p;
	uint64_t value;

	if (text == NULL || *text == '\0')
		return EINVAL;

	p = text;
	value = 0;
	while (*p >= '0' && *p <= '9') {
		value = value * 10 + (uint64_t)(*p - '0');
		if (value > (uint64_t)max)
			return EINVAL;
		p++;
	}

	if (*p != '\0')
		return EINVAL;
	if (value < min || value > max)
		return EINVAL;

	*out = (uint32_t)value;
	return OK;
}

static int ai_parse_u64(const char *text, uint64_t *out)
{
	const char *p;
	uint64_t value;

	if (text == NULL || *text == '\0')
		return EINVAL;

	p = text;
	value = 0;
	while (*p >= '0' && *p <= '9') {
		value = value * 10 + (uint64_t)(*p - '0');
		p++;
	}

	if (*p != '\0')
		return EINVAL;
	if (value == 0)
		value = AI_MODEL_DEFAULT_SEED;

	*out = value;
	return OK;
}

static int ai_model_request(int type, struct ai_model_generate_req *req,
	char *response_buf, size_t response_cap, message *reply_out)
{
	message m;
	endpoint_t model_endpt;
	cp_grant_id_t req_grant;
	int ds_r;
	int r;
	int err;

	if (!ai_state.model_endpt_valid) {
		ds_r = ds_retrieve_label_endpt(AI_MODEL_LABEL, &ai_state.model_endpt);
		if (ds_r != OK)
			return ai_errno_normalize(ds_r);
		ai_state.model_endpt_valid = TRUE;
		if (ai_state.verbose)
			ai_log("ai_model endpoint=%d", ai_state.model_endpt);
	}

	model_endpt = ai_state.model_endpt;
	req_grant = GRANT_INVALID;

	memset(&m, 0, sizeof(m));
	m.m_type = type;
	m.AI_M_VERSION = AI_IPC_PROTOCOL_VERSION;
	if (ai_state.verbose)
		ai_log("checkpoint: before sendrec type=%d endpoint=%d", type, model_endpt);

	if (req != NULL) {
		if (ai_state.verbose)
			ai_log("checkpoint: before request-grant create");
		req_grant = cpf_grant_direct(model_endpt, (vir_bytes)req,
			sizeof(*req), CPF_READ);
		if (!GRANT_VALID(req_grant))
			return ENOMEM;
		if (ai_state.verbose)
			ai_log("checkpoint: after request-grant create grant=%d", req_grant);

		m.AI_M_REQ_GRANT = req_grant;
		m.AI_M_REQ_SIZE = sizeof(*req);
	}

	r = ipc_sendrec(model_endpt, &m);
	if (ai_state.verbose)
		ai_log("checkpoint: after sendrec raw=%d", r);
	if (GRANT_VALID(req_grant))
		cpf_revoke(req_grant);

	if (r != OK) {
		err = ai_errno_normalize(r);
		ai_log("sendrec to ai_model endpoint=%d failed: raw=%d err=%d",
			model_endpt, r, err);
		ai_state.model_endpt_valid = FALSE;
		return err;
	}

	if (m.m_type != AI_MODEL_REPLY)
		return EPROTO;
	if (m.AI_M_VERSION != AI_IPC_PROTOCOL_VERSION)
		return EPROTO;
	if (ai_state.verbose)
		ai_log("checkpoint: reply protocol/version validated");

	ai_state.backend_state = m.AI_M_BACKEND_STATE;

	if (reply_out != NULL)
		*reply_out = m;
	if (ai_state.verbose)
		ai_log("checkpoint: reply fields extracted result=%ld len=%ld", m.AI_M_RESULT,
			m.AI_M_RESPONSE_LEN);

	if (m.AI_M_RESULT != OK)
		return ai_errno_normalize((int)m.AI_M_RESULT);

	if (response_buf != NULL && response_cap > 0) {
		size_t len = (size_t)m.AI_M_RESPONSE_LEN;
		if (len >= response_cap)
			return E2BIG;
		response_buf[len] = '\0';
		if (ai_state.verbose)
			ai_log("checkpoint: response length validated len=%u cap=%u",
				(unsigned)len, (unsigned)response_cap);
	}

	return OK;
}

static int ai_generate_via_model(const char *prompt)
{
	struct ai_model_generate_req req;
	char model_response[AI_GEN_RESPONSE_CAP + 1];
	cp_grant_id_t prompt_grant;
	cp_grant_id_t response_grant;
	message reply;
	int err;
	int r;

	memset(&req, 0, sizeof(req));
	memset(&reply, 0, sizeof(reply));
	model_response[0] = '\0';
	prompt_grant = GRANT_INVALID;
	response_grant = GRANT_INVALID;
	err = OK;

	if (ai_state.in_progress)
		return EBUSY;
	ai_state.in_progress = TRUE;
	if (ai_state.verbose)
		ai_log("checkpoint: begin generate request=%lu", ai_state.request_no);

	if (!ai_state.model_endpt_valid) {
		if (ai_state.verbose)
			ai_log("checkpoint: before endpoint lookup");
		r = ds_retrieve_label_endpt(AI_MODEL_LABEL, &ai_state.model_endpt);
		if (r != OK) {
			err = ai_errno_normalize(r);
			ai_log("ai_model endpoint lookup failed: %d", err);
			goto cleanup;
		}
		ai_state.model_endpt_valid = TRUE;
		if (ai_state.verbose)
			ai_log("ai_model endpoint=%d", ai_state.model_endpt);
	}

	req.version = AI_IPC_PROTOCOL_VERSION;
	req.operation = AI_MODEL_GENERATE;
	req.request_no = ai_state.request_no;
	req.seed = ai_state.seed;
	req.prompt_len = strlen(prompt);
	req.response_cap = AI_GEN_RESPONSE_CAP;
	req.temperature_milli = ai_state.temperature_milli;
	req.topp_milli = ai_state.topp_milli;
	req.steps = ai_state.steps;

	prompt_grant = cpf_grant_direct(ai_state.model_endpt,
		(vir_bytes)prompt, req.prompt_len, CPF_READ);
	if (ai_state.verbose)
		ai_log("checkpoint: prompt grant create result=%d", prompt_grant);
	if (!GRANT_VALID(prompt_grant)) {
		err = ENOMEM;
		goto cleanup;
	}

	response_grant = cpf_grant_direct(ai_state.model_endpt,
		(vir_bytes)model_response, req.response_cap, CPF_WRITE);
	if (ai_state.verbose)
		ai_log("checkpoint: response grant create result=%d", response_grant);
	if (!GRANT_VALID(response_grant)) {
		err = ENOMEM;
		goto cleanup;
	}

	req.prompt_grant = prompt_grant;
	req.response_grant = response_grant;

	r = ai_model_request(AI_MODEL_GENERATE, &req, model_response,
		sizeof(model_response), &reply);
	if (ai_state.verbose)
		ai_log("checkpoint: after ai_model_request r=%d", r);
	if (r != OK) {
		err = ai_errno_normalize(r);
		ai_log("AI_MODEL_GENERATE sendrec failed: %d", err);
		goto cleanup;
	}

	ai_state.last_generated_tokens = (uint32_t)reply.AI_M_GEN_TOKENS;
	ai_state.last_elapsed_ms = (uint32_t)reply.AI_M_ELAPSED_MS;

	r = ai_set_response_fmt(
	    "request=%lu\n"
	    "backend=ai_model\n"
	    "tokens=%u\n"
	    "elapsed_ms=%u\n"
	    "response=%s\n",
	    ai_state.request_no,
	    ai_state.last_generated_tokens,
	    ai_state.last_elapsed_ms,
	    model_response);
	if (ai_state.verbose)
		ai_log("checkpoint: response formatting r=%d", r);
	if (r != OK) {
		err = ai_errno_normalize(r);
		goto cleanup;
	}

cleanup:
	if (GRANT_VALID(prompt_grant))
		if (ai_state.verbose)
			ai_log("prompt grant revoked: %d", prompt_grant);
	if (GRANT_VALID(prompt_grant))
		cpf_revoke(prompt_grant);
	if (GRANT_VALID(response_grant))
		if (ai_state.verbose)
			ai_log("response grant revoked: %d", response_grant);
	if (GRANT_VALID(response_grant))
		cpf_revoke(response_grant);
	ai_state.in_progress = FALSE;

	if (err != OK)
		ai_set_failure_state("ai_model request failed", err);

	return err;
}

static int ai_handle_command(const char *line)
{
	int r;
	const char *state_name;
	char prompt[AI_MAX_PROMPT_LEN + 1];
	uint32_t parsed_u32;
	uint64_t parsed_u64;

	if (strcmp(line, "-v on") == 0) {
		ai_state.verbose = TRUE;
		ai_state.state = AI_STATE_DONE;
		r = ai_set_response_fmt("verbose=on\n");
		if (r != OK)
			return r;
		ai_clear_errors();
		return OK;
	}

	if (strcmp(line, "-v off") == 0) {
		ai_state.verbose = FALSE;
		ai_state.state = AI_STATE_DONE;
		r = ai_set_response_fmt("verbose=off\n");
		if (r != OK)
			return r;
		ai_clear_errors();
		return OK;
	}

	if (strcmp(line, "status") == 0) {
		if ((r = ai_state_name(ai_state.state, &state_name)) != OK)
			return r;

		r = ai_set_response_fmt(
		    "state=%s\n"
		    "request=%lu\n"
		    "backend_state=%d\n"
		    "verbose=%s\n"
		    "temperature=%.3f\n"
		    "top_p=%.3f\n"
		    "steps=%u\n"
		    "seed=%llu\n"
		    "generated_tokens=%u\n"
		    "elapsed_ms=%u\n"
		    "prompt_length=%zu\n"
		    "response_length=%zu\n"
		    "last_error=%d\n"
		    "last_error_text=%s\n",
		    state_name,
		    ai_state.request_no,
		    ai_state.backend_state,
		    ai_state.verbose ? "on" : "off",
		    (double)ai_state.temperature_milli / 1000.0,
		    (double)ai_state.topp_milli / 1000.0,
		    ai_state.steps,
		    (unsigned long long)ai_state.seed,
		    ai_state.last_generated_tokens,
		    ai_state.last_elapsed_ms,
		    strlen(ai_state.last_prompt),
		    ai_state.response_len,
		    ai_state.last_error_code,
		    ai_state.last_error_text[0] != '\0' ? ai_state.last_error_text : "none");
		if (r != OK)
			return r;

		ai_state.state = AI_STATE_DONE;
		return OK;
	}

	if (strcmp(line, "reset") == 0) {
		ai_state.temperature_milli = AI_MODEL_DEFAULT_TEMP_MILLI;
		ai_state.topp_milli = AI_MODEL_DEFAULT_TOPP_MILLI;
		ai_state.steps = AI_MODEL_DEFAULT_STEPS;
		ai_state.seed = AI_MODEL_DEFAULT_SEED;
		ai_clear_text_state();
		ai_clear_errors();
		(void)ai_model_request(AI_MODEL_RESET, NULL, NULL, 0, NULL);
		r = ai_set_response_fmt("reset=ok\n");
		if (r != OK)
			return r;
		ai_state.state = AI_STATE_DONE;
		return OK;
	}

	if (strncmp(line, "-t ", 3) == 0) {
		r = ai_parse_decimal_milli(line + 3, 5000, &parsed_u32);
		if (r != OK)
			return r;
		ai_state.temperature_milli = parsed_u32;
		ai_state.state = AI_STATE_DONE;
		ai_clear_errors();
		return ai_set_response_fmt("temperature=%.3f\n",
			(double)ai_state.temperature_milli / 1000.0);
	}

	if (strncmp(line, "-p ", 3) == 0) {
		r = ai_parse_decimal_milli(line + 3, 1000, &parsed_u32);
		if (r != OK)
			return r;
		ai_state.topp_milli = parsed_u32;
		ai_state.state = AI_STATE_DONE;
		ai_clear_errors();
		return ai_set_response_fmt("top_p=%.3f\n",
			(double)ai_state.topp_milli / 1000.0);
	}

	if (strncmp(line, "-n ", 3) == 0) {
		r = ai_parse_u32(line + 3, 1, AI_MODEL_MAX_STEPS, &parsed_u32);
		if (r != OK)
			return r;
		ai_state.steps = parsed_u32;
		ai_state.state = AI_STATE_DONE;
		ai_clear_errors();
		return ai_set_response_fmt("steps=%u\n", ai_state.steps);
	}

	if (strncmp(line, "-s ", 3) == 0) {
		r = ai_parse_u64(line + 3, &parsed_u64);
		if (r != OK)
			return r;
		ai_state.seed = parsed_u64;
		ai_state.state = AI_STATE_DONE;
		ai_clear_errors();
		return ai_set_response_fmt("seed=%llu\n",
			(unsigned long long)ai_state.seed);
	}

	if (strncmp(line, "i ", 2) == 0) {
		r = ai_parse_prompt(line + 2, prompt, sizeof(prompt));
		if (r != OK)
			return r;

		ai_state.request_no++;
		strlcpy(ai_state.last_prompt, prompt, sizeof(ai_state.last_prompt));
		r = ai_generate_via_model(ai_state.last_prompt);
		if (r != OK)
			return ai_errno_normalize(r);

		ai_state.state = AI_STATE_DONE;
		ai_clear_errors();
		return OK;
	}

	return EINVAL;
}

static int ai_open(devminor_t minor, int UNUSED(access), endpoint_t UNUSED(user_endpt))
{
	return ai_check_minor(minor);
}

static int ai_close(devminor_t minor)
{
	return ai_check_minor(minor);
}

static ssize_t ai_read(devminor_t minor, u64_t position, endpoint_t endpt,
	cp_grant_id_t grant, size_t size, int UNUSED(flags), cdev_id_t UNUSED(id))
{
	size_t bytes_left, chunk;
	size_t pos;
	int r;

	if ((r = ai_check_minor(minor)) != OK)
		return r;

	if (size == 0)
		return 0;

	if (position >= (u64_t)ai_state.response_len)
		return 0;

	pos = (size_t)position;
	bytes_left = ai_state.response_len - pos;
	chunk = (size < bytes_left) ? size : bytes_left;

	r = sys_safecopyto(endpt, grant, 0,
	    (vir_bytes)&ai_state.response[pos], chunk);
	if (r != OK) {
		ai_set_error(EFAULT, "read safecopyto failed");
		return EFAULT;
	}

	return (ssize_t)chunk;
}

static ssize_t ai_write(devminor_t minor, u64_t UNUSED(position), endpoint_t endpt,
	cp_grant_id_t grant, size_t size, int UNUSED(flags), cdev_id_t UNUSED(id))
{
	char command[AI_MAX_COMMAND_LEN + 1];
	size_t line_len;
	char *nl;
	int r;

	if ((r = ai_check_minor(minor)) != OK)
		return r;

	if (size == 0) {
		ai_set_error(EINVAL, "empty command");
		return EINVAL;
	}
	if (size > AI_MAX_COMMAND_LEN) {
		ai_set_error(E2BIG, "command too long");
		return E2BIG;
	}

	r = sys_safecopyfrom(endpt, grant, 0, (vir_bytes)command, size);
	if (r != OK) {
		ai_set_error(EFAULT, "write safecopyfrom failed");
		return EFAULT;
	}

	command[size] = '\0';
	if (memchr(command, '\0', size) != NULL) {
		ai_set_error(EINVAL, "command contains NUL byte");
		return EINVAL;
	}

	nl = strrchr(command, '\n');
	if (nl == NULL || nl != &command[size - 1]) {
		ai_set_error(EINVAL, "command must end with newline");
		return EINVAL;
	}

	while (size > 0 && (command[size - 1] == '\n' || command[size - 1] == '\r')) {
		command[size - 1] = '\0';
		size--;
	}

	line_len = strlen(command);
	if (line_len == 0) {
		ai_set_error(EINVAL, "empty command");
		return EINVAL;
	}

	strlcpy(ai_state.last_command, command, sizeof(ai_state.last_command));

	r = ai_handle_command(command);
	if (r != OK) {
		r = ai_errno_normalize(r);
		if (ai_state.last_error_code == OK) {
			if (r == E2BIG)
				ai_set_error(E2BIG, "prompt or response too large");
			else
				ai_set_error(r, "malformed or unknown command");
			ai_set_failure_state("command failed", r);
		}
		return r;
	}

	return (ssize_t)(line_len + 1);
}

static int ai_ioctl(devminor_t minor, unsigned long UNUSED(request),
	endpoint_t UNUSED(endpt), cp_grant_id_t UNUSED(grant), int UNUSED(flags),
	endpoint_t UNUSED(user_endpt), cdev_id_t UNUSED(id))
{
	int r;

	if ((r = ai_check_minor(minor)) != OK)
		return r;

	return ENOTTY;
}

static int ai_cancel(devminor_t minor, endpoint_t endpt, cdev_id_t id)
{
	if (ai_check_minor(minor) != OK)
		return ENXIO;

	if (ai_state.pending_read.active &&
	    ai_state.pending_read.endpt == endpt &&
	    ai_state.pending_read.id == id) {
		ai_state.pending_read.active = FALSE;
		return EINTR;
	}

	return EDONTREPLY;
}

/*
 * Tutorial note for async follow-up version:
 *
 * This prototype answers writes synchronously and serves reads from an in-memory
 * response buffer. A future asynchronous version can keep one pending read by
 * returning EDONTREPLY from ai_read, storing endpoint/grant/size/id/position,
 * and finishing later through chardriver_reply_task(endpt, id, result) once an
 * external ai_model service returns a typed IPC response. The cancel callback
 * should match endpoint+id and return EINTR for the pending request. While one
 * request is pending, a second request should return EBUSY.
 */

static void sef_local_startup(void)
{
	sef_setcb_init_fresh(sef_cb_init_fresh);
	sef_setcb_init_restart(sef_cb_init_fresh);

	sef_startup();
}

static int sef_cb_init_fresh(int UNUSED(type), sef_init_info_t *UNUSED(info))
{
	memset(&ai_state, 0, sizeof(ai_state));
	ai_state.state = AI_STATE_IDLE;
	ai_state.backend_state = AI_BACKEND_UNAVAILABLE;
	ai_state.seed = AI_MODEL_DEFAULT_SEED;
	ai_state.steps = AI_MODEL_DEFAULT_STEPS;
	ai_state.temperature_milli = AI_MODEL_DEFAULT_TEMP_MILLI;
	ai_state.topp_milli = AI_MODEL_DEFAULT_TOPP_MILLI;
	ai_clear_errors();
	(void)ai_model_request(AI_MODEL_STATUS, NULL, NULL, 0, NULL);

	chardriver_announce();
	return OK;
}

int main(void)
{
	sef_local_startup();
	chardriver_task(&ai_tab);
	return OK;
}
