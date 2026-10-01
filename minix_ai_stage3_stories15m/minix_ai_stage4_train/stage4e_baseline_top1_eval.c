#include "stage4_adapter_train.h"

#include <ctype.h>
#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int minix_llama_embedded_main(int argc, char **argv);

#define main minix_llama_embedded_main
#include "../minix_llama.c"
#undef main

#define STAGE4E_MAX_LINE 8192
#define STAGE4E_MAX_TEXT 4096
#define STAGE4E_MAX_FIELD 256
#define STAGE4E_SPLIT_COUNT 4
#define STAGE4E_TASK_COUNT 6

#define STAGE4E_EXPECTED_BASE_SHA256 "cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a"
#define STAGE4E_EXPECTED_TOKENIZER_SHA256 "50a52ef822ee9e83de5ce9d0be0a025a773d019437f58b5ff9dcafb063ece361"

typedef struct {
	char id[STAGE4E_MAX_FIELD];
	char task[STAGE4E_MAX_FIELD];
	char split[STAGE4E_MAX_FIELD];
	double weight;
	char prompt[STAGE4E_MAX_TEXT];
	char target_visible[STAGE4E_MAX_TEXT];
	char canonical_prompt[STAGE4E_MAX_TEXT * 2];
	char canonical_training_seq[STAGE4E_MAX_TEXT * 2];
	int *prompt_tokens;
	int prompt_count;
	int *training_seq_tokens;
	int training_seq_count;
	int first_target_index;
	int target_token_count;
	int begin_line;
} stage4e_record_t;

typedef struct {
	stage4e_record_t *items;
	size_t count;
	size_t cap;
} stage4e_dataset_t;

typedef struct {
	uint64_t records;
	uint64_t pred_tokens;
	double loss_sum;
	uint64_t top1_correct;
} stage4e_metric_t;

enum {
	STAGE4E_SECTION_OUTSIDE = 0,
	STAGE4E_SECTION_HEADER,
	STAGE4E_SECTION_PROMPT,
	STAGE4E_SECTION_TARGET
};

static const char *stage4e_split_names[STAGE4E_SPLIT_COUNT] = {
	"train", "validation", "test", "regression"
};

static const char *stage4e_task_names[STAGE4E_TASK_COUNT] = {
	"command",
	"explanation",
	"source_navigation",
	"diagnosis",
	"command_review",
	"continuation"
};

static void stage4e_usage(const char *prog) NORETURN;

static void stage4e_usage(const char *prog)
{
	fprintf(stderr,
	    "Usage: %s <checkpoint.bin> -d <stage4e-approved.records> -z <tokenizer.bin> [options]\n"
	    "\n"
	    "Options:\n"
	    "  -o <report.txt> output report path\n"
	    "                (default stage4e-baseline-unadapted-top1-report.txt)\n"
	    "  -v            verbose per-record output\n",
	    prog);
	exit(EXIT_FAILURE);
}

static int stage4e_split_index(const char *split)
{
	if (strcmp(split, "train") == 0)
		return 0;
	if (strcmp(split, "validation") == 0)
		return 1;
	if (strcmp(split, "test") == 0)
		return 2;
	if (strcmp(split, "regression") == 0)
		return 3;
	return -1;
}

static int stage4e_task_index(const char *task)
{
	if (strcmp(task, "command") == 0)
		return 0;
	if (strcmp(task, "explanation") == 0)
		return 1;
	if (strcmp(task, "source_navigation") == 0)
		return 2;
	if (strcmp(task, "diagnosis") == 0)
		return 3;
	if (strcmp(task, "command_review") == 0)
		return 4;
	if (strcmp(task, "continuation") == 0)
		return 5;
	return -1;
}

static void stage4e_rtrim_newline(char *line)
{
	size_t n = strlen(line);
	while (n > 0 && (line[n - 1] == '\n' || line[n - 1] == '\r')) {
		line[n - 1] = '\0';
		n--;
	}
}

static const char *stage4e_skip_ws(const char *s)
{
	while (*s != '\0' && isspace((unsigned char)*s))
		s++;
	return s;
}

static void stage4e_trim_in_place(char *s)
{
	char *start;
	char *end;

	if (s == NULL || s[0] == '\0')
		return;
	start = s;
	while (*start != '\0' && isspace((unsigned char)*start))
		start++;
	if (start != s)
		memmove(s, start, strlen(start) + 1U);
	end = s + strlen(s);
	while (end > s && isspace((unsigned char)end[-1]))
		end--;
	*end = '\0';
}

static int stage4e_append_line(char *dst, size_t dst_cap, const char *line)
{
	size_t used = strlen(dst);
	size_t len = strlen(line);
	if (used + len + 2U > dst_cap)
		return 0;
	memcpy(dst + used, line, len);
	dst[used + len] = '\n';
	dst[used + len + 1U] = '\0';
	return 1;
}

static void stage4e_dataset_free(stage4e_dataset_t *ds)
{
	size_t i;

	for (i = 0; i < ds->count; i++) {
		free(ds->items[i].prompt_tokens);
		free(ds->items[i].training_seq_tokens);
	}
	free(ds->items);
	memset(ds, 0, sizeof(*ds));
}

static int stage4e_dataset_push(stage4e_dataset_t *ds, const stage4e_record_t *rec)
{
	stage4e_record_t *new_items;

	if (ds->count == ds->cap) {
		size_t new_cap = ds->cap == 0 ? 16U : ds->cap * 2U;
		new_items = realloc(ds->items, new_cap * sizeof(stage4e_record_t));
		if (new_items == NULL)
			return 0;
		ds->items = new_items;
		ds->cap = new_cap;
	}
	ds->items[ds->count++] = *rec;
	return 1;
}

static int stage4e_set_field(stage4e_record_t *rec,
	const char *key,
	const char *value,
	int line_no)
{
	double parsed;
	char *endp = NULL;

	if (strcmp(key, "id") == 0) {
		if (rec->id[0] != '\0') {
			fprintf(stderr, "error:%d duplicate id\n", line_no);
			return 0;
		}
		if (value[0] == '\0') {
			fprintf(stderr, "error:%d empty id\n", line_no);
			return 0;
		}
		strncpy(rec->id, value, sizeof(rec->id) - 1U);
		return 1;
	}
	if (strcmp(key, "task") == 0) {
		if (rec->task[0] != '\0') {
			fprintf(stderr, "error:%d duplicate task\n", line_no);
			return 0;
		}
		strncpy(rec->task, value, sizeof(rec->task) - 1U);
		return 1;
	}
	if (strcmp(key, "split") == 0) {
		if (rec->split[0] != '\0') {
			fprintf(stderr, "error:%d duplicate split\n", line_no);
			return 0;
		}
		if (stage4e_split_index(value) < 0) {
			fprintf(stderr, "error:%d invalid split '%s'\n", line_no, value);
			return 0;
		}
		strncpy(rec->split, value, sizeof(rec->split) - 1U);
		return 1;
	}
	if (strcmp(key, "weight") == 0) {
		if (rec->weight > 0.0) {
			fprintf(stderr, "error:%d duplicate weight\n", line_no);
			return 0;
		}
		errno = 0;
		parsed = strtod(value, &endp);
		if (errno != 0 || endp == value || *endp != '\0' || !(parsed > 0.0) ||
		    !isfinite(parsed)) {
			fprintf(stderr, "error:%d invalid weight '%s'\n", line_no, value);
			return 0;
		}
		rec->weight = parsed;
		return 1;
	}
	if (strcmp(key, "source") == 0 || strcmp(key, "source_unit") == 0)
		return 1;
	fprintf(stderr, "error:%d unknown metadata field '%s'\n", line_no, key);
	return 0;
}

static int stage4e_record_validate_required(const stage4e_record_t *rec)
{
	if (rec->id[0] == '\0' || rec->task[0] == '\0' || rec->split[0] == '\0' ||
	    !(rec->weight > 0.0))
		return 0;
	if (rec->prompt[0] == '\0' || rec->target_visible[0] == '\0')
		return 0;
	if (stage4e_task_index(rec->task) < 0)
		return 0;
	return 1;
}

static int stage4e_load_records(const char *path, stage4e_dataset_t *ds)
{
	FILE *f;
	char line[STAGE4E_MAX_LINE];
	int line_no = 0;
	int section = STAGE4E_SECTION_OUTSIDE;
	stage4e_record_t cur;
	int target_lines = 0;

	memset(ds, 0, sizeof(*ds));
	memset(&cur, 0, sizeof(cur));

	f = fopen(path, "rb");
	if (f == NULL) {
		fprintf(stderr, "error: opening dataset '%s': %s\n", path, strerror(errno));
		return 0;
	}

	while (fgets(line, sizeof(line), f) != NULL) {
		char *t;
		const char *start;
		line_no++;

		if (strchr(line, '\n') == NULL && !feof(f)) {
			fprintf(stderr, "error:%d line exceeds parser limit\n", line_no);
			fclose(f);
			stage4e_dataset_free(ds);
			return 0;
		}

		stage4e_rtrim_newline(line);
		t = line;
		start = stage4e_skip_ws(t);

		if (section == STAGE4E_SECTION_OUTSIDE) {
			if (*start == '\0' || *start == '#')
				continue;
			if (strcmp(start, "%%BEGIN") == 0) {
				memset(&cur, 0, sizeof(cur));
				cur.begin_line = line_no;
				target_lines = 0;
				section = STAGE4E_SECTION_HEADER;
				continue;
			}
			fprintf(stderr, "error:%d expected %%BEGIN\n", line_no);
			fclose(f);
			stage4e_dataset_free(ds);
			return 0;
		}

		if (section == STAGE4E_SECTION_HEADER) {
			char key[STAGE4E_MAX_FIELD];
			char value[STAGE4E_MAX_TEXT];
			const char *eq;
			size_t key_len;

			if (strcmp(start, "%%PROMPT") == 0) {
				section = STAGE4E_SECTION_PROMPT;
				continue;
			}
			if (*start == '\0')
				continue;

			eq = strchr(start, '=');
			if (eq == NULL) {
				fprintf(stderr, "error:%d expected key=value metadata\n", line_no);
				fclose(f);
				stage4e_dataset_free(ds);
				return 0;
			}
			key_len = (size_t)(eq - start);
			if (key_len == 0 || key_len >= sizeof(key)) {
				fprintf(stderr, "error:%d malformed metadata key\n", line_no);
				fclose(f);
				stage4e_dataset_free(ds);
				return 0;
			}
			memcpy(key, start, key_len);
			key[key_len] = '\0';
			strncpy(value, eq + 1, sizeof(value) - 1U);
			value[sizeof(value) - 1U] = '\0';
			stage4e_trim_in_place(key);
			stage4e_trim_in_place(value);
			if (!stage4e_set_field(&cur, key, value, line_no)) {
				fclose(f);
				stage4e_dataset_free(ds);
				return 0;
			}
			continue;
		}

		if (section == STAGE4E_SECTION_PROMPT) {
			if (strcmp(start, "%%TARGET") == 0) {
				section = STAGE4E_SECTION_TARGET;
				continue;
			}
			if (!stage4e_append_line(cur.prompt, sizeof(cur.prompt), start)) {
				fprintf(stderr, "error:%d prompt too long\n", line_no);
				fclose(f);
				stage4e_dataset_free(ds);
				return 0;
			}
			continue;
		}

		if (section == STAGE4E_SECTION_TARGET) {
			if (strcmp(start, "%%END") == 0) {
				stage4e_record_t finalized;
				memset(&finalized, 0, sizeof(finalized));

				stage4e_trim_in_place(cur.prompt);
				stage4e_trim_in_place(cur.target_visible);

				if (!stage4e_record_validate_required(&cur)) {
					fprintf(stderr, "error:%d missing required fields\n", line_no);
					fclose(f);
					stage4e_dataset_free(ds);
					return 0;
				}
				if (target_lines != 1) {
					fprintf(stderr, "error:%d visible target must be single-line\n", line_no);
					fclose(f);
					stage4e_dataset_free(ds);
					return 0;
				}

				finalized = cur;
				if (!stage4e_dataset_push(ds, &finalized)) {
					fprintf(stderr, "error:%d cannot store record\n", line_no);
					fclose(f);
					stage4e_dataset_free(ds);
					return 0;
				}

				memset(&cur, 0, sizeof(cur));
				target_lines = 0;
				section = STAGE4E_SECTION_OUTSIDE;
				continue;
			}

			if (start[0] == '\0')
				continue;
			target_lines++;
			if (target_lines == 1)
				strncpy(cur.target_visible, start, sizeof(cur.target_visible) - 1U);
			continue;
		}
	}

	if (ferror(f)) {
		fprintf(stderr, "error:%d reading dataset failed\n", line_no);
		fclose(f);
		stage4e_dataset_free(ds);
		return 0;
	}
	if (section != STAGE4E_SECTION_OUTSIDE) {
		fprintf(stderr, "error:%d unterminated record\n", line_no);
		fclose(f);
		stage4e_dataset_free(ds);
		return 0;
	}
	if (fclose(f) != 0) {
		fprintf(stderr, "error: closing dataset failed\n");
		stage4e_dataset_free(ds);
		return 0;
	}

	return 1;
}

static int stage4e_prepare_record_tokens(Tokenizer *tok,
	const Transformer *tr,
	stage4e_record_t *r)
{
	if (snprintf(r->canonical_prompt, sizeof(r->canonical_prompt),
	    "User: %s\nAssistant:", r->prompt) >= (int)sizeof(r->canonical_prompt)) {
		fprintf(stderr, "error:%d canonical prompt too long for id='%s'\n",
		    r->begin_line, r->id);
		return 0;
	}
	if (snprintf(r->canonical_training_seq, sizeof(r->canonical_training_seq),
	    "User: %s\nAssistant: %s\n", r->prompt, r->target_visible) >=
	    (int)sizeof(r->canonical_training_seq)) {
		fprintf(stderr, "error:%d canonical sequence too long for id='%s'\n",
		    r->begin_line, r->id);
		return 0;
	}

	if (!encode_text(tok, r->canonical_prompt, 1, 0,
	    &r->prompt_tokens, &r->prompt_count) ||
	    !encode_text(tok, r->canonical_training_seq, 1, 0,
	    &r->training_seq_tokens, &r->training_seq_count)) {
		fprintf(stderr, "error:%d tokenization failed for id='%s'\n",
		    r->begin_line, r->id);
		return 0;
	}

	if (r->training_seq_count > tr->config.seq_len) {
		fprintf(stderr,
		    "error:%d record id='%s' exceeds context (%d > %d)\n",
		    r->begin_line, r->id,
		    r->training_seq_count, tr->config.seq_len);
		return 0;
	}

	r->first_target_index = r->prompt_count;
	if (r->first_target_index <= 0 || r->first_target_index >= r->training_seq_count) {
		fprintf(stderr,
		    "error:%d record id='%s' has invalid target boundary\n",
		    r->begin_line, r->id);
		return 0;
	}

	r->target_token_count = r->training_seq_count - r->first_target_index;
	if (r->target_token_count <= 0) {
		fprintf(stderr,
		    "error:%d record id='%s' has empty target token window\n",
		    r->begin_line, r->id);
		return 0;
	}

	return 1;
}

static int stage4e_loss_from_logits(const float *logits,
	int vocab_size,
	int target,
	double *loss_out)
{
	double max_logit;
	double sum = 0.0;
	double logprob;
	int i;

	if (target < 0 || target >= vocab_size)
		return 0;

	max_logit = logits[0];
	for (i = 1; i < vocab_size; i++) {
		if (!isfinite(logits[i]))
			return 0;
		if (logits[i] > max_logit)
			max_logit = logits[i];
	}
	if (!isfinite(logits[0]))
		return 0;

	for (i = 0; i < vocab_size; i++)
		sum += exp((double)logits[i] - max_logit);
	if (!(sum > 0.0) || !isfinite(sum))
		return 0;

	logprob = (double)logits[target] - max_logit - log(sum);
	*loss_out = -logprob;
	return isfinite(*loss_out);
}

static int stage4e_argmax_token(const float *logits, int vocab)
{
	int i;
	int best = 0;
	float bestv = logits[0];
	for (i = 1; i < vocab; i++) {
		if (logits[i] > bestv) {
			bestv = logits[i];
			best = i;
		}
	}
	return best;
}

#ifdef STAGE4E_FULL_CONTEXT_TOP1
static int stage4e_top1_record0_context_check(Transformer *tr,
	Tokenizer *tok,
	stage4e_dataset_t *ds);
#endif

int main(int argc, char **argv)
{
	const char *checkpoint_path = NULL;
	const char *dataset_path = NULL;
	const char *tokenizer_path = NULL;
	const char *report_path = "stage4e-baseline-unadapted-top1-report.txt";
	int verbose = 0;
	int argi;
	stage4e_dataset_t ds;
	Transformer tr;
	Tokenizer tok;
	size_t i;
	char dataset_sha[65];
	uint64_t dataset_bytes = 0;
	char base_sha[65];
	char tokenizer_sha[65];
	FILE *report;
	stage4e_metric_t total;
	stage4e_metric_t split_metrics[STAGE4E_SPLIT_COUNT];
	stage4e_metric_t split_task_metrics[STAGE4E_SPLIT_COUNT][STAGE4E_TASK_COUNT];
#ifdef STAGE4E_FULL_CONTEXT_TOP1
	int record0_context_check = 0;
#endif

	memset(&ds, 0, sizeof(ds));
	memset(&tr, 0, sizeof(tr));
	memset(&tok, 0, sizeof(tok));
	memset(&total, 0, sizeof(total));
	memset(split_metrics, 0, sizeof(split_metrics));
	memset(split_task_metrics, 0, sizeof(split_task_metrics));

	if (argc < 2)
		stage4e_usage(argv[0]);
	checkpoint_path = argv[1];

	for (argi = 2; argi < argc; argi++) {
		if (strcmp(argv[argi], "-d") == 0 && argi + 1 < argc)
			dataset_path = argv[++argi];
		else if (strcmp(argv[argi], "-z") == 0 && argi + 1 < argc)
			tokenizer_path = argv[++argi];
		else if (strcmp(argv[argi], "-o") == 0 && argi + 1 < argc)
			report_path = argv[++argi];
		else if (strcmp(argv[argi], "-v") == 0)
			verbose = 1;
#ifdef STAGE4E_FULL_CONTEXT_TOP1
		else if (strcmp(argv[argi], "--record0-context-check") == 0)
			record0_context_check = 1;
#endif
		else
			stage4e_usage(argv[0]);
	}

	if (dataset_path == NULL || tokenizer_path == NULL)
		stage4e_usage(argv[0]);

	if (!stage4e_load_records(dataset_path, &ds))
		return EXIT_FAILURE;
	if (ds.count == 0) {
		fprintf(stderr, "error: dataset has zero records\n");
		stage4e_dataset_free(&ds);
		return EXIT_FAILURE;
	}
	if (!stage4_hash_file_sha256_hex(dataset_path, dataset_sha) ||
	    !stage4_file_size_bytes(dataset_path, &dataset_bytes) ||
	    !stage4_hash_file_sha256_hex(checkpoint_path, base_sha) ||
	    !stage4_hash_file_sha256_hex(tokenizer_path, tokenizer_sha)) {
		fprintf(stderr, "error: cannot compute artifact identities\n");
		stage4e_dataset_free(&ds);
		return EXIT_FAILURE;
	}

	load_transformer(&tr, checkpoint_path);
	load_tokenizer(&tok, tokenizer_path, tr.config.vocab_size);

#ifdef STAGE4E_FULL_CONTEXT_TOP1
	if (record0_context_check) {
		int passed = stage4e_top1_record0_context_check(&tr, &tok, &ds);
		stage4e_dataset_free(&ds);
		free_tokenizer(&tok);
		free_transformer(&tr);
		printf("record0_prompt_context_check=%s\n", passed ? "PASS" : "FAIL");
		return passed ? EXIT_SUCCESS : EXIT_FAILURE;
	}
#endif

	for (i = 0; i < ds.count; i++) {
		stage4e_record_t *r = &ds.items[i];
		int split_idx;
		int task_idx;
		int k;
		double rec_loss = 0.0;
		uint64_t rec_pred = 0;
		uint64_t rec_top1 = 0;

		if (!stage4e_prepare_record_tokens(&tok, &tr, r))
			goto fail;
		split_idx = stage4e_split_index(r->split);
		task_idx = stage4e_task_index(r->task);
		if (split_idx < 0 || task_idx < 0) {
			fprintf(stderr, "error:%d unknown split/task for id='%s'\n", r->begin_line, r->id);
			goto fail;
		}

		clear_run_state(&tr);
		for (k = 1; k < r->training_seq_count; k++) {
			const float *logits;
			double loss;
			int target;
			int pred;

#ifdef STAGE4E_FULL_CONTEXT_TOP1
			logits = forward(&tr, r->training_seq_tokens[k - 1], k - 1);
			if (k < r->first_target_index)
				continue;
#else
			if (k < r->first_target_index)
				continue;
			logits = forward(&tr, r->training_seq_tokens[k - 1], k - 1);
#endif
			target = r->training_seq_tokens[k];
			if (!stage4e_loss_from_logits(logits, tr.config.vocab_size, target, &loss)) {
				fprintf(stderr, "error:%d non-finite loss for id='%s'\n", r->begin_line, r->id);
				goto fail;
			}
			pred = stage4e_argmax_token(logits, tr.config.vocab_size);
			if (pred == target)
				rec_top1++;
			rec_loss += loss;
			rec_pred++;
		}
		if (rec_pred == 0) {
			fprintf(stderr, "error:%d zero target predictions for id='%s'\n", r->begin_line, r->id);
			goto fail;
		}

		total.records++;
		total.pred_tokens += rec_pred;
		total.loss_sum += rec_loss;
		total.top1_correct += rec_top1;
		split_metrics[split_idx].records++;
		split_metrics[split_idx].pred_tokens += rec_pred;
		split_metrics[split_idx].loss_sum += rec_loss;
		split_metrics[split_idx].top1_correct += rec_top1;
		split_task_metrics[split_idx][task_idx].records++;
		split_task_metrics[split_idx][task_idx].pred_tokens += rec_pred;
		split_task_metrics[split_idx][task_idx].loss_sum += rec_loss;
		split_task_metrics[split_idx][task_idx].top1_correct += rec_top1;

		if (verbose) {
			printf("record id=%s split=%s task=%s target_pred=%llu avg_loss=%.12f top1=%.12f\n",
			    r->id,
			    r->split,
			    r->task,
			    (unsigned long long)rec_pred,
			    rec_loss / (double)rec_pred,
			    (double)rec_top1 / (double)rec_pred);
		}
	}

	report = fopen(report_path, "wb");
	if (report == NULL) {
		fprintf(stderr, "error: cannot write report '%s': %s\n", report_path, strerror(errno));
		goto fail;
	}

#ifdef STAGE4E_FULL_CONTEXT_TOP1
	fprintf(report, "mode=stage4e_unadapted_top1_full_context_baseline\n");
#else
	fprintf(report, "mode=stage4e_unadapted_top1_baseline\n");
#endif
	fprintf(report, "dataset_path=%s\n", dataset_path);
	fprintf(report, "dataset_sha256=%s\n", dataset_sha);
	fprintf(report, "dataset_bytes=%llu\n", (unsigned long long)dataset_bytes);
	fprintf(report, "base_checkpoint_sha256=%s\n", base_sha);
	fprintf(report, "base_checkpoint_sha256_expected=%s\n", STAGE4E_EXPECTED_BASE_SHA256);
	fprintf(report, "tokenizer_sha256=%s\n", tokenizer_sha);
	fprintf(report, "tokenizer_sha256_expected=%s\n", STAGE4E_EXPECTED_TOKENIZER_SHA256);
	fprintf(report, "adapter_loaded=no\n");
	fprintf(report, "evaluation_updates_parameters=no\n");
	fprintf(report, "test_split_used_for_selection=no\n");
#ifdef STAGE4E_FULL_CONTEXT_TOP1
	{
		double validation_average_loss = split_metrics[1].loss_sum /
		    (double)split_metrics[1].pred_tokens;
		fprintf(report, "validation_baseline_loss=%.12f\n", validation_average_loss);
		fprintf(report, "validation_required_relative_improvement=0.05\n");
		fprintf(report, "validation_acceptance_loss_max=%.12f\n",
		    validation_average_loss * 0.95);
	}
#else
	fprintf(report, "validation_baseline_loss=9.153772893016\n");
	fprintf(report, "validation_required_relative_improvement=0.05\n");
	fprintf(report, "validation_acceptance_loss_max=8.696084248365\n");
#endif
	fprintf(report, "total_records=%llu\n", (unsigned long long)total.records);
	fprintf(report, "target_prediction_count=%llu\n", (unsigned long long)total.pred_tokens);
	fprintf(report, "total_target_loss=%.12f\n", total.loss_sum);
	fprintf(report, "average_target_loss=%.12f\n", total.loss_sum / (double)total.pred_tokens);
	fprintf(report, "target_perplexity=%.12f\n", exp(total.loss_sum / (double)total.pred_tokens));
	fprintf(report, "target_top1_correct=%llu\n", (unsigned long long)total.top1_correct);
	fprintf(report, "target_top1_accuracy=%.12f\n", (double)total.top1_correct / (double)total.pred_tokens);

	for (i = 0; i < STAGE4E_SPLIT_COUNT; i++) {
		if (split_metrics[i].pred_tokens == 0)
			continue;
		fprintf(report, "split.%s.records=%llu\n",
		    stage4e_split_names[i],
		    (unsigned long long)split_metrics[i].records);
		fprintf(report, "split.%s.target_prediction_count=%llu\n",
		    stage4e_split_names[i],
		    (unsigned long long)split_metrics[i].pred_tokens);
		fprintf(report, "split.%s.total_target_loss=%.12f\n",
		    stage4e_split_names[i],
		    split_metrics[i].loss_sum);
		fprintf(report, "split.%s.average_target_loss=%.12f\n",
		    stage4e_split_names[i],
		    split_metrics[i].loss_sum / (double)split_metrics[i].pred_tokens);
		fprintf(report, "split.%s.target_perplexity=%.12f\n",
		    stage4e_split_names[i],
		    exp(split_metrics[i].loss_sum / (double)split_metrics[i].pred_tokens));
		fprintf(report, "split.%s.target_top1_correct=%llu\n",
		    stage4e_split_names[i],
		    (unsigned long long)split_metrics[i].top1_correct);
		fprintf(report, "split.%s.target_top1_accuracy=%.12f\n",
		    stage4e_split_names[i],
		    (double)split_metrics[i].top1_correct / (double)split_metrics[i].pred_tokens);
	}

	for (i = 0; i < STAGE4E_SPLIT_COUNT; i++) {
		size_t j;
		for (j = 0; j < STAGE4E_TASK_COUNT; j++) {
			stage4e_metric_t *m = &split_task_metrics[i][j];
			if (m->pred_tokens == 0)
				continue;
			fprintf(report, "split.%s.task.%s.records=%llu\n",
			    stage4e_split_names[i], stage4e_task_names[j],
			    (unsigned long long)m->records);
			fprintf(report, "split.%s.task.%s.target_prediction_tokens=%llu\n",
			    stage4e_split_names[i], stage4e_task_names[j],
			    (unsigned long long)m->pred_tokens);
			fprintf(report, "split.%s.task.%s.average_target_loss=%.12f\n",
			    stage4e_split_names[i], stage4e_task_names[j],
			    m->loss_sum / (double)m->pred_tokens);
			fprintf(report, "split.%s.task.%s.target_top1_accuracy=%.12f\n",
			    stage4e_split_names[i], stage4e_task_names[j],
			    (double)m->top1_correct / (double)m->pred_tokens);
		}
	}

	if (fclose(report) != 0) {
		fprintf(stderr, "error: closing report failed\n");
		goto fail;
	}

	printf("mode=stage4e_unadapted_top1_baseline\n");
	printf("dataset_sha256=%s\n", dataset_sha);
	printf("target_prediction_count=%llu\n", (unsigned long long)total.pred_tokens);
	printf("target_top1_correct=%llu\n", (unsigned long long)total.top1_correct);
	printf("target_top1_accuracy=%.12f\n", (double)total.top1_correct / (double)total.pred_tokens);
	printf("report_path=%s\n", report_path);

	stage4e_dataset_free(&ds);
	free_tokenizer(&tok);
	free_transformer(&tr);
	return EXIT_SUCCESS;

fail:
	stage4e_dataset_free(&ds);
	free_tokenizer(&tok);
	free_transformer(&tr);
	return EXIT_FAILURE;
}

#ifdef STAGE4E_FULL_CONTEXT_TOP1
static uint64_t stage4e_top1_kv_checksum(const Transformer *tr, int last_position)
{
	const Config *config = &tr->config;
	const RunState *state = &tr->state;
	size_t kv_dim = ((size_t)config->dim * (size_t)config->n_kv_heads) /
	    (size_t)config->n_heads;
	uint64_t hash = UINT64_C(1469598103934665603);
	int layer;
	int position;
	size_t index;

	for (layer = 0; layer < config->n_layers; layer++) {
		size_t layer_offset = (size_t)layer * (size_t)config->seq_len * kv_dim;
		for (position = 0; position <= last_position; position++) {
			size_t position_offset = layer_offset + (size_t)position * kv_dim;
			for (index = 0; index < kv_dim; index++) {
				uint32_t key_bits;
				uint32_t value_bits;
				memcpy(&key_bits, &state->key_cache[position_offset + index], 4U);
				memcpy(&value_bits, &state->value_cache[position_offset + index], 4U);
				hash ^= key_bits;
				hash *= UINT64_C(1099511628211);
				hash ^= value_bits;
				hash *= UINT64_C(1099511628211);
			}
		}
	}
	return hash;
}

static int stage4e_top1_record0_context_check(Transformer *tr,
	Tokenizer *tok,
	stage4e_dataset_t *ds)
{
	stage4e_record_t *record;
	int position;
	int calls = 0;
	uint64_t checksum;

	if (ds->count == 0 || strcmp(ds->items[0].id,
	    "svc-status-basic-train-001") != 0) {
		fprintf(stderr, "error: approved record 0 identity mismatch\n");
		return 0;
	}
	record = &ds->items[0];
	if (!stage4e_prepare_record_tokens(tok, tr, record))
		return 0;
	clear_run_state(tr);
	for (position = 0; position < 15; position++) {
		(void)forward(tr, record->training_seq_tokens[position], position);
		calls++;
	}
	checksum = stage4e_top1_kv_checksum(tr, 14);
	printf("record0.sequence_count=%d\n", record->training_seq_count);
	printf("record0.first_target_index=%d\n", record->first_target_index);
	printf("record0.forward_calls_before_position_15=%d\n", calls);
	printf("record0.kv_checksum_before_position_15=%016llx\n",
	    (unsigned long long)checksum);
	(void)forward(tr, record->training_seq_tokens[15], 15);
	calls++;
	printf("record0.forward_calls_before_first_target=%d\n", calls);
	printf("record0.first_supervised_input_position=15\n");
	printf("record0.first_supervised_target_token_id=%d\n",
	    record->training_seq_tokens[16]);
	return record->training_seq_count == 19 &&
	    record->first_target_index == 16 && calls == 16 &&
	    record->training_seq_tokens[16] == 2669 &&
	    checksum == UINT64_C(0xabb568d3ab418288);
}
#endif
