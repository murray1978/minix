#include "stage4_adapter_train.h"

#include <ctype.h>
#include <errno.h>
#include <float.h>
#include <limits.h>
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

typedef struct {
	char id[STAGE4E_MAX_FIELD];
	char task[STAGE4E_MAX_FIELD];
	char source[STAGE4E_MAX_FIELD];
	char source_unit[STAGE4E_MAX_FIELD];
	char split[STAGE4E_MAX_FIELD];
	double weight;
	char prompt[STAGE4E_MAX_TEXT];
	char target_visible[STAGE4E_MAX_TEXT];
	char canonical_prompt[STAGE4E_MAX_TEXT * 2];
	char canonical_visible_seq[STAGE4E_MAX_TEXT * 2];
	char canonical_training_seq[STAGE4E_MAX_TEXT * 2];
	int begin_line;
	int target_line;
	int target_end_line;
	int *prompt_tokens;
	int prompt_count;
	int *visible_seq_tokens;
	int visible_seq_count;
	int *training_seq_tokens;
	int training_seq_count;
	int visible_target_token_count;
	int training_target_token_count;
	int terminator_token_count;
	int terminator_token_id;
	int first_target_index;
} stage4e_record_t;

typedef struct {
	stage4e_record_t *items;
	size_t count;
	size_t cap;
} stage4e_dataset_t;

typedef struct {
	const char *name;
	int count;
} stage4e_count_entry_t;

enum {
	STAGE4E_SECTION_OUTSIDE = 0,
	STAGE4E_SECTION_HEADER,
	STAGE4E_SECTION_PROMPT,
	STAGE4E_SECTION_TARGET
};

static void stage4e_usage(const char *prog) NORETURN;

static void stage4e_usage(const char *prog)
{
	fprintf(stderr,
	    "Usage: %s -d <stage4e-approved.records> -z <tokenizer.bin> [options]\n"
	    "\n"
	    "Options:\n"
	    "  -c <checkpoint.bin> model checkpoint for context-length checks\n"
	    "                     (default ../stories15M.bin)\n"
	    "  -o <report.txt>   machine-readable report path\n"
	    "                     (default stage4e-dataset-report.txt)\n"
	    "  -v                verbose per-record output\n"
	    "\n"
	    "Record format:\n"
	    "  %%BEGIN\n"
	    "  id=...\n"
	    "  task=...\n"
	    "  source=...\n"
	    "  source_unit=...\n"
	    "  split=train|validation|test|regression\n"
	    "  weight=...\n"
	    "\n"
	    "  %%PROMPT\n"
	    "  ...\n"
	    "\n"
	    "  %%TARGET\n"
	    "  single visible line\n"
	    "\n"
	    "  %%END\n",
	    prog);
	exit(EXIT_FAILURE);
}

static int stage4e_is_valid_split(const char *split)
{
	return strcmp(split, "train") == 0 ||
	    strcmp(split, "validation") == 0 ||
	    strcmp(split, "test") == 0 ||
	    strcmp(split, "regression") == 0;
}

static int stage4e_is_valid_task(const char *task)
{
	return strcmp(task, "command") == 0 ||
	    strcmp(task, "explanation") == 0 ||
	    strcmp(task, "source_navigation") == 0 ||
	    strcmp(task, "diagnosis") == 0 ||
	    strcmp(task, "command_review") == 0 ||
	    strcmp(task, "continuation") == 0;
}

static int stage4e_is_valid_source(const char *source)
{
	return strcmp(source, "project-authored") == 0 ||
	    strcmp(source, "observed-stage3-error") == 0 ||
	    strcmp(source, "observed-stage4-result") == 0 ||
	    strncmp(source, "man:", 4) == 0 ||
	    strncmp(source, "source:/", 8) == 0;
}

static const char *stage4e_source_type(const char *source)
{
	if (strcmp(source, "project-authored") == 0)
		return "project-authored";
	if (strcmp(source, "observed-stage3-error") == 0)
		return "observed-stage3-error";
	if (strcmp(source, "observed-stage4-result") == 0)
		return "observed-stage4-result";
	if (strncmp(source, "man:", 4) == 0)
		return "man";
	if (strncmp(source, "source:/", 8) == 0) {
		if (strrchr(source + 8, ':') != NULL)
			return "source-symbol";
		return "source-path";
	}
	return "unknown";
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

static void stage4e_dataset_free(stage4e_dataset_t *ds)
{
	size_t i;

	for (i = 0; i < ds->count; i++) {
		free(ds->items[i].prompt_tokens);
		free(ds->items[i].visible_seq_tokens);
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

static int stage4e_has_duplicate_field(const stage4e_record_t *rec, const char *key)
{
	if (strcmp(key, "id") == 0)
		return rec->id[0] != '\0';
	if (strcmp(key, "task") == 0)
		return rec->task[0] != '\0';
	if (strcmp(key, "source") == 0)
		return rec->source[0] != '\0';
	if (strcmp(key, "source_unit") == 0)
		return rec->source_unit[0] != '\0';
	if (strcmp(key, "split") == 0)
		return rec->split[0] != '\0';
	if (strcmp(key, "weight") == 0)
		return rec->weight > 0.0;
	return 0;
}

static int stage4e_set_field(stage4e_record_t *rec,
	const char *key,
	const char *value,
	int line_no)
{
	double parsed;
	char *endp = NULL;

	if (stage4e_has_duplicate_field(rec, key)) {
		fprintf(stderr, "error:%d duplicate field '%s'\n", line_no, key);
		return 0;
	}

	if (strcmp(key, "id") == 0) {
		if (value[0] == '\0') {
			fprintf(stderr, "error:%d empty id\n", line_no);
			return 0;
		}
		strncpy(rec->id, value, sizeof(rec->id) - 1U);
		return 1;
	}
	if (strcmp(key, "task") == 0) {
		if (!stage4e_is_valid_task(value)) {
			fprintf(stderr, "error:%d unknown task '%s'\n", line_no, value);
			return 0;
		}
		strncpy(rec->task, value, sizeof(rec->task) - 1U);
		return 1;
	}
	if (strcmp(key, "source") == 0) {
		if (!stage4e_is_valid_source(value)) {
			fprintf(stderr, "error:%d invalid source '%s'\n", line_no, value);
			return 0;
		}
		strncpy(rec->source, value, sizeof(rec->source) - 1U);
		return 1;
	}
	if (strcmp(key, "source_unit") == 0) {
		if (value[0] == '\0') {
			fprintf(stderr, "error:%d empty source_unit\n", line_no);
			return 0;
		}
		strncpy(rec->source_unit, value, sizeof(rec->source_unit) - 1U);
		return 1;
	}
	if (strcmp(key, "split") == 0) {
		if (!stage4e_is_valid_split(value)) {
			fprintf(stderr, "error:%d unknown split '%s'\n", line_no, value);
			return 0;
		}
		strncpy(rec->split, value, sizeof(rec->split) - 1U);
		return 1;
	}
	if (strcmp(key, "weight") == 0) {
		errno = 0;
		parsed = strtod(value, &endp);
		if (errno != 0 || endp == value || *endp != '\0' || !(parsed > 0.0) ||
		    !isfinite(parsed)) {
			fprintf(stderr, "error:%d non-positive or invalid weight '%s'\n", line_no, value);
			return 0;
		}
		rec->weight = parsed;
		return 1;
	}

	fprintf(stderr, "error:%d unknown metadata field '%s'\n", line_no, key);
	return 0;
}

static int stage4e_record_validate_required(const stage4e_record_t *rec)
{
	if (rec->id[0] == '\0' || rec->task[0] == '\0' || rec->source[0] == '\0' ||
	    rec->split[0] == '\0' || !(rec->weight > 0.0))
		return 0;
	if (rec->prompt[0] == '\0')
		return 0;
	if (rec->target_visible[0] == '\0')
		return 0;
	return 1;
}

static void stage4e_normalize_text(const char *src, char *dst, size_t cap)
{
	size_t out = 0;
	int last_space = 1;

	while (*src != '\0' && out + 1U < cap) {
		unsigned char c = (unsigned char)*src++;
		if (isalnum((int)c)) {
			dst[out++] = (char)tolower((int)c);
			last_space = 0;
		} else if (!last_space) {
			dst[out++] = ' ';
			last_space = 1;
		}
	}
	if (out > 0 && dst[out - 1] == ' ')
		out--;
	dst[out] = '\0';
}

static double stage4e_similarity_ratio(const char *a, const char *b)
{
	size_t la = strlen(a);
	size_t lb = strlen(b);
	size_t i;
	size_t j;
	size_t *dp;
	double ratio;
	size_t dist;
	size_t max_len = la > lb ? la : lb;

	if (max_len == 0)
		return 1.0;
	if (la > 1024 || lb > 1024)
		return 0.0;

	dp = calloc((lb + 1U) * (la + 1U), sizeof(size_t));
	if (dp == NULL)
		return 0.0;

	for (i = 0; i <= la; i++)
		dp[i * (lb + 1U)] = i;
	for (j = 0; j <= lb; j++)
		dp[j] = j;

	for (i = 1; i <= la; i++) {
		for (j = 1; j <= lb; j++) {
			size_t cost = (a[i - 1] == b[j - 1]) ? 0U : 1U;
			size_t del = dp[(i - 1U) * (lb + 1U) + j] + 1U;
			size_t ins = dp[i * (lb + 1U) + (j - 1U)] + 1U;
			size_t sub = dp[(i - 1U) * (lb + 1U) + (j - 1U)] + cost;
			size_t best = del < ins ? del : ins;
			if (sub < best)
				best = sub;
			dp[i * (lb + 1U) + j] = best;
		}
	}

	dist = dp[la * (lb + 1U) + lb];
	free(dp);
	ratio = 1.0 - ((double)dist / (double)max_len);
	if (ratio < 0.0)
		ratio = 0.0;
	return ratio;
}

static int stage4e_detect_cross_split_leak(const stage4e_dataset_t *ds)
{
	size_t i;
	size_t j;

	for (i = 0; i < ds->count; i++) {
		for (j = i + 1; j < ds->count; j++) {
			if (strcmp(ds->items[i].split, ds->items[j].split) != 0 &&
			    strcmp(ds->items[i].source_unit, ds->items[j].source_unit) == 0) {
				fprintf(stderr,
				    "error: repeated_source_unit_cross_split ids '%s' (line %d, split=%s) and '%s' (line %d, split=%s), source_unit='%s'\n",
				    ds->items[i].id, ds->items[i].begin_line, ds->items[i].split,
				    ds->items[j].id, ds->items[j].begin_line, ds->items[j].split,
				    ds->items[i].source_unit);
				return 0;
			}
		}
	}

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
			fprintf(stderr, "error:%d line exceeds parser limit (%d bytes)\n",
			    line_no, STAGE4E_MAX_LINE - 1);
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
			fprintf(stderr, "error:%d malformed section markers (expected %%BEGIN)\n", line_no);
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
				fprintf(stderr, "error:%d in record id='%s'\n", line_no,
				    cur.id[0] == '\0' ? "<pending>" : cur.id);
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
				fprintf(stderr, "error:%d prompt too long in record id='%s'\n",
				    line_no, cur.id);
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
					fprintf(stderr,
					    "error:%d missing required field or empty prompt/target in record id='%s'\n",
					    line_no, cur.id[0] == '\0' ? "<pending>" : cur.id);
					fclose(f);
					stage4e_dataset_free(ds);
					return 0;
				}
				if (target_lines != 1) {
					fprintf(stderr,
					    "error:%d record id='%s' must have a single-line visible target\n",
					    cur.target_end_line, cur.id);
					fclose(f);
					stage4e_dataset_free(ds);
					return 0;
				}
				if (cur.source_unit[0] == '\0') {
					strncpy(cur.source_unit, cur.source,
					    sizeof(cur.source_unit) - 1U);
				}

				finalized = cur;
				if (!stage4e_dataset_push(ds, &finalized)) {
					fprintf(stderr, "error:%d cannot store record id='%s'\n", line_no, cur.id);
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
			if (target_lines == 1) {
				strncpy(cur.target_visible, start, sizeof(cur.target_visible) - 1U);
				cur.target_line = line_no;
				cur.target_end_line = line_no;
			} else {
				cur.target_end_line = line_no;
			}
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
		fprintf(stderr, "error:%d malformed section markers (unterminated record)\n", line_no);
		fclose(f);
		stage4e_dataset_free(ds);
		return 0;
	}
	if (fclose(f) != 0) {
		fprintf(stderr, "error: closing dataset failed\n");
		stage4e_dataset_free(ds);
		return 0;
	}

	if (!stage4e_detect_cross_split_leak(ds)) {
		stage4e_dataset_free(ds);
		return 0;
	}

	return 1;
}

static int stage4e_prepare_record_tokens(Tokenizer *tok,
	const Transformer *tr,
	stage4e_record_t *r)
{
	int *newline_tokens = NULL;
	int newline_count = 0;

	if (snprintf(r->canonical_prompt, sizeof(r->canonical_prompt),
	    "User: %s\nAssistant:", r->prompt) >= (int)sizeof(r->canonical_prompt)) {
		fprintf(stderr, "error:%d record id='%s' canonical prompt too long\n",
		    r->begin_line, r->id);
		return 0;
	}
	if (snprintf(r->canonical_visible_seq, sizeof(r->canonical_visible_seq),
	    "User: %s\nAssistant: %s", r->prompt, r->target_visible) >=
	    (int)sizeof(r->canonical_visible_seq)) {
		fprintf(stderr, "error:%d record id='%s' canonical visible sequence too long\n",
		    r->begin_line, r->id);
		return 0;
	}
	if (snprintf(r->canonical_training_seq, sizeof(r->canonical_training_seq),
	    "User: %s\nAssistant: %s\n", r->prompt, r->target_visible) >=
	    (int)sizeof(r->canonical_training_seq)) {
		fprintf(stderr, "error:%d record id='%s' canonical training sequence too long\n",
		    r->begin_line, r->id);
		return 0;
	}

	if (!encode_text(tok, r->canonical_prompt, 1, 0,
	    &r->prompt_tokens, &r->prompt_count) ||
	    !encode_text(tok, r->canonical_visible_seq, 1, 0,
	    &r->visible_seq_tokens, &r->visible_seq_count) ||
	    !encode_text(tok, r->canonical_training_seq, 1, 0,
	    &r->training_seq_tokens, &r->training_seq_count)) {
		fprintf(stderr, "error:%d record id='%s' tokenization failed\n",
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
		    "error:%d record id='%s' target boundary outside token sequence\n",
		    r->begin_line, r->id);
		return 0;
	}

	r->visible_target_token_count = r->visible_seq_count - r->first_target_index;
	r->training_target_token_count = r->training_seq_count - r->first_target_index;
	r->terminator_token_count =
	    r->training_target_token_count - r->visible_target_token_count;

	if (r->visible_target_token_count <= 0 || r->training_target_token_count <= 0) {
		fprintf(stderr,
		    "error:%d record id='%s' has empty target token window\n",
		    r->begin_line, r->id);
		return 0;
	}

	if (!encode_text(tok, "\n", 0, 0, &newline_tokens, &newline_count)) {
		fprintf(stderr,
		    "error:%d record id='%s' newline terminator tokenization failed\n",
		    r->begin_line, r->id);
		return 0;
	}
	free(newline_tokens);

	if (r->terminator_token_count != 1) {
		fprintf(stderr,
		    "error:%d record id='%s' terminator token count=%d (expected 1)\n",
		    r->begin_line, r->id, r->terminator_token_count);
		return 0;
	}

	r->terminator_token_id = r->training_seq_tokens[r->training_seq_count - 1];
	return 1;
}

static int stage4e_add_count(stage4e_count_entry_t **items,
	size_t *count,
	size_t *cap,
	const char *name)
{
	size_t i;
	stage4e_count_entry_t *new_items;

	for (i = 0; i < *count; i++) {
		if (strcmp((*items)[i].name, name) == 0) {
			(*items)[i].count++;
			return 1;
		}
	}

	if (*count == *cap) {
		size_t new_cap = *cap == 0 ? 8U : (*cap * 2U);
		new_items = realloc(*items, new_cap * sizeof(stage4e_count_entry_t));
		if (new_items == NULL)
			return 0;
		*items = new_items;
		*cap = new_cap;
	}
	(*items)[*count].name = name;
	(*items)[*count].count = 1;
	(*count)++;
	return 1;
}

static int stage4e_count_duplicates(const char **values, size_t n)
{
	size_t i;
	size_t j;
	int dups = 0;

	for (i = 0; i < n; i++) {
		for (j = i + 1; j < n; j++) {
			if (strcmp(values[i], values[j]) == 0) {
				dups++;
				break;
			}
		}
	}
	return dups;
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

int main(int argc, char **argv)
{
	const char *dataset_path = NULL;
	const char *tokenizer_path = NULL;
	const char *checkpoint_path = "../stories15M.bin";
	const char *report_path = "stage4e-dataset-report.txt";
	int verbose = 0;
	int argi;
	stage4e_dataset_t ds;
	Transformer tr;
	Tokenizer tok;
	char dataset_sha[65];
	uint64_t dataset_bytes = 0;
	size_t i;
	uint64_t total_prompt_tokens = 0;
	uint64_t total_visible_target_tokens = 0;
	uint64_t total_training_target_tokens = 0;
	int min_seq_len = INT_MAX;
	int max_seq_len = 0;
	double avg_seq_len = 0.0;
	int distinct_visible_targets = 0;
	int distinct_first_target_tokens = 0;
	int duplicate_ids = 0;
	int duplicate_prompts = 0;
	int duplicate_targets = 0;
	int duplicate_target_same_split = 0;
	int duplicate_target_cross_split = 0;
	int repeated_source_units_cross_split = 0;
	int possible_cross_split_near_duplicates = 0;
	int records_with_newline_terminator = 0;
	int records_without_terminator = 0;
	int terminator_token_id = -1;
	int terminator_token_mismatches = 0;
	int multiline_visible_targets = 0;
	int empty_visible_targets = 0;
	int context_overflow_records = 0;
	int malformed_records = 0;
	int split_task_counts[4][6];
	int distinct_task_categories[4];
	const char *split_names[4] = { "train", "validation", "test", "regression" };
	const char *task_names[6] = {
		"command",
		"explanation",
		"source_navigation",
		"diagnosis",
		"command_review",
		"continuation"
	};
	stage4e_count_entry_t *split_counts = NULL;
	size_t split_count = 0;
	size_t split_cap = 0;
	stage4e_count_entry_t *task_counts = NULL;
	size_t task_count = 0;
	size_t task_cap = 0;
	stage4e_count_entry_t *source_counts = NULL;
	size_t source_count = 0;
	size_t source_cap = 0;
	stage4e_count_entry_t *source_unit_counts = NULL;
	size_t source_unit_count = 0;
	size_t source_unit_cap = 0;
	const char **ids = NULL;
	const char **prompts_norm = NULL;
	const char **targets_norm = NULL;
	char **prompt_norm_buf = NULL;
	char **target_norm_buf = NULL;
	FILE *report;

	memset(&ds, 0, sizeof(ds));
	memset(&tr, 0, sizeof(tr));
	memset(&tok, 0, sizeof(tok));
	memset(split_task_counts, 0, sizeof(split_task_counts));
	memset(distinct_task_categories, 0, sizeof(distinct_task_categories));

	for (argi = 1; argi < argc; argi++) {
		if (strcmp(argv[argi], "-d") == 0 && argi + 1 < argc)
			dataset_path = argv[++argi];
		else if (strcmp(argv[argi], "-z") == 0 && argi + 1 < argc)
			tokenizer_path = argv[++argi];
		else if (strcmp(argv[argi], "-c") == 0 && argi + 1 < argc)
			checkpoint_path = argv[++argi];
		else if (strcmp(argv[argi], "-o") == 0 && argi + 1 < argc)
			report_path = argv[++argi];
		else if (strcmp(argv[argi], "-v") == 0)
			verbose = 1;
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
	    !stage4_file_size_bytes(dataset_path, &dataset_bytes)) {
		fprintf(stderr, "error: cannot hash/stat dataset '%s'\n", dataset_path);
		stage4e_dataset_free(&ds);
		return EXIT_FAILURE;
	}

	load_transformer(&tr, checkpoint_path);
	load_tokenizer(&tok, tokenizer_path, tr.config.vocab_size);

	ids = calloc(ds.count, sizeof(char *));
	prompts_norm = calloc(ds.count, sizeof(char *));
	targets_norm = calloc(ds.count, sizeof(char *));
	prompt_norm_buf = calloc(ds.count, sizeof(char *));
	target_norm_buf = calloc(ds.count, sizeof(char *));
	if (ids == NULL || prompts_norm == NULL || targets_norm == NULL ||
	    prompt_norm_buf == NULL || target_norm_buf == NULL) {
		fprintf(stderr, "error: allocation failure\n");
		stage4e_dataset_free(&ds);
		free_tokenizer(&tok);
		free_transformer(&tr);
		free(ids);
		free(prompts_norm);
		free(targets_norm);
		free(prompt_norm_buf);
		free(target_norm_buf);
		return EXIT_FAILURE;
	}

	for (i = 0; i < ds.count; i++) {
		stage4e_record_t *r = &ds.items[i];
		char *p_norm;
		char *t_norm;
		int split_idx;
		int task_idx;

		if (!stage4e_prepare_record_tokens(&tok, &tr, r)) {
			malformed_records++;
			goto fail;
		}

		split_idx = stage4e_split_index(r->split);
		task_idx = stage4e_task_index(r->task);
		if (split_idx < 0 || task_idx < 0) {
			fprintf(stderr,
			    "error:%d internal split/task index failure for id='%s'\n",
			    r->begin_line, r->id);
			goto fail;
		}
		split_task_counts[split_idx][task_idx]++;

		if (!stage4e_add_count(&split_counts, &split_count, &split_cap, r->split) ||
		    !stage4e_add_count(&task_counts, &task_count, &task_cap, r->task) ||
		    !stage4e_add_count(&source_counts, &source_count, &source_cap,
		    stage4e_source_type(r->source)) ||
		    !stage4e_add_count(&source_unit_counts, &source_unit_count,
		    &source_unit_cap, r->source_unit)) {
			fprintf(stderr, "error: count aggregation failure\n");
			goto fail;
		}

		total_prompt_tokens += (uint64_t)r->prompt_count;
		total_visible_target_tokens += (uint64_t)r->visible_target_token_count;
		total_training_target_tokens += (uint64_t)r->training_target_token_count;

		if (r->terminator_token_count == 1)
			records_with_newline_terminator++;
		else
			records_without_terminator++;
		if (terminator_token_id < 0)
			terminator_token_id = r->terminator_token_id;
		else if (terminator_token_id != r->terminator_token_id)
			terminator_token_mismatches++;
		if (r->target_visible[0] == '\0')
			empty_visible_targets++;

		if (r->training_seq_count < min_seq_len)
			min_seq_len = r->training_seq_count;
		if (r->training_seq_count > max_seq_len)
			max_seq_len = r->training_seq_count;

		ids[i] = r->id;
		p_norm = calloc(STAGE4E_MAX_TEXT * 2U, 1);
		t_norm = calloc(STAGE4E_MAX_TEXT * 2U, 1);
		if (p_norm == NULL || t_norm == NULL) {
			free(p_norm);
			free(t_norm);
			fprintf(stderr, "error: normalization allocation failure\n");
			goto fail;
		}
		stage4e_normalize_text(r->prompt, p_norm, STAGE4E_MAX_TEXT * 2U);
		stage4e_normalize_text(r->target_visible, t_norm, STAGE4E_MAX_TEXT * 2U);
		prompt_norm_buf[i] = p_norm;
		target_norm_buf[i] = t_norm;
		prompts_norm[i] = p_norm;
		targets_norm[i] = t_norm;

		if (verbose) {
			int k;
			printf("record id=%s line=%d split=%s task=%s source=%s\n",
			    r->id, r->begin_line, r->split, r->task, r->source);
			printf("  visible_target_token_count=%d training_target_token_count=%d terminator_token_count=%d terminator_token_id=%d\n",
			    r->visible_target_token_count,
			    r->training_target_token_count,
			    r->terminator_token_count,
			    r->terminator_token_id);
			printf("  target_prediction_positions:");
			for (k = r->first_target_index; k < r->training_seq_count; k++)
				printf(" %d(input=%d->target=%d)", k, k - 1, k);
			printf("\n");
		}
	}

	avg_seq_len = (double)(total_training_target_tokens + total_prompt_tokens) /
	    (double)ds.count;
	distinct_visible_targets = (int)ds.count - stage4e_count_duplicates(targets_norm, ds.count);

	for (i = 0; i < 4; i++) {
		size_t j;
		int categories = 0;
		for (j = 0; j < 6; j++) {
			if (split_task_counts[i][j] > 0)
				categories++;
		}
		distinct_task_categories[i] = categories;
	}

	{
		int *first_tokens = calloc(ds.count, sizeof(int));
		size_t n = 0;
		if (first_tokens == NULL) {
			fprintf(stderr, "error: allocation failure\n");
			goto fail;
		}
		for (i = 0; i < ds.count; i++) {
			int token = ds.items[i].training_seq_tokens[ds.items[i].first_target_index];
			size_t j;
			int seen = 0;
			for (j = 0; j < n; j++) {
				if (first_tokens[j] == token) {
					seen = 1;
					break;
				}
			}
			if (!seen)
				first_tokens[n++] = token;
		}
		distinct_first_target_tokens = (int)n;
		free(first_tokens);
	}

	duplicate_ids = stage4e_count_duplicates(ids, ds.count);
	duplicate_prompts = stage4e_count_duplicates(prompts_norm, ds.count);
	duplicate_targets = stage4e_count_duplicates(targets_norm, ds.count);

	for (i = 0; i < ds.count; i++) {
		size_t j;
		for (j = i + 1; j < ds.count; j++) {
			double sim;

			if (strcmp(targets_norm[i], targets_norm[j]) == 0) {
				if (strcmp(ds.items[i].split, ds.items[j].split) == 0)
					duplicate_target_same_split++;
				else
					duplicate_target_cross_split++;
			}

			if (strcmp(ds.items[i].split, ds.items[j].split) != 0 &&
			    strcmp(ds.items[i].source_unit, ds.items[j].source_unit) == 0)
				repeated_source_units_cross_split++;

			if (strcmp(ds.items[i].split, ds.items[j].split) != 0) {
				sim = stage4e_similarity_ratio(prompts_norm[i], prompts_norm[j]);
				if (strcmp(prompts_norm[i], prompts_norm[j]) == 0 || sim >= 0.92 ||
				    (strcmp(targets_norm[i], targets_norm[j]) == 0 && sim >= 0.80))
					possible_cross_split_near_duplicates++;
			}
		}
	}

	report = fopen(report_path, "wb");
	if (report == NULL) {
		fprintf(stderr, "error: cannot write report '%s': %s\n",
		    report_path, strerror(errno));
		goto fail;
	}

	fprintf(report, "dataset_sha256=%s\n", dataset_sha);
	fprintf(report, "dataset_bytes=%llu\n", (unsigned long long)dataset_bytes);
	fprintf(report, "total_records=%lu\n", (unsigned long)ds.count);
	for (i = 0; i < split_count; i++)
		fprintf(report, "records_by_split.%s=%d\n", split_counts[i].name, split_counts[i].count);
	for (i = 0; i < task_count; i++)
		fprintf(report, "records_by_task.%s=%d\n", task_counts[i].name, task_counts[i].count);
	for (i = 0; i < source_count; i++)
		fprintf(report, "records_by_source_type.%s=%d\n", source_counts[i].name, source_counts[i].count);
	for (i = 0; i < source_unit_count; i++)
		fprintf(report, "records_by_source_unit.%s=%d\n", source_unit_counts[i].name, source_unit_counts[i].count);
	for (i = 0; i < 4; i++) {
		size_t j;
		for (j = 0; j < 6; j++) {
			fprintf(report,
			    "records_by_split_task.%s.%s=%d\n",
			    split_names[i],
			    task_names[j],
			    split_task_counts[i][j]);
		}
	}
	fprintf(report, "distinct_task_categories.train=%d\n", distinct_task_categories[0]);
	fprintf(report, "distinct_task_categories.validation=%d\n", distinct_task_categories[1]);
	fprintf(report, "distinct_task_categories.test=%d\n", distinct_task_categories[2]);
	fprintf(report, "total_prompt_tokens=%llu\n", (unsigned long long)total_prompt_tokens);
	fprintf(report, "total_visible_target_tokens=%llu\n", (unsigned long long)total_visible_target_tokens);
	fprintf(report, "total_training_target_tokens=%llu\n", (unsigned long long)total_training_target_tokens);
	fprintf(report, "minimum_sequence_length=%d\n", min_seq_len == INT_MAX ? 0 : min_seq_len);
	fprintf(report, "maximum_sequence_length=%d\n", max_seq_len);
	fprintf(report, "average_sequence_length=%.6f\n", avg_seq_len);
	fprintf(report, "distinct_visible_targets=%d\n", distinct_visible_targets);
	fprintf(report, "distinct_first_target_tokens=%d\n", distinct_first_target_tokens);
	fprintf(report, "duplicate_ids=%d\n", duplicate_ids);
	fprintf(report, "duplicate_prompts=%d\n", duplicate_prompts);
	fprintf(report, "duplicate_targets=%d\n", duplicate_targets);
	fprintf(report, "duplicate_target_same_split=%d\n", duplicate_target_same_split);
	fprintf(report, "duplicate_target_cross_split=%d\n", duplicate_target_cross_split);
	fprintf(report, "repeated_source_units_cross_split=%d\n", repeated_source_units_cross_split);
	fprintf(report, "possible_cross_split_near_duplicates=%d\n", possible_cross_split_near_duplicates);
	fprintf(report, "records_with_newline_terminator=%d\n", records_with_newline_terminator);
	fprintf(report, "records_without_terminator=%d\n", records_without_terminator);
	fprintf(report, "terminator_token_id=%d\n", terminator_token_id);
	fprintf(report, "terminator_token_mismatches=%d\n", terminator_token_mismatches);
	fprintf(report, "multiline_visible_targets=%d\n", multiline_visible_targets);
	fprintf(report, "empty_visible_targets=%d\n", empty_visible_targets);
	fprintf(report, "context_overflow_records=%d\n", context_overflow_records);
	fprintf(report, "malformed_records=%d\n", malformed_records);
	if (fclose(report) != 0) {
		fprintf(stderr, "error: closing report '%s' failed\n", report_path);
		goto fail;
	}

	printf("dataset_sha256=%s\n", dataset_sha);
	printf("total_records=%lu\n", (unsigned long)ds.count);
	printf("records_by_split:");
	for (i = 0; i < split_count; i++)
		printf(" %s=%d", split_counts[i].name, split_counts[i].count);
	printf("\n");
	printf("records_by_task:");
	for (i = 0; i < task_count; i++)
		printf(" %s=%d", task_counts[i].name, task_counts[i].count);
	printf("\n");
	printf("records_by_source_type:");
	for (i = 0; i < source_count; i++)
		printf(" %s=%d", source_counts[i].name, source_counts[i].count);
	printf("\n");
	printf("records_by_source_unit:");
	for (i = 0; i < source_unit_count; i++)
		printf(" %s=%d", source_unit_counts[i].name, source_unit_counts[i].count);
	printf("\n");
	for (i = 0; i < 4; i++) {
		size_t j;
		printf("records_by_split_task.%s:", split_names[i]);
		for (j = 0; j < 6; j++)
			printf(" %s=%d", task_names[j], split_task_counts[i][j]);
		printf("\n");
	}
	printf("distinct_task_categories.train=%d\n", distinct_task_categories[0]);
	printf("distinct_task_categories.validation=%d\n", distinct_task_categories[1]);
	printf("distinct_task_categories.test=%d\n", distinct_task_categories[2]);
	printf("total_prompt_tokens=%llu\n", (unsigned long long)total_prompt_tokens);
	printf("total_visible_target_tokens=%llu\n", (unsigned long long)total_visible_target_tokens);
	printf("total_training_target_tokens=%llu\n", (unsigned long long)total_training_target_tokens);
	printf("minimum_sequence_length=%d\n", min_seq_len == INT_MAX ? 0 : min_seq_len);
	printf("maximum_sequence_length=%d\n", max_seq_len);
	printf("average_sequence_length=%.6f\n", avg_seq_len);
	printf("distinct_visible_targets=%d\n", distinct_visible_targets);
	printf("distinct_first_target_tokens=%d\n", distinct_first_target_tokens);
	printf("duplicate_ids=%d\n", duplicate_ids);
	printf("duplicate_prompts=%d\n", duplicate_prompts);
	printf("duplicate_targets=%d\n", duplicate_targets);
	printf("duplicate_target_same_split=%d\n", duplicate_target_same_split);
	printf("duplicate_target_cross_split=%d\n", duplicate_target_cross_split);
	printf("repeated_source_units_cross_split=%d\n", repeated_source_units_cross_split);
	printf("possible_cross_split_near_duplicates=%d\n", possible_cross_split_near_duplicates);
	printf("records_with_newline_terminator=%d\n", records_with_newline_terminator);
	printf("records_without_terminator=%d\n", records_without_terminator);
	printf("terminator_token_id=%d\n", terminator_token_id);
	printf("terminator_token_mismatches=%d\n", terminator_token_mismatches);
	printf("multiline_visible_targets=%d\n", multiline_visible_targets);
	printf("empty_visible_targets=%d\n", empty_visible_targets);
	printf("context_overflow_records=%d\n", context_overflow_records);
	printf("malformed_records=%d\n", malformed_records);
	printf("report_path=%s\n", report_path);

	for (i = 0; i < ds.count; i++) {
		free(prompt_norm_buf[i]);
		free(target_norm_buf[i]);
	}
	free(prompt_norm_buf);
	free(target_norm_buf);
	free(ids);
	free(prompts_norm);
	free(targets_norm);
	free(split_counts);
	free(task_counts);
	free(source_counts);
	free(source_unit_counts);
	stage4e_dataset_free(&ds);
	free_tokenizer(&tok);
	free_transformer(&tr);
	return EXIT_SUCCESS;

fail:
	for (i = 0; i < ds.count; i++) {
		free(prompt_norm_buf != NULL ? prompt_norm_buf[i] : NULL);
		free(target_norm_buf != NULL ? target_norm_buf[i] : NULL);
	}
	free(prompt_norm_buf);
	free(target_norm_buf);
	free(ids);
	free(prompts_norm);
	free(targets_norm);
	free(split_counts);
	free(task_counts);
	free(source_counts);
	free(source_unit_counts);
	stage4e_dataset_free(&ds);
	free_tokenizer(&tok);
	free_transformer(&tr);
	return EXIT_FAILURE;
}
