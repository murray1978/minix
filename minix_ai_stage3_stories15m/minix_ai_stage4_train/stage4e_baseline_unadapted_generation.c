#include "stage4_adapter_train.h"

#include <ctype.h>
#include <errno.h>
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
#define STAGE4E_MAX_GEN_TOKENS 128

typedef struct {
	char id[STAGE4E_MAX_FIELD];
	char task[STAGE4E_MAX_FIELD];
	char split[STAGE4E_MAX_FIELD];
	double weight;
	char prompt[STAGE4E_MAX_TEXT];
	char target_visible[STAGE4E_MAX_TEXT];
	char canonical_prompt[STAGE4E_MAX_TEXT * 2];
	char canonical_visible_seq[STAGE4E_MAX_TEXT * 2];
	char canonical_training_seq[STAGE4E_MAX_TEXT * 2];
	int *prompt_tokens;
	int prompt_count;
	int *visible_seq_tokens;
	int visible_seq_count;
	int *training_seq_tokens;
	int training_seq_count;
	int first_target_index;
	int visible_target_count;
	int terminator_token_id;
	int begin_line;
} stage4e_record_t;

typedef struct {
	stage4e_record_t *items;
	size_t count;
	size_t cap;
} stage4e_dataset_t;

typedef struct {
	char key[1024];
	int count;
} stage4e_key_count_t;

typedef struct {
	stage4e_key_count_t *items;
	size_t count;
	size_t cap;
} stage4e_key_counts_t;

typedef struct {
	uint64_t records;
	uint64_t window_match;
	uint64_t term_match;
	uint64_t exact_match;
	stage4e_key_counts_t windows;
	stage4e_key_counts_t first_tokens;
} stage4e_gen_metric_t;

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
	    "                (default stage4e-baseline-unadapted-generation-report.txt)\n"
	    "  --max-visible-output-tokens <n>\n"
	    "                (default 64, max 128)\n"
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

static int stage4e_set_field(stage4e_record_t *rec,
	const char *key,
	const char *value,
	int line_no)
{
	double parsed;
	char *endp = NULL;

	if (strcmp(key, "id") == 0) {
		if (rec->id[0] != '\0')
			return 0;
		strncpy(rec->id, value, sizeof(rec->id) - 1U);
		return rec->id[0] != '\0';
	}
	if (strcmp(key, "task") == 0) {
		if (rec->task[0] != '\0')
			return 0;
		strncpy(rec->task, value, sizeof(rec->task) - 1U);
		return rec->task[0] != '\0';
	}
	if (strcmp(key, "split") == 0) {
		if (rec->split[0] != '\0')
			return 0;
		if (stage4e_split_index(value) < 0) {
			fprintf(stderr, "error:%d invalid split '%s'\n", line_no, value);
			return 0;
		}
		strncpy(rec->split, value, sizeof(rec->split) - 1U);
		return 1;
	}
	if (strcmp(key, "weight") == 0) {
		if (rec->weight > 0.0)
			return 0;
		errno = 0;
		parsed = strtod(value, &endp);
		if (errno != 0 || endp == value || *endp != '\0' || !(parsed > 0.0))
			return 0;
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
			if (eq == NULL)
				goto parse_fail;
			key_len = (size_t)(eq - start);
			if (key_len == 0 || key_len >= sizeof(key))
				goto parse_fail;
			memcpy(key, start, key_len);
			key[key_len] = '\0';
			strncpy(value, eq + 1, sizeof(value) - 1U);
			value[sizeof(value) - 1U] = '\0';
			stage4e_trim_in_place(key);
			stage4e_trim_in_place(value);
			if (!stage4e_set_field(&cur, key, value, line_no))
				goto parse_fail;
			continue;
		}

		if (section == STAGE4E_SECTION_PROMPT) {
			if (strcmp(start, "%%TARGET") == 0) {
				section = STAGE4E_SECTION_TARGET;
				continue;
			}
			if (!stage4e_append_line(cur.prompt, sizeof(cur.prompt), start))
				goto parse_fail;
			continue;
		}

		if (section == STAGE4E_SECTION_TARGET) {
			if (strcmp(start, "%%END") == 0) {
				stage4e_record_t finalized;
				memset(&finalized, 0, sizeof(finalized));
				stage4e_trim_in_place(cur.prompt);
				stage4e_trim_in_place(cur.target_visible);
				if (!stage4e_record_validate_required(&cur) || target_lines != 1)
					goto parse_fail;
				finalized = cur;
				if (!stage4e_dataset_push(ds, &finalized))
					goto parse_fail;
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

	if (ferror(f) || section != STAGE4E_SECTION_OUTSIDE)
		goto parse_fail;
	if (fclose(f) != 0)
		goto parse_fail_noclose;
	return 1;

parse_fail:
	fclose(f);
parse_fail_noclose:
	stage4e_dataset_free(ds);
	return 0;
}

static int stage4e_prepare_record_tokens(Tokenizer *tok,
	const Transformer *tr,
	stage4e_record_t *r)
{
	if (snprintf(r->canonical_prompt, sizeof(r->canonical_prompt),
	    "User: %s\nAssistant:", r->prompt) >= (int)sizeof(r->canonical_prompt))
		return 0;
	if (snprintf(r->canonical_visible_seq, sizeof(r->canonical_visible_seq),
	    "User: %s\nAssistant: %s", r->prompt, r->target_visible) >=
	    (int)sizeof(r->canonical_visible_seq))
		return 0;
	if (snprintf(r->canonical_training_seq, sizeof(r->canonical_training_seq),
	    "User: %s\nAssistant: %s\n", r->prompt, r->target_visible) >=
	    (int)sizeof(r->canonical_training_seq))
		return 0;

	if (!encode_text(tok, r->canonical_prompt, 1, 0,
	    &r->prompt_tokens, &r->prompt_count) ||
	    !encode_text(tok, r->canonical_visible_seq, 1, 0,
	    &r->visible_seq_tokens, &r->visible_seq_count) ||
	    !encode_text(tok, r->canonical_training_seq, 1, 0,
	    &r->training_seq_tokens, &r->training_seq_count))
		return 0;

	if (r->training_seq_count > tr->config.seq_len)
		return 0;
	r->first_target_index = r->prompt_count;
	if (r->first_target_index <= 0 || r->first_target_index >= r->training_seq_count)
		return 0;
	r->visible_target_count = r->visible_seq_count - r->first_target_index;
	if (r->visible_target_count <= 0)
		return 0;
	r->terminator_token_id = r->training_seq_tokens[r->training_seq_count - 1];
	return 1;
}

static void stage4e_key_counts_free(stage4e_key_counts_t *k)
{
	free(k->items);
	memset(k, 0, sizeof(*k));
}

static int stage4e_key_counts_add(stage4e_key_counts_t *k, const char *key)
{
	size_t i;
	stage4e_key_count_t *new_items;

	for (i = 0; i < k->count; i++) {
		if (strcmp(k->items[i].key, key) == 0) {
			k->items[i].count++;
			return 1;
		}
	}

	if (k->count == k->cap) {
		size_t new_cap = k->cap == 0 ? 8U : k->cap * 2U;
		new_items = realloc(k->items, new_cap * sizeof(stage4e_key_count_t));
		if (new_items == NULL)
			return 0;
		k->items = new_items;
		k->cap = new_cap;
	}
	strncpy(k->items[k->count].key, key, sizeof(k->items[k->count].key) - 1U);
	k->items[k->count].key[sizeof(k->items[k->count].key) - 1U] = '\0';
	k->items[k->count].count = 1;
	k->count++;
	return 1;
}

static int stage4e_key_counts_max(const stage4e_key_counts_t *k)
{
	size_t i;
	int best = 0;
	for (i = 0; i < k->count; i++) {
		if (k->items[i].count > best)
			best = k->items[i].count;
	}
	return best;
}

static void stage4e_tokens_to_csv(const int *tokens, int count, char *out, size_t out_cap)
{
	int i;
	size_t used = 0;
	if (out_cap == 0)
		return;
	out[0] = '\0';
	for (i = 0; i < count; i++) {
		int n = snprintf(out + used, out_cap - used, "%s%d", i == 0 ? "" : ",", tokens[i]);
		if (n < 0 || (size_t)n >= out_cap - used)
			return;
		used += (size_t)n;
	}
}

int main(int argc, char **argv)
{
	const char *checkpoint_path = NULL;
	const char *dataset_path = NULL;
	const char *tokenizer_path = NULL;
	const char *report_path = "stage4e-baseline-unadapted-generation-report.txt";
	int max_visible_output_tokens = 64;
	int verbose = 0;
	int argi;
	stage4e_dataset_t ds;
	Transformer tr;
	Tokenizer tok;
	size_t i;
	FILE *report;
	stage4e_gen_metric_t split_metrics[STAGE4E_SPLIT_COUNT];
	stage4e_gen_metric_t split_task_metrics[STAGE4E_SPLIT_COUNT][STAGE4E_TASK_COUNT];
	char dataset_sha[65];

	memset(&ds, 0, sizeof(ds));
	memset(&tr, 0, sizeof(tr));
	memset(&tok, 0, sizeof(tok));
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
		else if (strcmp(argv[argi], "--max-visible-output-tokens") == 0 && argi + 1 < argc) {
			max_visible_output_tokens = atoi(argv[++argi]);
			if (max_visible_output_tokens < 1 || max_visible_output_tokens > STAGE4E_MAX_GEN_TOKENS)
				stage4e_usage(argv[0]);
		} else if (strcmp(argv[argi], "-v") == 0)
			verbose = 1;
		else
			stage4e_usage(argv[0]);
	}
	if (dataset_path == NULL || tokenizer_path == NULL)
		stage4e_usage(argv[0]);

	if (!stage4e_load_records(dataset_path, &ds) || ds.count == 0)
		return EXIT_FAILURE;
	if (!stage4_hash_file_sha256_hex(dataset_path, dataset_sha))
		return EXIT_FAILURE;

	load_transformer(&tr, checkpoint_path);
	load_tokenizer(&tok, tokenizer_path, tr.config.vocab_size);

	report = fopen(report_path, "wb");
	if (report == NULL)
		goto fail;

	fprintf(report, "mode=stage4e_unadapted_generation_baseline\n");
	fprintf(report, "dataset_path=%s\n", dataset_path);
	fprintf(report, "dataset_sha256=%s\n", dataset_sha);
	fprintf(report, "adapter_loaded=no\n");
	fprintf(report, "generation_temperature=0\n");
	fprintf(report, "newline_terminator_token_id=13\n");
	fprintf(report, "max_visible_output_tokens=%d\n", max_visible_output_tokens);

	for (i = 0; i < ds.count; i++) {
		stage4e_record_t *r = &ds.items[i];
		int split_idx;
		int task_idx;
		int token;
		int next = 0;
		int position = 0;
		int gen_tokens[STAGE4E_MAX_GEN_TOKENS];
		int gen_count = 0;
		int target_window_match = 0;
		int term_match = 0;
		int exact_match = 0;
		char generated_visible_text[STAGE4E_MAX_TEXT * 2];
		size_t visible_text_len = 0;
		char expected_ids_csv[2048];
		char generated_ids_csv[2048];
		char generated_window_csv[2048];
		char first_token_key[64];
		char window_key[1024];
		const char *termination_reason = "seq_len";

		if (!stage4e_prepare_record_tokens(&tok, &tr, r))
			goto fail;
		split_idx = stage4e_split_index(r->split);
		task_idx = stage4e_task_index(r->task);
		if (split_idx < 0 || task_idx < 0)
			goto fail;

		clear_run_state(&tr);
		generated_visible_text[0] = '\0';
		token = r->prompt_tokens[0];

		while (position < tr.config.seq_len && gen_count < max_visible_output_tokens) {
			const float *logits = forward(&tr, token, position);
			if (position < r->prompt_count - 1) {
				next = r->prompt_tokens[position + 1];
				position++;
				token = next;
				continue;
			}

			next = stage4e_argmax_token(logits, tr.config.vocab_size);
			gen_tokens[gen_count++] = next;

			if (next == 13) {
				termination_reason = "newline";
				break;
			}
			if (next == 1) {
				termination_reason = "eos";
				break;
			}

			{
				char *piece = decode_piece(&tok, token, next);
				size_t piece_len = strlen(piece);
				if (visible_text_len + piece_len + 1U < sizeof(generated_visible_text)) {
					memcpy(generated_visible_text + visible_text_len, piece, piece_len);
					visible_text_len += piece_len;
					generated_visible_text[visible_text_len] = '\0';
				}
			}

			position++;
			token = next;
		}
		if (gen_count >= max_visible_output_tokens && strcmp(termination_reason, "newline") != 0 &&
		    strcmp(termination_reason, "eos") != 0)
			termination_reason = "max_visible_tokens";

		target_window_match = 1;
		if (gen_count < r->visible_target_count)
			target_window_match = 0;
		else {
			int j;
			for (j = 0; j < r->visible_target_count; j++) {
				if (gen_tokens[j] != r->visible_seq_tokens[r->first_target_index + j]) {
					target_window_match = 0;
					break;
				}
			}
		}
		term_match = (gen_count > r->visible_target_count &&
		    gen_tokens[r->visible_target_count] == r->terminator_token_id);
		exact_match = (target_window_match && term_match &&
		    gen_count == r->visible_target_count + 1);

		split_metrics[split_idx].records++;
		split_task_metrics[split_idx][task_idx].records++;
		if (target_window_match) {
			split_metrics[split_idx].window_match++;
			split_task_metrics[split_idx][task_idx].window_match++;
		}
		if (term_match) {
			split_metrics[split_idx].term_match++;
			split_task_metrics[split_idx][task_idx].term_match++;
		}
		if (exact_match) {
			split_metrics[split_idx].exact_match++;
			split_task_metrics[split_idx][task_idx].exact_match++;
		}

		stage4e_tokens_to_csv(r->visible_seq_tokens + r->first_target_index,
		    r->visible_target_count, expected_ids_csv, sizeof(expected_ids_csv));
		stage4e_tokens_to_csv(gen_tokens, gen_count, generated_ids_csv, sizeof(generated_ids_csv));
		stage4e_tokens_to_csv(gen_tokens,
		    gen_count < r->visible_target_count ? gen_count : r->visible_target_count,
		    generated_window_csv, sizeof(generated_window_csv));

		if (gen_count > 0)
			snprintf(first_token_key, sizeof(first_token_key), "%d", gen_tokens[0]);
		else
			strncpy(first_token_key, "none", sizeof(first_token_key) - 1U);

		snprintf(window_key, sizeof(window_key), "%s%s",
		    generated_window_csv,
		    (gen_count < r->visible_target_count) ? "|short" : "");

		if (!stage4e_key_counts_add(&split_metrics[split_idx].windows, window_key) ||
		    !stage4e_key_counts_add(&split_metrics[split_idx].first_tokens, first_token_key) ||
		    !stage4e_key_counts_add(&split_task_metrics[split_idx][task_idx].windows, window_key) ||
		    !stage4e_key_counts_add(&split_task_metrics[split_idx][task_idx].first_tokens, first_token_key))
			goto fail;

		fprintf(report, "record.id=%s\n", r->id);
		fprintf(report, "record.split=%s\n", r->split);
		fprintf(report, "record.task=%s\n", r->task);
		fprintf(report, "record.expected_visible_target=%s\n", r->target_visible);
		fprintf(report, "record.expected_target_token_ids=%s\n", expected_ids_csv);
		fprintf(report, "record.generated_visible_text=%s\n", generated_visible_text);
		fprintf(report, "record.generated_token_ids=%s\n", generated_ids_csv);
		fprintf(report, "record.generated_target_window_token_ids=%s\n", generated_window_csv);
		fprintf(report, "record.target_window_match=%s\n", target_window_match ? "yes" : "no");
		fprintf(report, "record.response_termination_match=%s\n", term_match ? "yes" : "no");
		fprintf(report, "record.response_exact_match=%s\n", exact_match ? "yes" : "no");
		fprintf(report, "record.termination_reason=%s\n", termination_reason);
		fprintf(report, "record.first_generated_token=%s\n", first_token_key);

		if (verbose) {
			printf("record id=%s split=%s task=%s target_window_match=%s exact=%s\n",
			    r->id, r->split, r->task,
			    target_window_match ? "yes" : "no",
			    exact_match ? "yes" : "no");
		}
	}

	for (i = 0; i < STAGE4E_SPLIT_COUNT; i++) {
		stage4e_gen_metric_t *m = &split_metrics[i];
		if (m->records == 0)
			continue;
		fprintf(report, "split.%s.target_window_match_count=%llu\n",
		    stage4e_split_names[i], (unsigned long long)m->window_match);
		fprintf(report, "split.%s.target_window_match_rate=%.12f\n",
		    stage4e_split_names[i], (double)m->window_match / (double)m->records);
		fprintf(report, "split.%s.response_termination_match_count=%llu\n",
		    stage4e_split_names[i], (unsigned long long)m->term_match);
		fprintf(report, "split.%s.response_termination_match_rate=%.12f\n",
		    stage4e_split_names[i], (double)m->term_match / (double)m->records);
		fprintf(report, "split.%s.response_exact_match_count=%llu\n",
		    stage4e_split_names[i], (unsigned long long)m->exact_match);
		fprintf(report, "split.%s.response_exact_match_rate=%.12f\n",
		    stage4e_split_names[i], (double)m->exact_match / (double)m->records);
		fprintf(report, "split.%s.distinct_generated_target_windows=%lu\n",
		    stage4e_split_names[i], (unsigned long)m->windows.count);
		fprintf(report, "split.%s.distinct_first_generated_tokens=%lu\n",
		    stage4e_split_names[i], (unsigned long)m->first_tokens.count);
		fprintf(report, "split.%s.most_common_generated_window_count=%d\n",
		    stage4e_split_names[i], stage4e_key_counts_max(&m->windows));
		fprintf(report, "split.%s.most_common_generated_window_rate=%.12f\n",
		    stage4e_split_names[i],
		    (double)stage4e_key_counts_max(&m->windows) / (double)m->records);
	}

	for (i = 0; i < STAGE4E_SPLIT_COUNT; i++) {
		size_t j;
		for (j = 0; j < STAGE4E_TASK_COUNT; j++) {
			stage4e_gen_metric_t *m = &split_task_metrics[i][j];
			if (m->records == 0)
				continue;
			fprintf(report, "split.%s.task.%s.records=%llu\n",
			    stage4e_split_names[i], stage4e_task_names[j],
			    (unsigned long long)m->records);
			fprintf(report, "split.%s.task.%s.target_window_match_rate=%.12f\n",
			    stage4e_split_names[i], stage4e_task_names[j],
			    (double)m->window_match / (double)m->records);
			fprintf(report, "split.%s.task.%s.response_termination_match_rate=%.12f\n",
			    stage4e_split_names[i], stage4e_task_names[j],
			    (double)m->term_match / (double)m->records);
			fprintf(report, "split.%s.task.%s.response_exact_match_rate=%.12f\n",
			    stage4e_split_names[i], stage4e_task_names[j],
			    (double)m->exact_match / (double)m->records);
		}
	}

	fclose(report);
	printf("mode=stage4e_unadapted_generation_baseline\n");
	printf("report_path=%s\n", report_path);

	for (i = 0; i < STAGE4E_SPLIT_COUNT; i++) {
		stage4e_key_counts_free(&split_metrics[i].windows);
		stage4e_key_counts_free(&split_metrics[i].first_tokens);
	}
	for (i = 0; i < STAGE4E_SPLIT_COUNT; i++) {
		size_t j;
		for (j = 0; j < STAGE4E_TASK_COUNT; j++) {
			stage4e_key_counts_free(&split_task_metrics[i][j].windows);
			stage4e_key_counts_free(&split_task_metrics[i][j].first_tokens);
		}
	}
	stage4e_dataset_free(&ds);
	free_tokenizer(&tok);
	free_transformer(&tr);
	return EXIT_SUCCESS;

fail:
	if (report != NULL)
		fclose(report);
	for (i = 0; i < STAGE4E_SPLIT_COUNT; i++) {
		stage4e_key_counts_free(&split_metrics[i].windows);
		stage4e_key_counts_free(&split_metrics[i].first_tokens);
	}
	for (i = 0; i < STAGE4E_SPLIT_COUNT; i++) {
		size_t j;
		for (j = 0; j < STAGE4E_TASK_COUNT; j++) {
			stage4e_key_counts_free(&split_task_metrics[i][j].windows);
			stage4e_key_counts_free(&split_task_metrics[i][j].first_tokens);
		}
	}
	stage4e_dataset_free(&ds);
	free_tokenizer(&tok);
	free_transformer(&tr);
	return EXIT_FAILURE;
}
