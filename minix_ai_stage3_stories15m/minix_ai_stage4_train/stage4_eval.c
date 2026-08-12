#include <ctype.h>
#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int minix_llama_embedded_main(int argc, char **argv);

#define main minix_llama_embedded_main
#include "../minix_llama.c"
#undef main

typedef struct {
	char *text;
} EvalRecord;

typedef struct {
	EvalRecord *items;
	size_t count;
	size_t cap;
} EvalDataset;

static void eval_usage(const char *prog) NORETURN;

static void eval_usage(const char *prog)
{
	fprintf(stderr,
	    "Usage: %s <checkpoint.bin> -d <dataset.txt> [options]\n"
	    "\n"
	    "Options:\n"
	    "  -z <tokenizer.bin> tokenizer path (default tokenizer.bin)\n"
	    "  -a <adapter.bin>   optional adapter sidecar\n"
	    "  -y <scale>         adapter correction scale (default 1.0)\n"
	    "  -v                 verbose record-level progress\n"
	    "\n"
	    "Dataset marker format:\n"
	    "  <record>\n"
	    "  text...\n"
	    "  </record>\n",
	    prog);
	exit(EXIT_FAILURE);
}

static void dataset_free(EvalDataset *ds)
{
	size_t i;

	for (i = 0; i < ds->count; i++)
		free(ds->items[i].text);
	free(ds->items);
	memset(ds, 0, sizeof(*ds));
}

static int append_text(char **text, size_t *used, size_t *cap, const char *line)
{
	size_t len = strlen(line);
	size_t need;
	char *new_text;

	if (!checked_add_size(*used, len, &need) ||
	    !checked_add_size(need, 2U, &need))
		return 0;

	if (need > *cap) {
		size_t new_cap = (*cap == 0U) ? 256U : *cap;
		while (new_cap < need) {
			if (!checked_mul_size(new_cap, 2U, &new_cap))
				return 0;
		}
		new_text = realloc(*text, new_cap);
		if (new_text == NULL)
			return 0;
		*text = new_text;
		*cap = new_cap;
	}

	memcpy(*text + *used, line, len);
	*used += len;
	(*text)[(*used)++] = '\n';
	(*text)[*used] = '\0';
	return 1;
}

static int push_record(EvalDataset *ds, char *text)
{
	EvalRecord *new_items;

	if (text == NULL || text[0] == '\0')
		return 0;

	if (ds->count == ds->cap) {
		size_t new_cap = (ds->cap == 0U) ? 8U : ds->cap * 2U;
		new_items = realloc(ds->items, new_cap * sizeof(EvalRecord));
		if (new_items == NULL)
			return 0;
		ds->items = new_items;
		ds->cap = new_cap;
	}

	ds->items[ds->count].text = text;
	ds->count++;
	return 1;
}

static char *trim_line(char *line)
{
	size_t len;

	if (line == NULL)
		return NULL;
	len = strlen(line);
	while (len > 0U && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
		line[len - 1] = '\0';
		len--;
	}
	return line;
}

static int dataset_load(const char *path, EvalDataset *ds)
{
	FILE *f;
	char line[4096];
	int in_record = 0;
	char *record_text = NULL;
	size_t used = 0;
	size_t cap = 0;

	memset(ds, 0, sizeof(*ds));
	f = fopen(path, "rb");
	if (f == NULL) {
		fprintf(stderr, "error: opening dataset '%s': %s\n", path, strerror(errno));
		return 0;
	}

	while (fgets(line, sizeof(line), f) != NULL) {
		char *t = trim_line(line);

		if (!in_record) {
			if (strcmp(t, "<record>") == 0) {
				in_record = 1;
				free(record_text);
				record_text = NULL;
				used = 0;
				cap = 0;
				continue;
			}
			if (t[0] == '\0' || t[0] == '#')
				continue;
			fprintf(stderr, "error: malformed dataset line outside record: %s\n", t);
			dataset_free(ds);
			fclose(f);
			return 0;
		}

		if (strcmp(t, "</record>") == 0) {
			if (used == 0) {
				fprintf(stderr, "error: empty record is not allowed\n");
				free(record_text);
				dataset_free(ds);
				fclose(f);
				return 0;
			}
			if (!push_record(ds, record_text)) {
				fprintf(stderr, "error: cannot store dataset record\n");
				free(record_text);
				dataset_free(ds);
				fclose(f);
				return 0;
			}
			record_text = NULL;
			used = 0;
			cap = 0;
			in_record = 0;
			continue;
		}

		if (!append_text(&record_text, &used, &cap, t)) {
			fprintf(stderr, "error: cannot append dataset text\n");
			free(record_text);
			dataset_free(ds);
			fclose(f);
			return 0;
		}
	}

	if (ferror(f)) {
		fprintf(stderr, "error: reading dataset '%s' failed\n", path);
		free(record_text);
		dataset_free(ds);
		fclose(f);
		return 0;
	}
	if (in_record) {
		fprintf(stderr, "error: dataset ended before </record>\n");
		free(record_text);
		dataset_free(ds);
		fclose(f);
		return 0;
	}

	fclose(f);
	return 1;
}

static int loss_from_logits(const float *logits, int vocab_size,
	int target, double *loss_out)
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

int main(int argc, char **argv)
{
	const char *checkpoint_path;
	const char *tokenizer_path = "tokenizer.bin";
	const char *dataset_path = NULL;
	const char *adapter_path = NULL;
	float adapter_scale = 1.0f;
	int verbose = 0;
	Transformer transformer;
	Tokenizer tokenizer;
	EvalDataset dataset;
	size_t record_index;
	double total_loss = 0.0;
	uint64_t prediction_tokens = 0;
	double start_time;
	double end_time;
	int argi;

	if (argc < 2)
		eval_usage(argv[0]);
	checkpoint_path = argv[1];

	for (argi = 2; argi < argc; argi++) {
		if (strcmp(argv[argi], "-z") == 0 && argi + 1 < argc) {
			tokenizer_path = argv[++argi];
		} else if (strcmp(argv[argi], "-d") == 0 && argi + 1 < argc) {
			dataset_path = argv[++argi];
		} else if (strcmp(argv[argi], "-a") == 0 && argi + 1 < argc) {
			adapter_path = argv[++argi];
		} else if (strcmp(argv[argi], "-y") == 0 && argi + 1 < argc) {
			double parsed = parse_double_value(argv[++argi], "adapter scale");
			if (!isfinite(parsed) || parsed < -FLT_MAX || parsed > FLT_MAX)
				die("adapter scale is outside supported range");
			adapter_scale = (float)parsed;
		} else if (strcmp(argv[argi], "-v") == 0) {
			verbose = 1;
		} else {
			eval_usage(argv[0]);
		}
	}

	if (dataset_path == NULL)
		eval_usage(argv[0]);

	if (!dataset_load(dataset_path, &dataset))
		return EXIT_FAILURE;
	if (dataset.count == 0) {
		fprintf(stderr, "error: dataset has zero records\n");
		return EXIT_FAILURE;
	}

	load_transformer(&transformer, checkpoint_path);
	load_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);
	if (adapter_path != NULL)
		load_adapter_runtime(adapter_path, &transformer, adapter_scale);

	start_time = monotonic_seconds();

	for (record_index = 0; record_index < dataset.count; record_index++) {
		int *tokens = NULL;
		int token_count = 0;
		int pos;

		if (!encode_text(&tokenizer, dataset.items[record_index].text,
		    1, 0, &tokens, &token_count)) {
			fprintf(stderr, "error: tokenization failed for record %lu\n",
			    (unsigned long)record_index);
			dataset_free(&dataset);
			free_tokenizer(&tokenizer);
			free_transformer(&transformer);
			unload_adapter_runtime();
			return EXIT_FAILURE;
		}

		if (token_count < 2) {
			fprintf(stderr, "error: record %lu has fewer than 2 tokens\n",
			    (unsigned long)record_index);
			free(tokens);
			dataset_free(&dataset);
			free_tokenizer(&tokenizer);
			free_transformer(&transformer);
			unload_adapter_runtime();
			return EXIT_FAILURE;
		}

		if (token_count > transformer.config.seq_len) {
			fprintf(stderr,
			    "error: record %lu exceeds context length (%d > %d)\n",
			    (unsigned long)record_index,
			    token_count, transformer.config.seq_len);
			free(tokens);
			dataset_free(&dataset);
			free_tokenizer(&tokenizer);
			free_transformer(&transformer);
			unload_adapter_runtime();
			return EXIT_FAILURE;
		}

		clear_run_state(&transformer);

		for (pos = 0; pos < token_count - 1; pos++) {
			const float *logits;
			double loss;
			int target = tokens[pos + 1];

			if (target < 0 || target >= transformer.config.vocab_size) {
				fprintf(stderr,
				    "error: invalid target token id %d in record %lu\n",
				    target, (unsigned long)record_index);
				free(tokens);
				dataset_free(&dataset);
				free_tokenizer(&tokenizer);
				free_transformer(&transformer);
				unload_adapter_runtime();
				return EXIT_FAILURE;
			}

			logits = forward(&transformer, tokens[pos], pos);
			if (!loss_from_logits(logits, transformer.config.vocab_size,
			    target, &loss)) {
				fprintf(stderr,
				    "error: non-finite logits/loss at record %lu position %d\n",
				    (unsigned long)record_index, pos);
				free(tokens);
				dataset_free(&dataset);
				free_tokenizer(&tokenizer);
				free_transformer(&transformer);
				unload_adapter_runtime();
				return EXIT_FAILURE;
			}

			total_loss += loss;
			prediction_tokens++;
		}

		if (verbose) {
			fprintf(stderr,
			    "record %lu: tokens=%d predictions=%d\n",
			    (unsigned long)record_index,
			    token_count, token_count - 1);
		}

		free(tokens);
	}

	end_time = monotonic_seconds();

	{
		double elapsed = end_time - start_time;
		double avg_loss = total_loss / (double)prediction_tokens;
		double ppl;

		printf("record_count: %lu\n", (unsigned long)dataset.count);
		printf("prediction_token_count: %llu\n",
		    (unsigned long long)prediction_tokens);
		printf("total_loss: %.12f\n", total_loss);
		printf("average_loss: %.12f\n", avg_loss);

		if (isfinite(avg_loss) && avg_loss < log(DBL_MAX)) {
			ppl = exp(avg_loss);
			if (isfinite(ppl))
				printf("perplexity: %.12f\n", ppl);
			else
				printf("perplexity: inf\n");
		} else {
			printf("perplexity: inf\n");
		}

		if (elapsed <= 0.0)
			elapsed = 1e-9;
		printf("elapsed_seconds: %.6f\n", elapsed);
		printf("prediction_tokens_per_second: %.6f\n",
		    (double)prediction_tokens / elapsed);
	}

	dataset_free(&dataset);
	free_tokenizer(&tokenizer);
	free_transformer(&transformer);
	unload_adapter_runtime();
	return EXIT_SUCCESS;
}
