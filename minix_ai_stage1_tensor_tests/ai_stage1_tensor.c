#define _POSIX_C_SOURCE 200112L
#define _NETBSD_SOURCE 1

#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TEST_PASS 0
#define TEST_FAIL 1
#define TEST_INFO 2

#define ARRAY_LEN(a) (sizeof(a) / sizeof((a)[0]))
#define DEFAULT_ITERATIONS 500U
#define DEFAULT_BENCH_REPS 32U

struct test_context {
    unsigned int iterations;
    unsigned int bench_reps;
    int verbose;
};

struct test_case {
    int number;
    const char *name;
    int (*run)(const struct test_context *ctx);
};

struct tiny_model {
    float rms_att[4];
    float rms_ffn[4];
    float wq[16];
    float wk[16];
    float wv[16];
    float wo[16];
    float w1[24];
    float w3[24];
    float w2[24];
};

union allocation_header {
    struct {
        size_t bytes;
    } meta;
    long double align_long_double;
    void *align_pointer;
};

static int g_passed;
static int g_failed;
static int g_info;

static size_t g_live_allocations;
static size_t g_live_bytes;
static size_t g_peak_bytes;

static void print_rule(void)
{
    puts("------------------------------------------------------------");
}

static void report_result(int number, const char *name, int result)
{
    const char *label;

    if (result == TEST_PASS) {
        label = "PASS";
        g_passed++;
    } else if (result == TEST_INFO) {
        label = "INFO";
        g_info++;
    } else {
        label = "FAIL";
        g_failed++;
    }

    printf("[%s] Test %d: %s\n", label, number, name);
}

static int nearly_equal(float actual, float expected,
    float absolute_tolerance, float relative_tolerance)
{
    float difference;
    float scale;

    difference = fabsf(actual - expected);
    if (difference <= absolute_tolerance)
        return 1;

    scale = fabsf(actual);
    if (fabsf(expected) > scale)
        scale = fabsf(expected);

    return difference <= scale * relative_tolerance;
}

static int checked_multiply_size(size_t left, size_t right, size_t *result)
{
    if (result == NULL)
        return -1;

    if (left != 0U && right > ((size_t)-1) / left)
        return -1;

    *result = left * right;
    return 0;
}

static int checked_add_size(size_t left, size_t right, size_t *result)
{
    if (result == NULL)
        return -1;

    if (right > ((size_t)-1) - left)
        return -1;

    *result = left + right;
    return 0;
}

static void *tracked_alloc(size_t count, size_t element_size, int clear)
{
    union allocation_header *header;
    size_t payload_bytes;
    size_t total_bytes;

    if (checked_multiply_size(count, element_size, &payload_bytes) != 0) {
        errno = ENOMEM;
        return NULL;
    }

    if (checked_add_size(sizeof(*header), payload_bytes, &total_bytes) != 0) {
        errno = ENOMEM;
        return NULL;
    }

    header = (union allocation_header *)malloc(total_bytes);
    if (header == NULL)
        return NULL;

    header->meta.bytes = payload_bytes;
    g_live_allocations++;
    g_live_bytes += payload_bytes;
    if (g_live_bytes > g_peak_bytes)
        g_peak_bytes = g_live_bytes;

    if (clear && payload_bytes != 0U)
        memset((void *)(header + 1), 0, payload_bytes);

    return (void *)(header + 1);
}

static void tracked_free(void *pointer)
{
    union allocation_header *header;

    if (pointer == NULL)
        return;

    header = ((union allocation_header *)pointer) - 1;
    if (g_live_allocations > 0U)
        g_live_allocations--;
    if (g_live_bytes >= header->meta.bytes)
        g_live_bytes -= header->meta.bytes;
    else
        g_live_bytes = 0U;

    free(header);
}

static void reset_allocation_counters(void)
{
    g_live_allocations = 0U;
    g_live_bytes = 0U;
    g_peak_bytes = 0U;
}

static void fill_pattern(float *values, size_t count,
    int multiplier, int addend, float scale)
{
    size_t index;

    for (index = 0U; index < count; index++) {
        int integer_value;

        integer_value = ((int)((index * (size_t)multiplier +
            (size_t)addend) % 17U)) - 8;
        values[index] = (float)integer_value * scale;
    }
}

static float maximum_absolute_error(const float *actual,
    const float *expected, size_t count)
{
    float maximum;
    size_t index;

    maximum = 0.0f;
    for (index = 0U; index < count; index++) {
        float difference;

        difference = fabsf(actual[index] - expected[index]);
        if (difference > maximum)
            maximum = difference;
    }

    return maximum;
}

static void print_vector(const char *label, const float *values, size_t count)
{
    size_t index;

    printf("%s", label);
    for (index = 0U; index < count; index++)
        printf("%s%.9g", index == 0U ? "" : " ", (double)values[index]);
    putchar('\n');
}

static int matmul(float *output, const float *left, const float *right,
    size_t rows, size_t shared, size_t columns)
{
    size_t row;
    size_t column;
    size_t inner;

    if (output == NULL || left == NULL || right == NULL)
        return -1;

    for (row = 0U; row < rows; row++) {
        for (column = 0U; column < columns; column++) {
            float sum;

            sum = 0.0f;
            for (inner = 0U; inner < shared; inner++) {
                sum += left[row * shared + inner] *
                    right[inner * columns + column];
            }
            output[row * columns + column] = sum;
        }
    }

    return 0;
}

static int stable_softmax(float *values, size_t count)
{
    float maximum;
    float sum;
    size_t index;

    if (values == NULL || count == 0U)
        return -1;

    maximum = values[0];
    for (index = 1U; index < count; index++) {
        if (values[index] > maximum)
            maximum = values[index];
    }

    sum = 0.0f;
    for (index = 0U; index < count; index++) {
        values[index] = expf(values[index] - maximum);
        sum += values[index];
    }

    if (!(sum > 0.0f) || !isfinite(sum))
        return -1;

    for (index = 0U; index < count; index++)
        values[index] /= sum;

    return 0;
}

static int rmsnorm_vector(float *output, const float *input,
    const float *weight, size_t width, float epsilon)
{
    float mean_square;
    float inverse_rms;
    size_t index;

    if (output == NULL || input == NULL || weight == NULL || width == 0U)
        return -1;

    mean_square = 0.0f;
    for (index = 0U; index < width; index++)
        mean_square += input[index] * input[index];
    mean_square /= (float)width;

    inverse_rms = 1.0f / sqrtf(mean_square + epsilon);
    if (!isfinite(inverse_rms))
        return -1;

    for (index = 0U; index < width; index++)
        output[index] = input[index] * inverse_rms * weight[index];

    return 0;
}

static int rmsnorm_rows(float *output, const float *input,
    const float *weight, size_t rows, size_t width, float epsilon)
{
    size_t row;

    for (row = 0U; row < rows; row++) {
        if (rmsnorm_vector(output + row * width, input + row * width,
                weight, width, epsilon) != 0)
            return -1;
    }

    return 0;
}

static float silu(float value)
{
    return value / (1.0f + expf(-value));
}

static int causal_attention(float *output, float *weight_matrix,
    const float *query, const float *key, const float *value,
    size_t sequence_length, size_t head_width)
{
    float *scores;
    float scale;
    size_t query_index;
    size_t key_index;
    size_t component;

    if (output == NULL || query == NULL || key == NULL || value == NULL ||
        sequence_length == 0U || head_width == 0U)
        return -1;

    scores = (float *)tracked_alloc(sequence_length, sizeof(float), 0);
    if (scores == NULL)
        return -1;

    scale = 1.0f / sqrtf((float)head_width);
    memset(output, 0, sequence_length * head_width * sizeof(float));
    if (weight_matrix != NULL)
        memset(weight_matrix, 0,
            sequence_length * sequence_length * sizeof(float));

    for (query_index = 0U; query_index < sequence_length; query_index++) {
        for (key_index = 0U; key_index <= query_index; key_index++) {
            float dot;

            dot = 0.0f;
            for (component = 0U; component < head_width; component++) {
                dot += query[query_index * head_width + component] *
                    key[key_index * head_width + component];
            }
            scores[key_index] = dot * scale;
        }

        if (stable_softmax(scores, query_index + 1U) != 0) {
            tracked_free(scores);
            return -1;
        }

        for (key_index = 0U; key_index <= query_index; key_index++) {
            float probability;

            probability = scores[key_index];
            if (weight_matrix != NULL) {
                weight_matrix[query_index * sequence_length + key_index] =
                    probability;
            }
            for (component = 0U; component < head_width; component++) {
                output[query_index * head_width + component] +=
                    probability * value[key_index * head_width + component];
            }
        }
    }

    tracked_free(scores);
    return 0;
}

static void initialize_tiny_model(struct tiny_model *model)
{
    static const float rms_att[] = { 1.0f, 0.95f, 1.05f, 0.9f };
    static const float rms_ffn[] = { 0.9f, 1.1f, 1.0f, 1.05f };

    memcpy(model->rms_att, rms_att, sizeof(rms_att));
    memcpy(model->rms_ffn, rms_ffn, sizeof(rms_ffn));

    fill_pattern(model->wq, ARRAY_LEN(model->wq), 3, 1, 0.04f);
    fill_pattern(model->wk, ARRAY_LEN(model->wk), 5, 2, 0.035f);
    fill_pattern(model->wv, ARRAY_LEN(model->wv), 7, 3, 0.03f);
    fill_pattern(model->wo, ARRAY_LEN(model->wo), 11, 4, 0.025f);
    fill_pattern(model->w1, ARRAY_LEN(model->w1), 13, 5, 0.03f);
    fill_pattern(model->w3, ARRAY_LEN(model->w3), 9, 6, 0.028f);
    fill_pattern(model->w2, ARRAY_LEN(model->w2), 4, 7, 0.026f);
}

static int tiny_transformer_block(float *output, const float *input,
    const struct tiny_model *model)
{
    const size_t sequence_length = 3U;
    const size_t model_width = 4U;
    const size_t hidden_width = 6U;
    const float epsilon = 1.0e-5f;
    float *norm1;
    float *query;
    float *key;
    float *value;
    float *attention;
    float *attention_projection;
    float *residual1;
    float *norm2;
    float *gate;
    float *up;
    float *hidden;
    float *feed_forward;
    size_t element;
    int status;

    norm1 = NULL;
    query = NULL;
    key = NULL;
    value = NULL;
    attention = NULL;
    attention_projection = NULL;
    residual1 = NULL;
    norm2 = NULL;
    gate = NULL;
    up = NULL;
    hidden = NULL;
    feed_forward = NULL;
    status = -1;

    norm1 = (float *)tracked_alloc(sequence_length * model_width,
        sizeof(float), 0);
    query = (float *)tracked_alloc(sequence_length * model_width,
        sizeof(float), 0);
    key = (float *)tracked_alloc(sequence_length * model_width,
        sizeof(float), 0);
    value = (float *)tracked_alloc(sequence_length * model_width,
        sizeof(float), 0);
    attention = (float *)tracked_alloc(sequence_length * model_width,
        sizeof(float), 1);
    attention_projection = (float *)tracked_alloc(
        sequence_length * model_width, sizeof(float), 0);
    residual1 = (float *)tracked_alloc(sequence_length * model_width,
        sizeof(float), 0);
    norm2 = (float *)tracked_alloc(sequence_length * model_width,
        sizeof(float), 0);
    gate = (float *)tracked_alloc(sequence_length * hidden_width,
        sizeof(float), 0);
    up = (float *)tracked_alloc(sequence_length * hidden_width,
        sizeof(float), 0);
    hidden = (float *)tracked_alloc(sequence_length * hidden_width,
        sizeof(float), 0);
    feed_forward = (float *)tracked_alloc(sequence_length * model_width,
        sizeof(float), 0);

    if (norm1 == NULL || query == NULL || key == NULL || value == NULL ||
        attention == NULL || attention_projection == NULL ||
        residual1 == NULL || norm2 == NULL || gate == NULL || up == NULL ||
        hidden == NULL || feed_forward == NULL)
        goto cleanup;

    if (rmsnorm_rows(norm1, input, model->rms_att,
            sequence_length, model_width, epsilon) != 0)
        goto cleanup;

    if (matmul(query, norm1, model->wq,
            sequence_length, model_width, model_width) != 0 ||
        matmul(key, norm1, model->wk,
            sequence_length, model_width, model_width) != 0 ||
        matmul(value, norm1, model->wv,
            sequence_length, model_width, model_width) != 0)
        goto cleanup;

    if (causal_attention(attention, NULL, query, key, value,
            sequence_length, model_width) != 0)
        goto cleanup;

    if (matmul(attention_projection, attention, model->wo,
            sequence_length, model_width, model_width) != 0)
        goto cleanup;

    for (element = 0U; element < sequence_length * model_width; element++)
        residual1[element] = input[element] + attention_projection[element];

    if (rmsnorm_rows(norm2, residual1, model->rms_ffn,
            sequence_length, model_width, epsilon) != 0)
        goto cleanup;

    if (matmul(gate, norm2, model->w1,
            sequence_length, model_width, hidden_width) != 0 ||
        matmul(up, norm2, model->w3,
            sequence_length, model_width, hidden_width) != 0)
        goto cleanup;

    for (element = 0U; element < sequence_length * hidden_width; element++)
        hidden[element] = silu(gate[element]) * up[element];

    if (matmul(feed_forward, hidden, model->w2,
            sequence_length, hidden_width, model_width) != 0)
        goto cleanup;

    for (element = 0U; element < sequence_length * model_width; element++)
        output[element] = residual1[element] + feed_forward[element];

    status = 0;

cleanup:
    tracked_free(feed_forward);
    tracked_free(hidden);
    tracked_free(up);
    tracked_free(gate);
    tracked_free(norm2);
    tracked_free(residual1);
    tracked_free(attention_projection);
    tracked_free(attention);
    tracked_free(value);
    tracked_free(key);
    tracked_free(query);
    tracked_free(norm1);

    return status;
}

static int elapsed_seconds(struct timespec start, struct timespec end,
    double *seconds)
{
    time_t second_delta;
    long nanosecond_delta;

    if (seconds == NULL)
        return -1;

    second_delta = end.tv_sec - start.tv_sec;
    nanosecond_delta = end.tv_nsec - start.tv_nsec;
    if (nanosecond_delta < 0L) {
        second_delta--;
        nanosecond_delta += 1000000000L;
    }

    if (second_delta < 0)
        return -1;

    *seconds = (double)second_delta +
        (double)nanosecond_delta / 1000000000.0;
    return 0;
}

static int test_matrix_multiplication(const struct test_context *ctx)
{
    static const float left[] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        -1.0f, 0.5f, 2.0f, -2.0f,
        0.25f, -0.5f, 1.5f, 2.0f
    };
    static const float right[] = {
        1.0f, 0.0f, -1.0f,
        2.0f, 1.0f, 0.5f,
        -1.0f, 3.0f, 2.0f,
        0.5f, -2.0f, 1.0f
    };
    static const float expected[] = {
        4.0f, 3.0f, 10.0f,
        -3.0f, 10.5f, 3.25f,
        -1.25f, 0.0f, 4.5f
    };
    const size_t benchmark_width = 64U;
    float output[9];
    float *benchmark_left;
    float *benchmark_right;
    float *benchmark_output;
    struct timespec start;
    struct timespec end;
    volatile float benchmark_sink;
    double seconds;
    double operations;
    double megaflops;
    unsigned int repetition;
    int failed;

    failed = 0;
    benchmark_left = NULL;
    benchmark_right = NULL;
    benchmark_output = NULL;
    benchmark_sink = 0.0f;

    if (matmul(output, left, right, 3U, 4U, 3U) != 0)
        return TEST_FAIL;

    print_vector("matrix output: ", output, ARRAY_LEN(output));
    printf("maximum absolute error: %.9g\n",
        (double)maximum_absolute_error(output, expected,
            ARRAY_LEN(expected)));

    for (repetition = 0U; repetition < ARRAY_LEN(expected); repetition++) {
        if (!nearly_equal(output[repetition], expected[repetition],
                1.0e-6f, 1.0e-6f))
            failed = 1;
    }

    benchmark_left = (float *)tracked_alloc(
        benchmark_width * benchmark_width, sizeof(float), 0);
    benchmark_right = (float *)tracked_alloc(
        benchmark_width * benchmark_width, sizeof(float), 0);
    benchmark_output = (float *)tracked_alloc(
        benchmark_width * benchmark_width, sizeof(float), 0);

    if (benchmark_left == NULL || benchmark_right == NULL ||
        benchmark_output == NULL) {
        fprintf(stderr, "benchmark allocation failed: %s\n", strerror(errno));
        failed = 1;
        goto cleanup;
    }

    fill_pattern(benchmark_left, benchmark_width * benchmark_width,
        7, 3, 0.015f);
    fill_pattern(benchmark_right, benchmark_width * benchmark_width,
        11, 5, 0.0125f);

    if (clock_gettime(CLOCK_MONOTONIC, &start) != 0) {
        fprintf(stderr, "clock_gettime(start): %s\n", strerror(errno));
        failed = 1;
        goto cleanup;
    }

    for (repetition = 0U; repetition < ctx->bench_reps; repetition++) {
        benchmark_left[repetition % (benchmark_width * benchmark_width)] +=
            1.0e-7f;
        if (matmul(benchmark_output, benchmark_left, benchmark_right,
                benchmark_width, benchmark_width, benchmark_width) != 0) {
            failed = 1;
            goto cleanup;
        }
        benchmark_sink += benchmark_output[
            (repetition * 17U) % (benchmark_width * benchmark_width)];
    }

    if (clock_gettime(CLOCK_MONOTONIC, &end) != 0) {
        fprintf(stderr, "clock_gettime(end): %s\n", strerror(errno));
        failed = 1;
        goto cleanup;
    }

    if (elapsed_seconds(start, end, &seconds) == 0 && seconds > 0.0) {
        operations = 2.0 * (double)benchmark_width *
            (double)benchmark_width * (double)benchmark_width *
            (double)ctx->bench_reps;
        megaflops = operations / seconds / 1000000.0;
        printf("64x64 matmul x %u: %.6f seconds, approximately %.3f MFLOP/s\n",
            ctx->bench_reps, seconds, megaflops);
    } else {
        printf("benchmark elapsed time was below usable timer resolution\n");
    }
    printf("benchmark sink: %.9g\n", (double)benchmark_sink);

cleanup:
    tracked_free(benchmark_output);
    tracked_free(benchmark_right);
    tracked_free(benchmark_left);
    return failed ? TEST_FAIL : TEST_PASS;
}

struct softmax_case {
    const char *name;
    const float *input;
    const float *expected;
    size_t count;
};

static int test_softmax(const struct test_context *ctx)
{
    static const float input_basic[] = { 1.0f, 2.0f, 3.0f, 4.0f };
    static const float input_large[] = { 1000.0f, 1001.0f, 1002.0f, 1003.0f };
    static const float input_negative[] = {
        -1000.0f, -1001.0f, -1002.0f, -1003.0f
    };
    static const float input_equal[] = { 5.0f, 5.0f, 5.0f, 5.0f };
    static const float input_single[] = { 42.0f };
    static const float expected_forward[] = {
        0.0320586033f, 0.0871443187f, 0.2368828181f, 0.6439142599f
    };
    static const float expected_reverse[] = {
        0.6439142599f, 0.2368828181f, 0.0871443187f, 0.0320586033f
    };
    static const float expected_equal[] = { 0.25f, 0.25f, 0.25f, 0.25f };
    static const float expected_single[] = { 1.0f };
    static const struct softmax_case cases[] = {
        { "basic", input_basic, expected_forward, ARRAY_LEN(input_basic) },
        { "large positive", input_large, expected_forward, ARRAY_LEN(input_large) },
        { "large negative", input_negative, expected_reverse, ARRAY_LEN(input_negative) },
        { "equal logits", input_equal, expected_equal, ARRAY_LEN(input_equal) },
        { "single logit", input_single, expected_single, ARRAY_LEN(input_single) }
    };
    size_t case_index;
    int failed;

    (void)ctx;
    failed = 0;

    for (case_index = 0U; case_index < ARRAY_LEN(cases); case_index++) {
        float values[4];
        float sum;
        size_t index;

        memset(values, 0, sizeof(values));
        memcpy(values, cases[case_index].input,
            cases[case_index].count * sizeof(float));

        if (stable_softmax(values, cases[case_index].count) != 0) {
            fprintf(stderr, "%s softmax failed\n", cases[case_index].name);
            failed = 1;
            continue;
        }

        sum = 0.0f;
        for (index = 0U; index < cases[case_index].count; index++) {
            sum += values[index];
            if (!isfinite(values[index]) || values[index] < 0.0f ||
                values[index] > 1.0f ||
                !nearly_equal(values[index], cases[case_index].expected[index],
                    3.0e-6f, 3.0e-6f))
                failed = 1;
        }

        printf("%s: sum=%.9g max_abs_error=%.9g\n",
            cases[case_index].name, (double)sum,
            (double)maximum_absolute_error(values,
                cases[case_index].expected, cases[case_index].count));
        if (!nearly_equal(sum, 1.0f, 3.0e-6f, 3.0e-6f))
            failed = 1;
    }

    return failed ? TEST_FAIL : TEST_PASS;
}

static int test_rmsnorm(const struct test_context *ctx)
{
    static const float input[] = { 1.0f, -2.0f, 3.0f, -4.0f };
    static const float weight[] = { 1.0f, 0.5f, -1.0f, 2.0f };
    static const float expected[] = {
        0.36514813f, -0.36514813f, -1.09544438f, -2.92118503f
    };
    static const float zero_input[] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float output[4];
    float zero_output[4];
    size_t index;
    int failed;

    (void)ctx;
    failed = 0;

    if (rmsnorm_vector(output, input, weight, ARRAY_LEN(input), 1.0e-5f) != 0)
        return TEST_FAIL;
    if (rmsnorm_vector(zero_output, zero_input, weight,
            ARRAY_LEN(zero_input), 1.0e-5f) != 0)
        return TEST_FAIL;

    print_vector("RMSNorm output: ", output, ARRAY_LEN(output));
    printf("maximum absolute error: %.9g\n",
        (double)maximum_absolute_error(output, expected,
            ARRAY_LEN(expected)));

    for (index = 0U; index < ARRAY_LEN(output); index++) {
        if (!nearly_equal(output[index], expected[index],
                5.0e-6f, 5.0e-6f))
            failed = 1;
        if (zero_output[index] != 0.0f)
            failed = 1;
    }

    return failed ? TEST_FAIL : TEST_PASS;
}

static int test_attention(const struct test_context *ctx)
{
    static const float query[] = {
        1.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 1.0f
    };
    static const float key[] = {
        1.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 1.0f
    };
    static const float value[] = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f
    };
    static const float expected_output[] = {
        1.0f, 2.0f,
        2.33952310f, 3.33952310f,
        3.51046953f, 4.51046953f
    };
    static const float expected_weights[] = {
        1.0f, 0.0f, 0.0f,
        0.33023845f, 0.66976155f, 0.0f,
        0.24825508f, 0.24825508f, 0.50348984f
    };
    float output[6];
    float weights[9];
    size_t index;
    int failed;

    (void)ctx;
    failed = 0;
    reset_allocation_counters();

    if (causal_attention(output, weights, query, key, value, 3U, 2U) != 0) {
        fprintf(stderr, "causal attention failed: %s\n", strerror(errno));
        return TEST_FAIL;
    }

    print_vector("attention output: ", output, ARRAY_LEN(output));
    print_vector("attention weights: ", weights, ARRAY_LEN(weights));
    printf("output max_abs_error=%.9g weights max_abs_error=%.9g\n",
        (double)maximum_absolute_error(output, expected_output,
            ARRAY_LEN(expected_output)),
        (double)maximum_absolute_error(weights, expected_weights,
            ARRAY_LEN(expected_weights)));

    for (index = 0U; index < ARRAY_LEN(output); index++) {
        if (!nearly_equal(output[index], expected_output[index],
                1.0e-5f, 1.0e-5f))
            failed = 1;
    }
    for (index = 0U; index < ARRAY_LEN(weights); index++) {
        if (!nearly_equal(weights[index], expected_weights[index],
                1.0e-5f, 1.0e-5f))
            failed = 1;
    }

    if (g_live_allocations != 0U || g_live_bytes != 0U) {
        fprintf(stderr, "attention leaked %lu allocations / %lu bytes\n",
            (unsigned long)g_live_allocations,
            (unsigned long)g_live_bytes);
        failed = 1;
    }

    return failed ? TEST_FAIL : TEST_PASS;
}

static int test_transformer_block(const struct test_context *ctx)
{
    static const float input[] = {
        0.5f, -1.0f, 0.25f, 2.0f,
        1.5f, 0.0f, -0.5f, 1.0f,
        -1.0f, 0.75f, 1.25f, -0.25f
    };
    static const float expected[] = {
        0.5428679945f, -1.1171119737f, 0.2181009198f, 2.1218231920f,
        1.5623042705f, -0.0899857298f, -0.5172602880f, 1.0627149982f,
        -0.9687400909f, 0.7018352556f, 1.2381107078f, -0.1963186134f
    };
    struct tiny_model model;
    float output[12];
    size_t index;
    int failed;

    (void)ctx;
    failed = 0;
    reset_allocation_counters();
    initialize_tiny_model(&model);

    if (tiny_transformer_block(output, input, &model) != 0) {
        fprintf(stderr, "tiny transformer block failed: %s\n", strerror(errno));
        return TEST_FAIL;
    }

    print_vector("transformer output: ", output, ARRAY_LEN(output));
    printf("maximum absolute error: %.9g\n",
        (double)maximum_absolute_error(output, expected,
            ARRAY_LEN(expected)));

    for (index = 0U; index < ARRAY_LEN(output); index++) {
        if (!nearly_equal(output[index], expected[index],
                1.5e-4f, 1.5e-4f))
            failed = 1;
    }

    if (g_live_allocations != 0U || g_live_bytes != 0U) {
        fprintf(stderr, "transformer block leaked %lu allocations / %lu bytes\n",
            (unsigned long)g_live_allocations,
            (unsigned long)g_live_bytes);
        failed = 1;
    }

    printf("tracked peak workspace: %lu bytes\n",
        (unsigned long)g_peak_bytes);

    return failed ? TEST_FAIL : TEST_PASS;
}

static int test_repeat_stability(const struct test_context *ctx)
{
    static const float input_template[] = {
        0.5f, -1.0f, 0.25f, 2.0f,
        1.5f, 0.0f, -0.5f, 1.0f,
        -1.0f, 0.75f, 1.25f, -0.25f
    };
    static const float expected[] = {
        0.5428679945f, -1.1171119737f, 0.2181009198f, 2.1218231920f,
        1.5623042705f, -0.0899857298f, -0.5172602880f, 1.0627149982f,
        -0.9687400909f, 0.7018352556f, 1.2381107078f, -0.1963186134f
    };
    struct tiny_model model;
    struct timespec start;
    struct timespec end;
    volatile float checksum;
    double seconds;
    unsigned int iteration;
    int failed;

    failed = 0;
    checksum = 0.0f;
    initialize_tiny_model(&model);
    reset_allocation_counters();

    if (clock_gettime(CLOCK_MONOTONIC, &start) != 0) {
        fprintf(stderr, "clock_gettime(start): %s\n", strerror(errno));
        return TEST_FAIL;
    }

    for (iteration = 0U; iteration < ctx->iterations; iteration++) {
        float *input;
        float *output;
        size_t index;

        input = (float *)tracked_alloc(ARRAY_LEN(input_template),
            sizeof(float), 0);
        output = (float *)tracked_alloc(ARRAY_LEN(expected),
            sizeof(float), 0);

        if (input == NULL || output == NULL) {
            fprintf(stderr, "iteration %u allocation failed: %s\n",
                iteration, strerror(errno));
            tracked_free(output);
            tracked_free(input);
            failed = 1;
            break;
        }

        memcpy(input, input_template, sizeof(input_template));
        if (tiny_transformer_block(output, input, &model) != 0) {
            fprintf(stderr, "iteration %u transformer failed: %s\n",
                iteration, strerror(errno));
            tracked_free(output);
            tracked_free(input);
            failed = 1;
            break;
        }

        for (index = 0U; index < ARRAY_LEN(expected); index++) {
            if (!nearly_equal(output[index], expected[index],
                    1.5e-4f, 1.5e-4f)) {
                fprintf(stderr,
                    "iteration %u output mismatch at %lu: %.9g vs %.9g\n",
                    iteration, (unsigned long)index,
                    (double)output[index], (double)expected[index]);
                failed = 1;
                break;
            }
        }

        checksum += output[iteration % ARRAY_LEN(expected)];
        tracked_free(output);
        tracked_free(input);

        if (g_live_allocations != 0U || g_live_bytes != 0U) {
            fprintf(stderr,
                "iteration %u ended with %lu allocations / %lu bytes live\n",
                iteration, (unsigned long)g_live_allocations,
                (unsigned long)g_live_bytes);
            failed = 1;
            break;
        }

        if (failed)
            break;
    }

    if (clock_gettime(CLOCK_MONOTONIC, &end) != 0) {
        fprintf(stderr, "clock_gettime(end): %s\n", strerror(errno));
        failed = 1;
    }

    if (elapsed_seconds(start, end, &seconds) == 0) {
        printf("%u transformer iterations: %.6f seconds",
            iteration, seconds);
        if (iteration != 0U)
            printf(" (%.3f ms/iteration)",
                seconds * 1000.0 / (double)iteration);
        putchar('\n');
    }

    printf("repeat checksum: %.9g\n", (double)checksum);
    printf("tracked peak live bytes: %lu\n", (unsigned long)g_peak_bytes);
    printf("final live allocations=%lu live bytes=%lu\n",
        (unsigned long)g_live_allocations,
        (unsigned long)g_live_bytes);

    return failed ? TEST_FAIL : TEST_PASS;
}

static const struct test_case test_cases[] = {
    { 12, "matrix multiplication", test_matrix_multiplication },
    { 13, "numerically stable softmax", test_softmax },
    { 14, "RMSNorm", test_rmsnorm },
    { 15, "one causal attention head", test_attention },
    { 16, "one tiny transformer block", test_transformer_block },
    { 17, "repeated allocation/inference/free stability", test_repeat_stability }
};

static void usage(const char *program)
{
    fprintf(stderr,
        "Usage: %s [--list] [--test N] [--iterations N] "
        "[--bench-reps N] [--verbose]\n",
        program);
}

static int parse_unsigned(const char *text, unsigned int *value)
{
    char *end;
    unsigned long parsed;

    errno = 0;
    end = NULL;
    parsed = strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed == 0UL ||
        parsed > 1000000UL)
        return -1;

    *value = (unsigned int)parsed;
    return 0;
}

int main(int argc, char **argv)
{
    struct test_context context;
    int selected_test;
    int list_only;
    int argument;
    size_t index;
    int found;

    context.iterations = DEFAULT_ITERATIONS;
    context.bench_reps = DEFAULT_BENCH_REPS;
    context.verbose = 0;
    selected_test = 0;
    list_only = 0;

    for (argument = 1; argument < argc; argument++) {
        if (strcmp(argv[argument], "--list") == 0) {
            list_only = 1;
        } else if (strcmp(argv[argument], "--verbose") == 0) {
            context.verbose = 1;
        } else if (strcmp(argv[argument], "--test") == 0) {
            if (++argument >= argc) {
                usage(argv[0]);
                return 2;
            }
            selected_test = atoi(argv[argument]);
        } else if (strcmp(argv[argument], "--iterations") == 0) {
            if (++argument >= argc ||
                parse_unsigned(argv[argument], &context.iterations) != 0) {
                usage(argv[0]);
                return 2;
            }
        } else if (strcmp(argv[argument], "--bench-reps") == 0) {
            if (++argument >= argc ||
                parse_unsigned(argv[argument], &context.bench_reps) != 0) {
                usage(argv[0]);
                return 2;
            }
        } else if (strcmp(argv[argument], "--help") == 0 ||
            strcmp(argv[argument], "-h") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    if (list_only) {
        for (index = 0U; index < ARRAY_LEN(test_cases); index++)
            printf("%d: %s\n", test_cases[index].number,
                test_cases[index].name);
        return 0;
    }

    printf("MINIX AI Stage 1 tensor-kernel and transformer probe\n");
    printf("iterations=%u bench_reps=%u selected_test=%d\n",
        context.iterations, context.bench_reps, selected_test);
    print_rule();

    found = selected_test == 0;
    for (index = 0U; index < ARRAY_LEN(test_cases); index++) {
        int result;

        if (selected_test != 0 && test_cases[index].number != selected_test)
            continue;

        found = 1;
        printf("\nTest %d: %s\n",
            test_cases[index].number, test_cases[index].name);
        print_rule();
        result = test_cases[index].run(&context);
        report_result(test_cases[index].number, test_cases[index].name, result);
    }

    if (!found) {
        fprintf(stderr, "Unknown test number: %d\n", selected_test);
        return 2;
    }

    print_rule();
    printf("Summary: %d passed, %d failed, %d informational\n",
        g_passed, g_failed, g_info);

    return g_failed == 0 ? 0 : 1;
}
