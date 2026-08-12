#define _POSIX_C_SOURCE 200112L
#define _NETBSD_SOURCE 1

#include <sys/types.h>
#include <sys/stat.h>

#include <errno.h>
#include <float.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define TEST_PASS 0
#define TEST_FAIL 1
#define TEST_INFO 2

#define ARRAY_LEN(a) (sizeof(a) / sizeof((a)[0]))
#define MIB ((size_t)1024 * (size_t)1024)

struct test_context {
    const char *workdir;
    unsigned int max_mib;
    int verbose;
};

struct test_case {
    int number;
    const char *name;
    int (*run)(const struct test_context *ctx);
};

static int g_passed;
static int g_failed;
static int g_info;

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
    if (difference <= scale * relative_tolerance)
        return 1;

    return 0;
}

static uint32_t float_bits(float value)
{
    uint32_t bits;

    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static const char *byte_order_name(void)
{
    union {
        uint32_t word;
        unsigned char bytes[4];
    } value;

    value.word = UINT32_C(0x01020304);
    if (value.bytes[0] == 0x04)
        return "little-endian";
    if (value.bytes[0] == 0x01)
        return "big-endian";
    return "mixed/unknown";
}

static int make_path(char *buffer, size_t buffer_size,
    const char *workdir, const char *stem)
{
    int written;

    written = snprintf(buffer, buffer_size, "%s/%s-%ld.bin",
        workdir, stem, (long)getpid());
    if (written < 0 || (size_t)written >= buffer_size)
        return -1;
    return 0;
}

static int write_bytes(const char *path, const void *data, size_t length)
{
    FILE *stream;
    size_t written;
    int close_status;

    stream = fopen(path, "wb");
    if (stream == NULL) {
        fprintf(stderr, "fopen(%s, wb): %s\n", path, strerror(errno));
        return -1;
    }

    written = fwrite(data, 1, length, stream);
    if (written != length) {
        fprintf(stderr, "fwrite(%s): wrote %lu of %lu bytes\n",
            path, (unsigned long)written, (unsigned long)length);
        fclose(stream);
        return -1;
    }

    if (fflush(stream) != 0) {
        fprintf(stderr, "fflush(%s): %s\n", path, strerror(errno));
        fclose(stream);
        return -1;
    }

    close_status = fclose(stream);
    if (close_status != 0) {
        fprintf(stderr, "fclose(%s): %s\n", path, strerror(errno));
        return -1;
    }

    return 0;
}

static int test_abi(const struct test_context *ctx)
{
    void *pointer;
    uintptr_t address;
    int failed;

    (void)ctx;
    failed = 0;

    printf("char=%lu short=%lu int=%lu long=%lu long long=%lu\n",
        (unsigned long)sizeof(char), (unsigned long)sizeof(short),
        (unsigned long)sizeof(int), (unsigned long)sizeof(long),
        (unsigned long)sizeof(long long));
    printf("pointer=%lu size_t=%lu off_t=%lu time_t=%lu clock_t=%lu\n",
        (unsigned long)sizeof(void *), (unsigned long)sizeof(size_t),
        (unsigned long)sizeof(off_t), (unsigned long)sizeof(time_t),
        (unsigned long)sizeof(clock_t));
    printf("float=%lu double=%lu long double=%lu\n",
        (unsigned long)sizeof(float), (unsigned long)sizeof(double),
        (unsigned long)sizeof(long double));
    printf("FLT_RADIX=%d FLT_MANT_DIG=%d FLT_MAX_EXP=%d\n",
        FLT_RADIX, FLT_MANT_DIG, FLT_MAX_EXP);
    printf("byte order: %s\n", byte_order_name());

    if (CHAR_BIT != 8) {
        fprintf(stderr, "Expected 8-bit bytes, got CHAR_BIT=%d\n", CHAR_BIT);
        failed = 1;
    }
    if (sizeof(uint32_t) != 4 || sizeof(float) != 4) {
        fprintf(stderr, "Stage 1 binary format requires 32-bit uint32_t and float\n");
        failed = 1;
    }
    if (FLT_RADIX != 2 || FLT_MANT_DIG != 24) {
        fprintf(stderr, "Stage 1 expects IEEE-like binary32 float semantics\n");
        failed = 1;
    }

    pointer = malloc(64);
    if (pointer == NULL) {
        fprintf(stderr, "malloc(64) failed: %s\n", strerror(errno));
        failed = 1;
    } else {
        address = (uintptr_t)pointer;
        printf("malloc alignment: address modulo 4=%lu, 8=%lu, 16=%lu\n",
            (unsigned long)(address % 4U),
            (unsigned long)(address % 8U),
            (unsigned long)(address % 16U));
        if ((address % 4U) != 0U) {
            fprintf(stderr, "malloc result is not 4-byte aligned\n");
            failed = 1;
        }
        free(pointer);
    }

    return failed ? TEST_FAIL : TEST_PASS;
}

static int test_fp32_basic(const struct test_context *ctx)
{
    static const float left[] = { 1.5f, -2.0f, 0.25f, 1000.0f };
    static const float right[] = { 2.0f, 4.0f, -8.0f, 0.001f };
    static const float expected_product[] = { 3.0f, -8.0f, -2.0f, 1.0f };
    float sum;
    size_t index;
    int failed;

    (void)ctx;
    failed = 0;
    sum = 0.0f;

    for (index = 0; index < ARRAY_LEN(left); index++) {
        float product;

        product = left[index] * right[index];
        printf("product[%lu]=%.9g expected=%.9g bits=0x%08lx\n",
            (unsigned long)index, (double)product,
            (double)expected_product[index],
            (unsigned long)float_bits(product));
        if (!nearly_equal(product, expected_product[index], 1.0e-6f, 1.0e-6f))
            failed = 1;
        sum += product;
    }

    printf("accumulated sum=%.9g expected=-6\n", (double)sum);
    if (!nearly_equal(sum, -6.0f, 1.0e-6f, 1.0e-6f))
        failed = 1;

    return failed ? TEST_FAIL : TEST_PASS;
}

struct unary_reference {
    const char *name;
    float (*function)(float);
    float input;
    float expected;
    float abs_tol;
    float rel_tol;
};

static int test_libm(const struct test_context *ctx)
{
    static const struct unary_reference references[] = {
        { "expf(0)", expf, 0.0f, 1.0f, 1.0e-6f, 1.0e-6f },
        { "expf(1)", expf, 1.0f, 2.7182817459106445f, 2.0e-6f, 2.0e-6f },
        { "expf(-1)", expf, -1.0f, 0.3678794503211975f, 2.0e-6f, 2.0e-6f },
        { "logf(1)", logf, 1.0f, 0.0f, 1.0e-6f, 1.0e-6f },
        { "logf(e)", logf, 2.7182817459106445f, 1.0f, 2.0e-6f, 2.0e-6f },
        { "sqrtf(2)", sqrtf, 2.0f, 1.4142135381698608f, 2.0e-6f, 2.0e-6f },
        { "tanhf(1)", tanhf, 1.0f, 0.7615941762924194f, 2.0e-6f, 2.0e-6f }
    };
    size_t index;
    int failed;
    float result;

    (void)ctx;
    failed = 0;

    for (index = 0; index < ARRAY_LEN(references); index++) {
        result = references[index].function(references[index].input);
        printf("%-12s actual=% .9g expected=% .9g\n",
            references[index].name, (double)result,
            (double)references[index].expected);
        if (!nearly_equal(result, references[index].expected,
            references[index].abs_tol, references[index].rel_tol)) {
            failed = 1;
        }
    }

    result = powf(2.0f, 10.0f);
    printf("powf(2,10) actual=% .9g expected=1024\n", (double)result);
    if (!nearly_equal(result, 1024.0f, 1.0e-4f, 1.0e-6f))
        failed = 1;

    return failed ? TEST_FAIL : TEST_PASS;
}

static int test_nan_inf(const struct test_context *ctx)
{
    volatile float zero;
    float positive_inf;
    float negative_inf;
    float nan_value;
    float negative_zero;
    int failed;

    (void)ctx;
    failed = 0;
    zero = 0.0f;
    positive_inf = 1.0f / zero;
    negative_inf = -1.0f / zero;
    nan_value = zero / zero;
    negative_zero = -0.0f;

    printf("+inf classification: isinf=%d signbit=%d\n",
        isinf(positive_inf) != 0, signbit(positive_inf) != 0);
    printf("-inf classification: isinf=%d signbit=%d\n",
        isinf(negative_inf) != 0, signbit(negative_inf) != 0);
    printf("nan classification: isnan=%d\n", isnan(nan_value) != 0);
    printf("-0 classification: signbit=%d bits=0x%08lx\n",
        signbit(negative_zero) != 0,
        (unsigned long)float_bits(negative_zero));

    if (!isinf(positive_inf) || signbit(positive_inf))
        failed = 1;
    if (!isinf(negative_inf) || !signbit(negative_inf))
        failed = 1;
    if (!isnan(nan_value))
        failed = 1;
    if (!signbit(negative_zero))
        failed = 1;

    errno = 0;
    nan_value = sqrtf(-1.0f);
    printf("sqrtf(-1): isnan=%d errno=%d\n",
        isnan(nan_value) != 0, errno);
    if (!isnan(nan_value))
        failed = 1;

    errno = 0;
    negative_inf = logf(0.0f);
    printf("logf(0): isinf=%d signbit=%d errno=%d\n",
        isinf(negative_inf) != 0, signbit(negative_inf) != 0, errno);
    if (!isinf(negative_inf) || !signbit(negative_inf))
        failed = 1;

    return failed ? TEST_FAIL : TEST_PASS;
}

static void touch_allocation(unsigned char *memory, size_t length)
{
    const size_t stride = 4096;
    size_t offset;

    for (offset = 0; offset < length; offset += stride)
        memory[offset] = (unsigned char)((offset / stride) & 0xffU);
    if (length > 0)
        memory[length - 1] = 0xa5U;
}

static int verify_allocation(const unsigned char *memory, size_t length)
{
    const size_t stride = 4096;
    size_t offset;

    for (offset = 0; offset < length; offset += stride) {
        unsigned char expected;

        expected = (unsigned char)((offset / stride) & 0xffU);
        if (memory[offset] != expected)
            return -1;
    }
    if (length > 0 && memory[length - 1] != 0xa5U)
        return -1;
    return 0;
}

static int test_heap_growth(const struct test_context *ctx)
{
    static const unsigned int steps[] = { 1, 4, 16, 32, 64, 128, 256 };
    size_t index;
    int failed;
    unsigned int tested;

    failed = 0;
    tested = 0;

    for (index = 0; index < ARRAY_LEN(steps); index++) {
        unsigned int mib;
        size_t bytes;
        unsigned char *memory;

        mib = steps[index];
        if (mib > ctx->max_mib)
            continue;

        bytes = (size_t)mib * MIB;
        errno = 0;
        memory = (unsigned char *)malloc(bytes);
        if (memory == NULL) {
            fprintf(stderr, "malloc(%u MiB) failed: errno=%d (%s)\n",
                mib, errno, strerror(errno));
            failed = 1;
            continue;
        }

        touch_allocation(memory, bytes);
        if (verify_allocation(memory, bytes) != 0) {
            fprintf(stderr, "memory verification failed at %u MiB\n", mib);
            failed = 1;
        } else {
            printf("allocated, touched and verified %u MiB\n", mib);
        }
        free(memory);
        tested++;
    }

    if (tested == 0) {
        fprintf(stderr, "No heap sizes were tested\n");
        failed = 1;
    }

    return failed ? TEST_FAIL : TEST_PASS;
}

static int test_allocation_failure(const struct test_context *ctx)
{
    volatile size_t maximum_source;
    size_t maximum;
    size_t count;
    void *pointer;
    int failed;

    (void)ctx;
    failed = 0;
    maximum_source = (size_t)-1;
    maximum = maximum_source;

    count = maximum / 16U + 1U;
    errno = 0;
    pointer = calloc(count, 16U);
    if (pointer != NULL) {
        fprintf(stderr,
            "calloc overflow request unexpectedly returned non-NULL; freeing it\n");
        free(pointer);
        failed = 1;
    } else {
        printf("calloc overflow request rejected: errno=%d (%s)\n",
            errno, strerror(errno));
    }

    errno = 0;
    pointer = malloc(maximum - 4095U);
    if (pointer != NULL) {
        printf("warning: near-SIZE_MAX malloc returned non-NULL; freeing without touching\n");
        free(pointer);
    } else {
        printf("near-SIZE_MAX malloc rejected: errno=%d (%s)\n",
            errno, strerror(errno));
    }

    return failed ? TEST_FAIL : TEST_PASS;
}

static int test_binary_roundtrip(const struct test_context *ctx)
{
    static const uint32_t patterns[] = {
        UINT32_C(0x00000000),
        UINT32_C(0x80000000),
        UINT32_C(0x3f800000),
        UINT32_C(0xbf800000),
        UINT32_C(0x00800000),
        UINT32_C(0x00000001),
        UINT32_C(0x7f7fffff),
        UINT32_C(0x7fc00001)
    };
    char path[512];
    uint32_t readback[ARRAY_LEN(patterns)];
    FILE *stream;
    size_t count;
    int failed;

    failed = 0;
    if (make_path(path, sizeof(path), ctx->workdir, "ai-stage1-roundtrip") != 0) {
        fprintf(stderr, "roundtrip path is too long\n");
        return TEST_FAIL;
    }

    if (write_bytes(path, patterns, sizeof(patterns)) != 0)
        return TEST_FAIL;

    memset(readback, 0, sizeof(readback));
    stream = fopen(path, "rb");
    if (stream == NULL) {
        fprintf(stderr, "fopen(%s, rb): %s\n", path, strerror(errno));
        unlink(path);
        return TEST_FAIL;
    }

    count = fread(readback, sizeof(readback[0]), ARRAY_LEN(readback), stream);
    if (count != ARRAY_LEN(readback)) {
        fprintf(stderr, "fread returned %lu elements, expected %lu\n",
            (unsigned long)count, (unsigned long)ARRAY_LEN(readback));
        failed = 1;
    }
    if (ferror(stream)) {
        fprintf(stderr, "fread set stream error\n");
        failed = 1;
    }
    if (memcmp(patterns, readback, sizeof(patterns)) != 0) {
        fprintf(stderr, "roundtrip bytes differ\n");
        failed = 1;
    }

    if (fclose(stream) != 0) {
        fprintf(stderr, "fclose(%s): %s\n", path, strerror(errno));
        failed = 1;
    }
    if (unlink(path) != 0)
        fprintf(stderr, "warning: unlink(%s): %s\n", path, strerror(errno));

    if (!failed)
        printf("exactly reproduced %lu bytes of FP32 bit patterns\n",
            (unsigned long)sizeof(patterns));

    return failed ? TEST_FAIL : TEST_PASS;
}

struct tensor_header {
    uint32_t magic;
    uint32_t version;
    uint32_t count;
};

static int test_truncated_file(const struct test_context *ctx)
{
    char path[512];
    FILE *stream;
    struct tensor_header header;
    float partial_data[2];
    float destination[4];
    size_t count;
    int failed;

    failed = 0;
    header.magic = UINT32_C(0x3149414d); /* "MAI1" in little-endian bytes. */
    header.version = 1;
    header.count = 4;
    partial_data[0] = 1.0f;
    partial_data[1] = 2.0f;

    if (make_path(path, sizeof(path), ctx->workdir, "ai-stage1-truncated") != 0)
        return TEST_FAIL;

    stream = fopen(path, "wb");
    if (stream == NULL) {
        fprintf(stderr, "fopen(%s, wb): %s\n", path, strerror(errno));
        return TEST_FAIL;
    }
    if (fwrite(&header, sizeof(header), 1, stream) != 1 ||
        fwrite(partial_data, sizeof(partial_data[0]), 2, stream) != 2) {
        fprintf(stderr, "failed to create truncated input\n");
        fclose(stream);
        unlink(path);
        return TEST_FAIL;
    }
    fclose(stream);

    stream = fopen(path, "rb");
    if (stream == NULL) {
        fprintf(stderr, "fopen(%s, rb): %s\n", path, strerror(errno));
        unlink(path);
        return TEST_FAIL;
    }

    if (fread(&header, sizeof(header), 1, stream) != 1) {
        fprintf(stderr, "could not read tensor header\n");
        failed = 1;
    } else {
        count = fread(destination, sizeof(destination[0]), header.count, stream);
        printf("declared elements=%lu, actual elements read=%lu, feof=%d, ferror=%d\n",
            (unsigned long)header.count, (unsigned long)count,
            feof(stream) != 0, ferror(stream) != 0);
        if (count == header.count) {
            fprintf(stderr, "truncated input was not detected\n");
            failed = 1;
        }
    }

    fclose(stream);
    unlink(path);
    return failed ? TEST_FAIL : TEST_PASS;
}

static int get_regular_file_size(FILE *stream, off_t *size_out)
{
    struct stat status;
    int descriptor;

    descriptor = fileno(stream);
    if (descriptor < 0)
        return -1;
    if (fstat(descriptor, &status) != 0)
        return -1;
    if (!S_ISREG(status.st_mode)) {
        errno = EINVAL;
        return -1;
    }
    *size_out = status.st_size;
    return 0;
}

static int test_fseeko_bounds(const struct test_context *ctx)
{
    static const unsigned char payload[32] = {
        0, 1, 2, 3, 4, 5, 6, 7,
        8, 9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23,
        24, 25, 26, 27, 28, 29, 30, 31
    };
    char path[512];
    FILE *stream;
    off_t length;
    int have_length;
    off_t position;
    int failed;

    failed = 0;
    length = 0;
    have_length = 0;
    if (make_path(path, sizeof(path), ctx->workdir, "ai-stage1-seek") != 0)
        return TEST_FAIL;
    if (write_bytes(path, payload, sizeof(payload)) != 0)
        return TEST_FAIL;

    stream = fopen(path, "rb");
    if (stream == NULL) {
        fprintf(stderr, "fopen(%s, rb): %s\n", path, strerror(errno));
        unlink(path);
        return TEST_FAIL;
    }

    if (get_regular_file_size(stream, &length) != 0) {
        fprintf(stderr, "fstat regular-file length failed: %s\n", strerror(errno));
        failed = 1;
    } else {
        have_length = 1;
        printf("regular file length=%lld bytes\n", (long long)length);
        if (length != (off_t)sizeof(payload))
            failed = 1;
    }

    if (fseeko(stream, 0, SEEK_END) != 0) {
        fprintf(stderr, "fseeko(SEEK_END): %s\n", strerror(errno));
        failed = 1;
    } else {
        position = ftello(stream);
        printf("ftello after SEEK_END=%lld\n", (long long)position);
        if (position != (off_t)sizeof(payload))
            failed = 1;
    }

    /* Loader-side check: reject an out-of-range offset before seeking. */
    if (have_length) {
        off_t requested_offset;

        requested_offset = (off_t)33;
        if (requested_offset >= 0 && requested_offset <= length) {
            fprintf(stderr, "loader bounds check logic is incorrect\n");
            failed = 1;
        } else {
            printf("loader-side bounds check rejected offset 33 for a 32-byte file\n");
        }
    }

    fclose(stream);
    unlink(path);
    return failed ? TEST_FAIL : TEST_PASS;
}

static int test_negative_fseek(const struct test_context *ctx)
{
    static const unsigned char payload[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    char path[512];
    FILE *stream;
    int seek_result;
    int saved_errno;
    long position;
    size_t read_count;
    unsigned char byte;

    if (make_path(path, sizeof(path), ctx->workdir, "ai-stage1-negseek") != 0)
        return TEST_FAIL;
    if (write_bytes(path, payload, sizeof(payload)) != 0)
        return TEST_FAIL;

    stream = fopen(path, "rb");
    if (stream == NULL) {
        fprintf(stderr, "fopen(%s, rb): %s\n", path, strerror(errno));
        unlink(path);
        return TEST_FAIL;
    }

    errno = 0;
    seek_result = fseek(stream, -1L, SEEK_SET);
    saved_errno = errno;
    position = ftell(stream);
    read_count = fread(&byte, 1, 1, stream);

    printf("fseek(-1, SEEK_SET): result=%d errno=%d (%s) ftell=%ld fread=%lu\n",
        seek_result, saved_errno, strerror(saved_errno), position,
        (unsigned long)read_count);

    fclose(stream);
    unlink(path);

    puts("This test is characterisation only; the Stage 1 loader must reject negative offsets itself.");
    return TEST_INFO;
}

static int64_t timespec_to_ns(const struct timespec *value)
{
    return (int64_t)value->tv_sec * INT64_C(1000000000) +
        (int64_t)value->tv_nsec;
}

static int test_monotonic_clock(const struct test_context *ctx)
{
#ifdef CLOCK_MONOTONIC
    struct timespec resolution;
    struct timespec previous;
    struct timespec current;
    struct timespec start;
    struct timespec end;
    int64_t minimum_positive_delta;
    int64_t delta;
    unsigned long index;
    volatile unsigned long accumulator;
    int failed;

    (void)ctx;
    failed = 0;
    minimum_positive_delta = INT64_MAX;

    if (clock_getres(CLOCK_MONOTONIC, &resolution) != 0) {
        fprintf(stderr, "clock_getres(CLOCK_MONOTONIC): %s\n", strerror(errno));
        return TEST_FAIL;
    }
    printf("reported CLOCK_MONOTONIC resolution: %ld.%09ld seconds\n",
        (long)resolution.tv_sec, (long)resolution.tv_nsec);

    if (clock_gettime(CLOCK_MONOTONIC, &previous) != 0) {
        fprintf(stderr, "clock_gettime(CLOCK_MONOTONIC): %s\n", strerror(errno));
        return TEST_FAIL;
    }

    for (index = 0; index < 10000UL; index++) {
        if (clock_gettime(CLOCK_MONOTONIC, &current) != 0) {
            fprintf(stderr, "clock_gettime failed during sample loop\n");
            return TEST_FAIL;
        }
        delta = timespec_to_ns(&current) - timespec_to_ns(&previous);
        if (delta < 0) {
            fprintf(stderr, "CLOCK_MONOTONIC moved backwards by %" PRId64 " ns\n",
                -delta);
            failed = 1;
            break;
        }
        if (delta > 0 && delta < minimum_positive_delta)
            minimum_positive_delta = delta;
        previous = current;
    }

    if (minimum_positive_delta == INT64_MAX)
        puts("No positive delta observed in tight loop; timer is coarser than loop duration.");
    else
        printf("minimum positive observed delta: %" PRId64 " ns\n",
            minimum_positive_delta);

    if (clock_gettime(CLOCK_MONOTONIC, &start) != 0)
        return TEST_FAIL;
    accumulator = 0;
    for (index = 0; index < 5000000UL; index++)
        accumulator += index ^ (index >> 3);
    if (clock_gettime(CLOCK_MONOTONIC, &end) != 0)
        return TEST_FAIL;

    delta = timespec_to_ns(&end) - timespec_to_ns(&start);
    printf("busy-loop elapsed time: %" PRId64 " ns (accumulator=%lu)\n",
        delta, accumulator);
    if (delta < 0)
        failed = 1;

    return failed ? TEST_FAIL : TEST_PASS;
#else
    (void)ctx;
    fprintf(stderr, "CLOCK_MONOTONIC is not defined by the build environment\n");
    return TEST_FAIL;
#endif
}

static const struct test_case tests[] = {
    { 1, "ABI and type-size report", test_abi },
    { 2, "basic FP32 arithmetic", test_fp32_basic },
    { 3, "libm conformance", test_libm },
    { 4, "NaN and infinity handling", test_nan_inf },
    { 5, "progressively larger heap allocations", test_heap_growth },
    { 6, "calloc overflow and allocation failure", test_allocation_failure },
    { 7, "binary FP32 write/read round trip", test_binary_roundtrip },
    { 8, "truncated-file handling", test_truncated_file },
    { 9, "fseeko and file-length validation", test_fseeko_bounds },
    { 10, "fseek negative SEEK_SET characterisation", test_negative_fseek },
    { 11, "CLOCK_MONOTONIC resolution and ordering", test_monotonic_clock }
};

static void usage(const char *program)
{
    printf("Usage: %s [--max-mib N] [--workdir DIR] [--test N] [--list] [--verbose]\n",
        program);
    puts("\nDefaults:");
    puts("  --max-mib 128");
    puts("  --workdir .");
}

static void list_tests(void)
{
    size_t index;

    for (index = 0; index < ARRAY_LEN(tests); index++)
        printf("%2d  %s\n", tests[index].number, tests[index].name);
}

static int parse_unsigned(const char *text, unsigned int *value_out)
{
    char *end;
    unsigned long value;

    errno = 0;
    value = strtoul(text, &end, 10);
    if (errno != 0 || *text == '\0' || *end != '\0' || value > UINT_MAX)
        return -1;
    *value_out = (unsigned int)value;
    return 0;
}

int main(int argc, char **argv)
{
    struct test_context context;
    int selected_test;
    int index;
    size_t test_index;
    int ran_any;

    context.workdir = ".";
    context.max_mib = 128;
    context.verbose = 0;
    selected_test = 0;
    ran_any = 0;

    for (index = 1; index < argc; index++) {
        if (strcmp(argv[index], "--max-mib") == 0) {
            if (++index >= argc || parse_unsigned(argv[index], &context.max_mib) != 0) {
                fprintf(stderr, "Invalid --max-mib value\n");
                return 2;
            }
        } else if (strcmp(argv[index], "--workdir") == 0) {
            if (++index >= argc) {
                fprintf(stderr, "Missing --workdir value\n");
                return 2;
            }
            context.workdir = argv[index];
        } else if (strcmp(argv[index], "--test") == 0) {
            unsigned int value;

            if (++index >= argc || parse_unsigned(argv[index], &value) != 0) {
                fprintf(stderr, "Invalid --test value\n");
                return 2;
            }
            selected_test = (int)value;
        } else if (strcmp(argv[index], "--list") == 0) {
            list_tests();
            return 0;
        } else if (strcmp(argv[index], "--verbose") == 0) {
            context.verbose = 1;
        } else if (strcmp(argv[index], "--help") == 0 ||
            strcmp(argv[index], "-h") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[index]);
            usage(argv[0]);
            return 2;
        }
    }

    printf("MINIX AI Stage 1 platform readiness probe\n");
    printf("workdir=%s max_mib=%u selected_test=%d\n",
        context.workdir, context.max_mib, selected_test);
    print_rule();

    for (test_index = 0; test_index < ARRAY_LEN(tests); test_index++) {
        int result;

        if (selected_test != 0 && tests[test_index].number != selected_test)
            continue;

        printf("\nTest %d: %s\n", tests[test_index].number, tests[test_index].name);
        print_rule();
        result = tests[test_index].run(&context);
        report_result(tests[test_index].number, tests[test_index].name, result);
        ran_any = 1;
    }

    if (!ran_any) {
        fprintf(stderr, "No such test: %d\n", selected_test);
        return 2;
    }

    print_rule();
    printf("Summary: %d passed, %d failed, %d informational\n",
        g_passed, g_failed, g_info);

    return g_failed == 0 ? 0 : 1;
}
