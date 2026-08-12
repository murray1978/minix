#include <errno.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ADAPTER_MAGIC "MLAD"
#define ADAPTER_VERSION 2U
#define SHA256_HEX_LEN 64

/* Verified Stage 3 base checkpoint identity. */
static const char *k_expected_checkpoint_sha256 =
	"cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a";

typedef struct {
	int dim;
	int hidden_dim;
	int n_layers;
	int n_heads;
	int n_kv_heads;
	int vocab_size;
	int seq_len;
	int shared_weights;
} Config;

typedef struct {
	char magic[4];
	uint32_t version;
	uint32_t rank;
	uint32_t dim;
	uint32_t n_heads;
	uint32_t n_kv_heads;
	uint32_t hidden_dim;
	uint32_t n_layers;
	uint32_t vocab_size;
	uint32_t seq_len;
	uint32_t reserved0;
	uint64_t checkpoint_bytes;
	uint8_t checkpoint_sha256[32];
	uint32_t layout_version;
	uint32_t reserved1;
	uint64_t a_params;
	uint64_t b_params;
	uint64_t a_bytes;
	uint64_t b_bytes;
	uint64_t params_count;
	uint64_t data_bytes;
} AdapterFileHeader;

typedef struct {
	uint64_t a_params;
	uint64_t b_params;
	uint64_t total_params;
	uint64_t a_bytes;
	uint64_t b_bytes;
	uint64_t total_bytes;
} AdapterLayout;

typedef struct {
	uint8_t bytes[32];
} Sha256Digest;

typedef struct {
	uint32_t state[8];
	uint64_t total_bytes;
	uint8_t buffer[64];
	size_t buffer_used;
} Sha256Ctx;

static void usage(const char *prog)
{
	fprintf(stderr,
	    "usage: %s <checkpoint.bin> <adapter.bin> [-r rank]\n"
	    "       %s --inspect <checkpoint.bin> <adapter.bin>\n"
	    "  default rank: 8\n"
	    "  output: zero-initialized adapter weights sidecar\n"
	    "  inspect mode validates adapter content without writing\n",
	    prog, prog);
}

static uint32_t rotr32(uint32_t value, unsigned int bits)
{
	return (value >> bits) | (value << (32U - bits));
}

static uint32_t be32(const uint8_t *p)
{
	return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
	    ((uint32_t)p[2] << 8) | (uint32_t)p[3];
}

static void store_be32(uint8_t *p, uint32_t value)
{
	p[0] = (uint8_t)(value >> 24);
	p[1] = (uint8_t)(value >> 16);
	p[2] = (uint8_t)(value >> 8);
	p[3] = (uint8_t)value;
}

static void sha256_compress(Sha256Ctx *ctx, const uint8_t block[64])
{
	static const uint32_t k[64] = {
		0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
		0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
		0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
		0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
		0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
		0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
		0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
		0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
		0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
		0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
		0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
		0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
		0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
		0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
		0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
		0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U
	};
	uint32_t w[64];
	uint32_t a, b, c, d, e, f, g, h;
	int i;

	for (i = 0; i < 16; i++)
		w[i] = be32(block + i * 4);

	for (i = 16; i < 64; i++) {
		uint32_t s0 = rotr32(w[i - 15], 7) ^ rotr32(w[i - 15], 18) ^
		    (w[i - 15] >> 3);
		uint32_t s1 = rotr32(w[i - 2], 17) ^ rotr32(w[i - 2], 19) ^
		    (w[i - 2] >> 10);
		w[i] = w[i - 16] + s0 + w[i - 7] + s1;
	}

	a = ctx->state[0];
	b = ctx->state[1];
	c = ctx->state[2];
	d = ctx->state[3];
	e = ctx->state[4];
	f = ctx->state[5];
	g = ctx->state[6];
	h = ctx->state[7];

	for (i = 0; i < 64; i++) {
		uint32_t s1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
		uint32_t ch = (e & f) ^ ((~e) & g);
		uint32_t t1 = h + s1 + ch + k[i] + w[i];
		uint32_t s0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
		uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
		uint32_t t2 = s0 + maj;

		h = g;
		g = f;
		f = e;
		e = d + t1;
		d = c;
		c = b;
		b = a;
		a = t1 + t2;
	}

	ctx->state[0] += a;
	ctx->state[1] += b;
	ctx->state[2] += c;
	ctx->state[3] += d;
	ctx->state[4] += e;
	ctx->state[5] += f;
	ctx->state[6] += g;
	ctx->state[7] += h;
}

static void sha256_init(Sha256Ctx *ctx)
{
	memset(ctx, 0, sizeof(*ctx));
	ctx->state[0] = 0x6a09e667U;
	ctx->state[1] = 0xbb67ae85U;
	ctx->state[2] = 0x3c6ef372U;
	ctx->state[3] = 0xa54ff53aU;
	ctx->state[4] = 0x510e527fU;
	ctx->state[5] = 0x9b05688cU;
	ctx->state[6] = 0x1f83d9abU;
	ctx->state[7] = 0x5be0cd19U;
}

static void sha256_update(Sha256Ctx *ctx, const void *data, size_t len)
{
	const uint8_t *p = (const uint8_t *)data;

	ctx->total_bytes += (uint64_t)len;
	while (len > 0) {
		size_t space = sizeof(ctx->buffer) - ctx->buffer_used;
		size_t n = len < space ? len : space;

		memcpy(ctx->buffer + ctx->buffer_used, p, n);
		ctx->buffer_used += n;
		p += n;
		len -= n;

		if (ctx->buffer_used == sizeof(ctx->buffer)) {
			sha256_compress(ctx, ctx->buffer);
			ctx->buffer_used = 0;
		}
	}
}

static void sha256_final(Sha256Ctx *ctx, Sha256Digest *digest)
{
	uint8_t tail[128];
	uint64_t bits = ctx->total_bytes * 8U;
	size_t used = ctx->buffer_used;
	size_t pad_len;
	int i;

	memcpy(tail, ctx->buffer, used);
	tail[used++] = 0x80U;

	pad_len = (used <= 56) ? (56 - used) : (120 - used);
	memset(tail + used, 0, pad_len);
	used += pad_len;

	for (i = 0; i < 8; i++)
		tail[used + i] = (uint8_t)(bits >> (56 - 8 * i));
	used += 8;

	for (i = 0; i < (int)used; i += 64)
		sha256_compress(ctx, tail + i);

	for (i = 0; i < 8; i++)
		store_be32(digest->bytes + i * 4, ctx->state[i]);
}

static void sha256_to_hex(const Sha256Digest *digest, char out[SHA256_HEX_LEN + 1])
{
	static const char *hex = "0123456789abcdef";
	int i;

	for (i = 0; i < 32; i++) {
		out[i * 2] = hex[(digest->bytes[i] >> 4) & 0xF];
		out[i * 2 + 1] = hex[digest->bytes[i] & 0xF];
	}
	out[SHA256_HEX_LEN] = '\0';
}

static int checked_mul_u64(uint64_t a, uint64_t b, uint64_t *out)
{
	if (out == NULL || (a != 0 && b > UINT64_MAX / a))
		return 0;
	*out = a * b;
	return 1;
}

static int checked_add_u64(uint64_t a, uint64_t b, uint64_t *out)
{
	if (out == NULL || b > UINT64_MAX - a)
		return 0;
	*out = a + b;
	return 1;
}

static int validate_config(const Config *p)
{
	if (p->dim <= 0 || p->hidden_dim <= 0 || p->n_layers <= 0 ||
	    p->n_heads <= 0 || p->n_kv_heads <= 0 ||
	    p->vocab_size <= 3 || p->seq_len <= 0)
		return 0;

	if (p->dim % p->n_heads != 0)
		return 0;
	if (p->n_heads % p->n_kv_heads != 0)
		return 0;
	if ((p->dim / p->n_heads) % 2 != 0)
		return 0;

	return 1;
}

static int read_stage3_config(const char *checkpoint_path, Config *cfg)
{
	FILE *f;
	int32_t disk[7];
	int32_t raw_vocab;

	f = fopen(checkpoint_path, "rb");
	if (f == NULL) {
		fprintf(stderr, "error: opening checkpoint '%s': %s\n",
		    checkpoint_path, strerror(errno));
		return 0;
	}

	if (fread(disk, sizeof(disk), 1, f) != 1) {
		fprintf(stderr, "error: cannot read 28-byte checkpoint header\n");
		fclose(f);
		return 0;
	}

	if (fclose(f) != 0) {
		fprintf(stderr, "error: closing checkpoint '%s': %s\n",
		    checkpoint_path, strerror(errno));
		return 0;
	}

	raw_vocab = disk[5];
	if (raw_vocab == INT32_MIN) {
		fprintf(stderr, "error: invalid signed vocabulary size in checkpoint\n");
		return 0;
	}

	cfg->dim = disk[0];
	cfg->hidden_dim = disk[1];
	cfg->n_layers = disk[2];
	cfg->n_heads = disk[3];
	cfg->n_kv_heads = disk[4];
	cfg->shared_weights = raw_vocab > 0;
	cfg->vocab_size = raw_vocab < 0 ? -raw_vocab : raw_vocab;
	cfg->seq_len = disk[6];

	if (!validate_config(cfg)) {
		fprintf(stderr, "error: invalid or unsupported checkpoint dimensions\n");
		return 0;
	}

	return 1;
}

static int file_size_bytes(const char *path, uint64_t *size_out)
{
	FILE *f;
	long size;

	f = fopen(path, "rb");
	if (f == NULL)
		return 0;

	if (fseek(f, 0, SEEK_END) != 0) {
		fclose(f);
		return 0;
	}
	size = ftell(f);
	fclose(f);

	if (size < 0)
		return 0;
	*size_out = (uint64_t)size;
	return 1;
}

static int hash_file_sha256(const char *path, Sha256Digest *digest)
{
	FILE *f;
	unsigned char buffer[4096];
	Sha256Ctx ctx;
	size_t n;

	f = fopen(path, "rb");
	if (f == NULL)
		return 0;

	sha256_init(&ctx);
	for (;;) {
		n = fread(buffer, 1, sizeof(buffer), f);
		if (n > 0)
			sha256_update(&ctx, buffer, n);
		if (n < sizeof(buffer)) {
			if (ferror(f)) {
				fclose(f);
				return 0;
			}
			break;
		}
	}

	if (fclose(f) != 0)
		return 0;

	sha256_final(&ctx, digest);
	return 1;
}

static int compute_layout(const Config *cfg, uint32_t rank, AdapterLayout *layout)
{
	uint64_t a_params;
	uint64_t b_params;
	uint64_t total_params;
	uint64_t a_bytes;
	uint64_t b_bytes;
	uint64_t total_bytes;

	if (rank == 0)
		return 0;

	if (!checked_mul_u64((uint64_t)rank, (uint64_t)cfg->dim, &a_params))
		return 0;
	if (!checked_mul_u64((uint64_t)cfg->vocab_size, (uint64_t)rank, &b_params))
		return 0;
	if (!checked_add_u64(a_params, b_params, &total_params))
		return 0;

	if (!checked_mul_u64(a_params, (uint64_t)sizeof(float), &a_bytes))
		return 0;
	if (!checked_mul_u64(b_params, (uint64_t)sizeof(float), &b_bytes))
		return 0;
	if (!checked_add_u64(a_bytes, b_bytes, &total_bytes))
		return 0;

	layout->a_params = a_params;
	layout->b_params = b_params;
	layout->total_params = total_params;
	layout->a_bytes = a_bytes;
	layout->b_bytes = b_bytes;
	layout->total_bytes = total_bytes;
	return 1;
}

static void print_layout_accounting(const Config *cfg, uint32_t rank,
	const AdapterLayout *layout)
{
	printf("adapter tensor layout\n");
	printf("  A: rank x dim = %u x %d, params=%llu, bytes=%llu\n",
	    rank, cfg->dim,
	    (unsigned long long)layout->a_params,
	    (unsigned long long)layout->a_bytes);
	printf("  B: vocab x rank = %d x %u, params=%llu, bytes=%llu\n",
	    cfg->vocab_size, rank,
	    (unsigned long long)layout->b_params,
	    (unsigned long long)layout->b_bytes);
	printf("  total params=%llu bytes=%llu\n",
	    (unsigned long long)layout->total_params,
	    (unsigned long long)layout->total_bytes);
}

static int validate_adapter_file(const char *adapter_path,
	const Config *cfg,
	uint32_t expected_rank,
	const Sha256Digest *checkpoint_sha,
	uint64_t checkpoint_bytes,
	int require_b_zero,
	AdapterFileHeader *header_out,
	Sha256Digest *adapter_sha_out)
{
	FILE *f;
	AdapterFileHeader hdr;
	AdapterLayout layout;
	uint64_t expected_file_bytes;
	uint64_t actual_file_bytes;
	uint64_t float_count;
	uint64_t i;
	float value;
	char checkpoint_hex[SHA256_HEX_LEN + 1];
	char header_checkpoint_hex[SHA256_HEX_LEN + 1];

	if (!compute_layout(cfg, expected_rank, &layout)) {
		fprintf(stderr, "error: layout overflow while validating adapter\n");
		return 0;
	}

	if (!file_size_bytes(adapter_path, &actual_file_bytes)) {
		fprintf(stderr, "error: cannot stat adapter '%s'\n", adapter_path);
		return 0;
	}

	f = fopen(adapter_path, "rb");
	if (f == NULL) {
		fprintf(stderr, "error: opening adapter '%s': %s\n",
		    adapter_path, strerror(errno));
		return 0;
	}

	if (fread(&hdr, sizeof(hdr), 1, f) != 1) {
		fprintf(stderr, "error: cannot read adapter header from '%s'\n",
		    adapter_path);
		fclose(f);
		return 0;
	}

	if (memcmp(hdr.magic, ADAPTER_MAGIC, 4) != 0) {
		fprintf(stderr, "error: invalid adapter magic\n");
		fclose(f);
		return 0;
	}
	if (hdr.version != ADAPTER_VERSION) {
		fprintf(stderr, "error: unsupported adapter version %u\n", hdr.version);
		fclose(f);
		return 0;
	}
	if (hdr.layout_version != 1U) {
		fprintf(stderr, "error: unsupported adapter layout version %u\n",
		    hdr.layout_version);
		fclose(f);
		return 0;
	}

	if (hdr.rank != expected_rank) {
		fprintf(stderr, "error: adapter rank mismatch: file=%u expected=%u\n",
		    hdr.rank, expected_rank);
		fclose(f);
		return 0;
	}
	if (hdr.dim != (uint32_t)cfg->dim ||
	    hdr.vocab_size != (uint32_t)cfg->vocab_size ||
	    hdr.seq_len != (uint32_t)cfg->seq_len ||
	    hdr.n_heads != (uint32_t)cfg->n_heads ||
	    hdr.n_kv_heads != (uint32_t)cfg->n_kv_heads ||
	    hdr.hidden_dim != (uint32_t)cfg->hidden_dim ||
	    hdr.n_layers != (uint32_t)cfg->n_layers) {
		fprintf(stderr, "error: adapter/model dimension mismatch\n");
		fclose(f);
		return 0;
	}

	if (hdr.checkpoint_bytes != checkpoint_bytes) {
		fprintf(stderr,
		    "error: base checkpoint length mismatch in adapter: file=%llu expected=%llu\n",
		    (unsigned long long)hdr.checkpoint_bytes,
		    (unsigned long long)checkpoint_bytes);
		fclose(f);
		return 0;
	}

	sha256_to_hex(checkpoint_sha, checkpoint_hex);
	{
		Sha256Digest header_sha;
		memcpy(header_sha.bytes, hdr.checkpoint_sha256, 32);
		sha256_to_hex(&header_sha, header_checkpoint_hex);
	}
	if (memcmp(hdr.checkpoint_sha256, checkpoint_sha->bytes, 32) != 0) {
		fprintf(stderr,
		    "error: base checkpoint hash mismatch in adapter\n"
		    "  file:     %s\n"
		    "  expected: %s\n",
		    header_checkpoint_hex, checkpoint_hex);
		fclose(f);
		return 0;
	}

	if (hdr.a_params != layout.a_params ||
	    hdr.b_params != layout.b_params ||
	    hdr.params_count != layout.total_params ||
	    hdr.a_bytes != layout.a_bytes ||
	    hdr.b_bytes != layout.b_bytes ||
	    hdr.data_bytes != layout.total_bytes) {
		fprintf(stderr, "error: adapter tensor accounting mismatch\n");
		fclose(f);
		return 0;
	}

	if (!checked_add_u64((uint64_t)sizeof(hdr), hdr.data_bytes,
		    &expected_file_bytes)) {
		fprintf(stderr, "error: adapter file length overflow\n");
		fclose(f);
		return 0;
	}

	if (actual_file_bytes != expected_file_bytes) {
		fprintf(stderr,
		    "error: adapter length mismatch: file=%llu expected=%llu\n",
		    (unsigned long long)actual_file_bytes,
		    (unsigned long long)expected_file_bytes);
		fclose(f);
		return 0;
	}

	if (hdr.data_bytes % sizeof(float) != 0) {
		fprintf(stderr, "error: adapter payload length is not float-aligned\n");
		fclose(f);
		return 0;
	}

	float_count = hdr.data_bytes / sizeof(float);
	for (i = 0; i < float_count; i++) {
		if (fread(&value, sizeof(value), 1, f) != 1) {
			fprintf(stderr, "error: truncated adapter payload\n");
			fclose(f);
			return 0;
		}
		if (!isfinite(value)) {
			fprintf(stderr,
			    "error: adapter contains non-finite value at float index %llu\n",
			    (unsigned long long)i);
			fclose(f);
			return 0;
		}
		if (require_b_zero && i >= hdr.a_params && i < hdr.a_params + hdr.b_params &&
		    value != 0.0f) {
			fprintf(stderr,
			    "error: adapter B is not zero-initialized at float index %llu\n",
			    (unsigned long long)i);
			fclose(f);
			return 0;
		}
	}

	if (fgetc(f) != EOF) {
		fprintf(stderr, "error: adapter has trailing bytes\n");
		fclose(f);
		return 0;
	}

	if (fclose(f) != 0) {
		fprintf(stderr, "error: closing adapter '%s': %s\n",
		    adapter_path, strerror(errno));
		return 0;
	}

	if (!hash_file_sha256(adapter_path, adapter_sha_out)) {
		fprintf(stderr, "error: hashing adapter '%s' failed\n", adapter_path);
		return 0;
	}

	if (header_out != NULL)
		*header_out = hdr;

	return 1;
}

static int write_zero_weights(FILE *f, size_t bytes)
{
	char zero_block[4096];
	size_t remain = bytes;

	memset(zero_block, 0, sizeof(zero_block));
	while (remain > 0) {
		size_t chunk = remain < sizeof(zero_block) ? remain : sizeof(zero_block);
		if (fwrite(zero_block, 1, chunk, f) != chunk)
			return 0;
		remain -= chunk;
	}
	return 1;
}

static int write_deterministic_a_and_zero_b(FILE *f,
	const AdapterLayout *layout,
	float a_scale)
{
	uint64_t i;

	for (i = 0; i < layout->a_params; i++) {
		uint64_t r = (i % 2047ULL) + 1ULL;
		int32_t centered = (int32_t)r - 1024;
		float value = ((float)centered / 1024.0f) * a_scale;

		if (value == 0.0f)
			value = a_scale * 0.0009765625f;
		if (fwrite(&value, sizeof(value), 1, f) != 1)
			return 0;
	}

	if (!write_zero_weights(f, (size_t)layout->b_bytes))
		return 0;

	return 1;
}

int main(int argc, char **argv)
{
	const char *checkpoint_path;
	const char *adapter_path;
	int inspect_only = 0;
	Config cfg;
	AdapterFileHeader hdr;
	FILE *out;
	AdapterLayout layout;
	uint64_t trainable_bytes;
	uint64_t gradient_bytes;
	uint64_t adam_first_moment_bytes;
	uint64_t adam_second_moment_bytes;
	uint64_t total_trainable_storage_bytes;
	uint64_t tmp_u64;
	uint32_t rank = 8;
	int i;
	char *endp;
	Sha256Digest checkpoint_sha_before;
	Sha256Digest checkpoint_sha_after;
	Sha256Digest adapter_sha;
	char checkpoint_hex_before[SHA256_HEX_LEN + 1];
	char checkpoint_hex_after[SHA256_HEX_LEN + 1];
	char adapter_hex[SHA256_HEX_LEN + 1];
	uint64_t checkpoint_bytes;
	uint64_t adapter_file_bytes;
	const char *expected_mode = "write";

	if (argc < 2) {
		usage(argv[0]);
		return 1;
	}

	if (strcmp(argv[1], "--inspect") == 0) {
		inspect_only = 1;
		expected_mode = "inspect";
		if (argc < 4) {
			usage(argv[0]);
			return 1;
		}
		checkpoint_path = argv[2];
		adapter_path = argv[3];
		i = 4;
	} else {
		if (argc < 3) {
			usage(argv[0]);
			return 1;
		}
		checkpoint_path = argv[1];
		adapter_path = argv[2];
		i = 3;
	}

	for (; i < argc; i++) {
		if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
			unsigned long parsed = strtoul(argv[++i], &endp, 10);
			if (*endp != '\0' || parsed == 0 || parsed > 1024UL) {
				fprintf(stderr, "error: rank must be in range 1..1024\n");
				return 1;
			}
			rank = (uint32_t)parsed;
			continue;
		}
		fprintf(stderr, "error: unknown argument '%s'\n", argv[i]);
		usage(argv[0]);
		return 1;
	}

	if (!read_stage3_config(checkpoint_path, &cfg))
		return 1;

	if (!file_size_bytes(checkpoint_path, &checkpoint_bytes)) {
		fprintf(stderr, "error: cannot get checkpoint length for '%s'\n",
		    checkpoint_path);
		return 1;
	}

	if (!hash_file_sha256(checkpoint_path, &checkpoint_sha_before)) {
		fprintf(stderr, "error: cannot hash checkpoint '%s'\n", checkpoint_path);
		return 1;
	}
	sha256_to_hex(&checkpoint_sha_before, checkpoint_hex_before);

	if (strcmp(checkpoint_hex_before, k_expected_checkpoint_sha256) != 0) {
		fprintf(stderr,
		    "error: checkpoint SHA-256 mismatch\n"
		    "  got:      %s\n"
		    "  expected: %s\n",
		    checkpoint_hex_before, k_expected_checkpoint_sha256);
		return 1;
	}

	if (!compute_layout(&cfg, rank, &layout)) {
		fprintf(stderr, "error: adapter parameter count overflow\n");
		return 1;
	}

	printf("mode: %s\n", expected_mode);
	print_layout_accounting(&cfg, rank, &layout);
	printf("base checkpoint SHA-256 before: %s\n", checkpoint_hex_before);
	printf("base checkpoint bytes: %llu\n",
	    (unsigned long long)checkpoint_bytes);

	if (inspect_only) {
		if (!validate_adapter_file(adapter_path, &cfg, rank,
		    &checkpoint_sha_before, checkpoint_bytes,
		    0, &hdr, &adapter_sha))
			return 1;

		sha256_to_hex(&adapter_sha, adapter_hex);
		printf("adapter inspection passed\n");
		printf("adapter file SHA-256: %s\n", adapter_hex);
		printf("round-trip reload: PASS\n");
		return 0;
	}

	memset(&hdr, 0, sizeof(hdr));
	memcpy(hdr.magic, ADAPTER_MAGIC, sizeof(hdr.magic));
	hdr.version = ADAPTER_VERSION;
	hdr.rank = rank;
	hdr.dim = (uint32_t)cfg.dim;
	hdr.n_heads = (uint32_t)cfg.n_heads;
	hdr.n_kv_heads = (uint32_t)cfg.n_kv_heads;
	hdr.hidden_dim = (uint32_t)cfg.hidden_dim;
	hdr.n_layers = (uint32_t)cfg.n_layers;
	hdr.vocab_size = (uint32_t)cfg.vocab_size;
	hdr.seq_len = (uint32_t)cfg.seq_len;
	hdr.checkpoint_bytes = checkpoint_bytes;
	memcpy(hdr.checkpoint_sha256, checkpoint_sha_before.bytes,
	    sizeof(hdr.checkpoint_sha256));
	hdr.layout_version = 1U;
	hdr.a_params = layout.a_params;
	hdr.b_params = layout.b_params;
	hdr.a_bytes = layout.a_bytes;
	hdr.b_bytes = layout.b_bytes;
	hdr.params_count = layout.total_params;
	hdr.data_bytes = layout.total_bytes;

	out = fopen(adapter_path, "wb");
	if (out == NULL) {
		fprintf(stderr, "error: opening adapter output '%s': %s\n",
		    adapter_path, strerror(errno));
		return 1;
	}

	if (fwrite(&hdr, sizeof(hdr), 1, out) != 1 ||
	    !write_deterministic_a_and_zero_b(out, &layout, 0.01f)) {
		fprintf(stderr, "error: writing adapter output '%s': %s\n",
		    adapter_path, strerror(errno));
		fclose(out);
		return 1;
	}

	if (fclose(out) != 0) {
		fprintf(stderr, "error: closing adapter output '%s': %s\n",
		    adapter_path, strerror(errno));
		return 1;
	}

	if (!validate_adapter_file(adapter_path, &cfg, rank,
	    &checkpoint_sha_before, checkpoint_bytes,
	    1, &hdr, &adapter_sha))
		return 1;

	if (!file_size_bytes(adapter_path, &adapter_file_bytes)) {
		fprintf(stderr, "error: cannot stat adapter output '%s'\n", adapter_path);
		return 1;
	}

	if (!hash_file_sha256(checkpoint_path, &checkpoint_sha_after)) {
		fprintf(stderr, "error: cannot hash checkpoint after write\n");
		return 1;
	}
	sha256_to_hex(&checkpoint_sha_after, checkpoint_hex_after);
	sha256_to_hex(&adapter_sha, adapter_hex);

	if (memcmp(&checkpoint_sha_before, &checkpoint_sha_after,
	    sizeof(checkpoint_sha_before)) != 0) {
		fprintf(stderr,
		    "error: checkpoint hash changed unexpectedly\n"
		    "  before: %s\n"
		    "  after:  %s\n",
		    checkpoint_hex_before, checkpoint_hex_after);
		return 1;
	}

	trainable_bytes = layout.total_bytes;
	gradient_bytes = layout.total_bytes;
	adam_first_moment_bytes = layout.total_bytes;
	adam_second_moment_bytes = layout.total_bytes;
	if (!checked_add_u64(trainable_bytes, gradient_bytes, &tmp_u64) ||
	    !checked_add_u64(tmp_u64, adam_first_moment_bytes, &tmp_u64) ||
	    !checked_add_u64(tmp_u64, adam_second_moment_bytes,
	    &total_trainable_storage_bytes)) {
		fprintf(stderr, "error: trainable storage accounting overflow\n");
		return 1;
	}

	printf("stage4 adapter init complete\n");
	printf("checkpoint: %s\n", checkpoint_path);
	printf("adapter: %s\n", adapter_path);
	printf("rank: %u\n", rank);
	printf("config: dim=%d hidden=%d layers=%d heads=%d kv_heads=%d\n",
	    cfg.dim, cfg.hidden_dim, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads);
	printf("trainable_params: %llu\n", (unsigned long long)layout.total_params);
	printf("adapter_bytes: %llu\n", (unsigned long long)trainable_bytes);
	printf("adapter_file_bytes: %llu\n", (unsigned long long)adapter_file_bytes);
	printf("memory_accounting adapter_weight_bytes=%llu\n",
	    (unsigned long long)trainable_bytes);
	printf("memory_accounting gradient_bytes=%llu\n",
	    (unsigned long long)gradient_bytes);
	printf("memory_accounting adam_first_moment_bytes=%llu\n",
	    (unsigned long long)adam_first_moment_bytes);
	printf("memory_accounting adam_second_moment_bytes=%llu\n",
	    (unsigned long long)adam_second_moment_bytes);
	printf("memory_accounting total_trainable_storage_bytes=%llu\n",
	    (unsigned long long)total_trainable_storage_bytes);
	printf("adapter file SHA-256: %s\n", adapter_hex);
	printf("base checkpoint SHA-256 after: %s\n", checkpoint_hex_after);
	printf("round-trip reload: PASS\n");
	printf("note: stories15M.bin was not modified\n");

	return 0;
}
