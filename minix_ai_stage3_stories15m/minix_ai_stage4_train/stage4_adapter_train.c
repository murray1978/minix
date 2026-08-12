#include "stage4_adapter_train.h"

#include <errno.h>
#include <ctype.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STAGE4_ADAPTER_MAGIC "MLAD"
#define STAGE4_ADAPTER_VERSION 2U
#define STAGE4_LAYOUT_VERSION 1U

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
} stage4_adapter_header_t;

typedef struct {
	uint32_t state[8];
	uint64_t total_bytes;
	uint8_t buffer[64];
	size_t used;
} sha256_ctx_t;

static int checked_mul_size(size_t a, size_t b, size_t *out)
{
	if (out == NULL || (a != 0 && b > SIZE_MAX / a))
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

static int checked_mul_u64(uint64_t a, uint64_t b, uint64_t *out)
{
	if (out == NULL || (a != 0 && b > UINT64_MAX / a))
		return 0;
	*out = a * b;
	return 1;
}

static void *checked_calloc(size_t count, size_t size)
{
	void *p;
	size_t bytes;

	if (!checked_mul_size(count, size, &bytes))
		return NULL;
	p = calloc(count, size);
	if (p == NULL && bytes != 0)
		return NULL;
	return p;
}

static uint64_t xorshift64(uint64_t *state)
{
	uint64_t x = *state;
	x ^= x >> 12;
	x ^= x << 25;
	x ^= x >> 27;
	*state = x;
	return x * UINT64_C(0x2545F4914F6CDD1D);
}

static float random_signed_unit(uint64_t *state)
{
	uint32_t r = (uint32_t)(xorshift64(state) >> 32);
	float unit = (r >> 8) / 16777216.0f;
	return 2.0f * unit - 1.0f;
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

static void sha256_compress(sha256_ctx_t *ctx, const uint8_t block[64])
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
		uint32_t s0 = rotr32(w[i - 15], 7) ^ rotr32(w[i - 15], 18) ^ (w[i - 15] >> 3);
		uint32_t s1 = rotr32(w[i - 2], 17) ^ rotr32(w[i - 2], 19) ^ (w[i - 2] >> 10);
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

static void sha256_init(sha256_ctx_t *ctx)
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

static void sha256_update(sha256_ctx_t *ctx, const void *data, size_t len)
{
	const uint8_t *p = (const uint8_t *)data;

	ctx->total_bytes += (uint64_t)len;
	while (len > 0) {
		size_t space = sizeof(ctx->buffer) - ctx->used;
		size_t n = len < space ? len : space;
		memcpy(ctx->buffer + ctx->used, p, n);
		ctx->used += n;
		p += n;
		len -= n;
		if (ctx->used == sizeof(ctx->buffer)) {
			sha256_compress(ctx, ctx->buffer);
			ctx->used = 0;
		}
	}
}

static void sha256_final(sha256_ctx_t *ctx, uint8_t out[32])
{
	uint8_t tail[128];
	uint64_t bits = ctx->total_bytes * 8U;
	size_t used = ctx->used;
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
		store_be32(out + i * 4, ctx->state[i]);
}

static int hex_to_bytes32(const char *hex, uint8_t out[32])
{
	static const char *digits = "0123456789abcdef";
	size_t i;

	if (hex == NULL || strlen(hex) != 64)
		return 0;
	for (i = 0; i < 32; i++) {
		const char *h = strchr(digits, (char)tolower((unsigned char)hex[i * 2]));
		const char *l = strchr(digits, (char)tolower((unsigned char)hex[i * 2 + 1]));
		if (h == NULL || l == NULL)
			return 0;
		out[i] = (uint8_t)(((h - digits) << 4) | (l - digits));
	}
	return 1;
}

static void bytes_to_hex32(const uint8_t in[32], char out[65])
{
	static const char *hex = "0123456789abcdef";
	int i;

	for (i = 0; i < 32; i++) {
		out[i * 2] = hex[(in[i] >> 4) & 0xF];
		out[i * 2 + 1] = hex[in[i] & 0xF];
	}
	out[64] = '\0';
}

int stage4_hash_file_sha256_hex(const char *path, char out_hex[65])
{
	FILE *f;
	unsigned char buffer[4096];
	size_t n;
	sha256_ctx_t ctx;
	uint8_t digest[32];

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
	bytes_to_hex32(digest, out_hex);
	return 1;
}

int stage4_file_size_bytes(const char *path, uint64_t *size_out)
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
	if (fclose(f) != 0)
		return 0;
	if (size < 0)
		return 0;
	*size_out = (uint64_t)size;
	return 1;
}

int stage4_adapter_init(stage4_adapter_t *ad, int dim, int vocab, int rank,
	float scale)
{
	size_t a_count;
	size_t b_count;

	if (ad == NULL || dim <= 0 || vocab <= 0 || rank <= 0)
		return 0;
	if (!checked_mul_size((size_t)rank, (size_t)dim, &a_count))
		return 0;
	if (!checked_mul_size((size_t)vocab, (size_t)rank, &b_count))
		return 0;

	memset(ad, 0, sizeof(*ad));
	ad->dim = dim;
	ad->vocab = vocab;
	ad->rank = rank;
	ad->scale = scale;
	ad->a_count = a_count;
	ad->b_count = b_count;

	ad->a = checked_calloc(a_count, sizeof(float));
	ad->b = checked_calloc(b_count, sizeof(float));
	ad->grad_a = checked_calloc(a_count, sizeof(float));
	ad->grad_b = checked_calloc(b_count, sizeof(float));
	ad->m1_a = checked_calloc(a_count, sizeof(float));
	ad->m1_b = checked_calloc(b_count, sizeof(float));
	ad->m2_a = checked_calloc(a_count, sizeof(float));
	ad->m2_b = checked_calloc(b_count, sizeof(float));
	ad->u = checked_calloc((size_t)rank, sizeof(float));
	ad->dlogits = checked_calloc((size_t)vocab, sizeof(float));

	if (ad->a == NULL || ad->b == NULL || ad->grad_a == NULL ||
	    ad->grad_b == NULL || ad->m1_a == NULL || ad->m1_b == NULL ||
	    ad->m2_a == NULL || ad->m2_b == NULL || ad->u == NULL ||
	    ad->dlogits == NULL) {
		stage4_adapter_free(ad);
		return 0;
	}

	return 1;
}

void stage4_adapter_free(stage4_adapter_t *ad)
{
	if (ad == NULL)
		return;
	free(ad->a);
	free(ad->b);
	free(ad->grad_a);
	free(ad->grad_b);
	free(ad->m1_a);
	free(ad->m1_b);
	free(ad->m2_a);
	free(ad->m2_b);
	free(ad->u);
	free(ad->dlogits);
	memset(ad, 0, sizeof(*ad));
}

void stage4_adapter_zero_grad(stage4_adapter_t *ad)
{
	if (ad == NULL)
		return;
	memset(ad->grad_a, 0, ad->a_count * sizeof(float));
	memset(ad->grad_b, 0, ad->b_count * sizeof(float));
}

int stage4_adapter_gradient_is_finite(const stage4_adapter_t *ad)
{
	size_t i;

	if (ad == NULL)
		return 0;
	for (i = 0; i < ad->a_count; i++) {
		if (!isfinite(ad->grad_a[i]))
			return 0;
	}
	for (i = 0; i < ad->b_count; i++) {
		if (!isfinite(ad->grad_b[i]))
			return 0;
	}
	return 1;
}

int stage4_adapter_parameters_are_finite(const stage4_adapter_t *ad)
{
	size_t i;

	if (ad == NULL)
		return 0;
	for (i = 0; i < ad->a_count; i++) {
		if (!isfinite(ad->a[i]) || !isfinite(ad->m1_a[i]) ||
		    !isfinite(ad->m2_a[i]))
			return 0;
	}
	for (i = 0; i < ad->b_count; i++) {
		if (!isfinite(ad->b[i]) || !isfinite(ad->m1_b[i]) ||
		    !isfinite(ad->m2_b[i]))
			return 0;
	}
	return 1;
}

double stage4_adapter_gradient_norm(const stage4_adapter_t *ad)
{
	double sum = 0.0;
	size_t i;

	for (i = 0; i < ad->a_count; i++)
		sum += (double)ad->grad_a[i] * (double)ad->grad_a[i];
	for (i = 0; i < ad->b_count; i++)
		sum += (double)ad->grad_b[i] * (double)ad->grad_b[i];
	return sqrt(sum);
}

void stage4_adapter_clip_gradients(stage4_adapter_t *ad, double max_norm)
{
	double norm;
	double scale;
	size_t i;

	if (ad == NULL || !(max_norm > 0.0))
		return;
	norm = stage4_adapter_gradient_norm(ad);
	if (!(norm > max_norm))
		return;
	scale = max_norm / norm;
	for (i = 0; i < ad->a_count; i++)
		ad->grad_a[i] = (float)((double)ad->grad_a[i] * scale);
	for (i = 0; i < ad->b_count; i++)
		ad->grad_b[i] = (float)((double)ad->grad_b[i] * scale);
}

void stage4_adapter_scale_gradients(stage4_adapter_t *ad, float scale)
{
	size_t i;

	if (ad == NULL)
		return;
	for (i = 0; i < ad->a_count; i++)
		ad->grad_a[i] *= scale;
	for (i = 0; i < ad->b_count; i++)
		ad->grad_b[i] *= scale;
}

void stage4_adapter_zero_moments(stage4_adapter_t *ad)
{
	if (ad == NULL)
		return;
	memset(ad->m1_a, 0, ad->a_count * sizeof(float));
	memset(ad->m1_b, 0, ad->b_count * sizeof(float));
	memset(ad->m2_a, 0, ad->a_count * sizeof(float));
	memset(ad->m2_b, 0, ad->b_count * sizeof(float));
}

void stage4_adapter_fill_a_deterministic(stage4_adapter_t *ad, uint64_t seed,
	float magnitude)
{
	size_t i;
	uint64_t state = seed == 0 ? UINT64_C(1) : seed;

	if (ad == NULL)
		return;
	if (!(magnitude > 0.0f))
		magnitude = 0.01f;
	for (i = 0; i < ad->a_count; i++)
		ad->a[i] = magnitude * random_signed_unit(&state);
}

void stage4_adapter_zero_b(stage4_adapter_t *ad)
{
	if (ad == NULL)
		return;
	memset(ad->b, 0, ad->b_count * sizeof(float));
}

void stage4_adapter_forward_logits(const stage4_adapter_t *ad,
	const float *hidden,
	const float *base_logits,
	float *final_logits)
{
	int r;
	int v;

	for (v = 0; v < ad->vocab; v++)
		final_logits[v] = base_logits[v];

	for (r = 0; r < ad->rank; r++) {
		double sum = 0.0;
		int j;
		for (j = 0; j < ad->dim; j++)
			sum += (double)ad->a[(size_t)r * (size_t)ad->dim + (size_t)j] *
			    (double)hidden[j];
		ad->u[r] = (float)sum;
	}

	for (v = 0; v < ad->vocab; v++) {
		double corr = 0.0;
		int r2;
		for (r2 = 0; r2 < ad->rank; r2++)
			corr += (double)ad->b[(size_t)v * (size_t)ad->rank + (size_t)r2] *
			    (double)ad->u[r2];
		final_logits[v] += ad->scale * (float)corr;
	}
}

int stage4_softmax_loss_and_dlogits(const float *final_logits,
	int vocab,
	int target,
	double *loss_out,
	float *dlogits_out)
{
	double max_logit;
	double sum = 0.0;
	double logsum;
	int i;

	if (target < 0 || target >= vocab)
		return 0;
	if (!isfinite(final_logits[0]))
		return 0;

	max_logit = final_logits[0];
	for (i = 1; i < vocab; i++) {
		if (!isfinite(final_logits[i]))
			return 0;
		if (final_logits[i] > max_logit)
			max_logit = final_logits[i];
	}

	for (i = 0; i < vocab; i++) {
		double p = exp((double)final_logits[i] - max_logit);
		if (!isfinite(p))
			return 0;
		sum += p;
	}
	if (!(sum > 0.0) || !isfinite(sum))
		return 0;

	logsum = max_logit + log(sum);
	*loss_out = logsum - (double)final_logits[target];
	if (!isfinite(*loss_out))
		return 0;

	for (i = 0; i < vocab; i++) {
		double p = exp((double)final_logits[i] - logsum);
		if (!isfinite(p))
			return 0;
		dlogits_out[i] = (float)p;
	}
	dlogits_out[target] -= 1.0f;

	for (i = 0; i < vocab; i++) {
		if (!isfinite(dlogits_out[i]))
			return 0;
	}

	return 1;
}

void stage4_adapter_backward(stage4_adapter_t *ad,
	const float *hidden,
	const float *dlogits)
{
	int v;
	int r;

	for (v = 0; v < ad->vocab; v++) {
		for (r = 0; r < ad->rank; r++) {
			ad->grad_b[(size_t)v * (size_t)ad->rank + (size_t)r] +=
			    ad->scale * dlogits[v] * ad->u[r];
		}
	}

	for (r = 0; r < ad->rank; r++) {
		double du = 0.0;
		int j;
		for (v = 0; v < ad->vocab; v++) {
			du += (double)ad->scale *
			    (double)ad->b[(size_t)v * (size_t)ad->rank + (size_t)r] *
			    (double)dlogits[v];
		}
		for (j = 0; j < ad->dim; j++) {
			ad->grad_a[(size_t)r * (size_t)ad->dim + (size_t)j] +=
			    (float)(du * (double)hidden[j]);
		}
	}
}

double stage4_adapter_sgd_step(stage4_adapter_t *ad, float learning_rate)
{
	double update_sq = 0.0;
	size_t i;

	for (i = 0; i < ad->a_count; i++) {
		double delta = (double)learning_rate * (double)ad->grad_a[i];
		ad->a[i] -= (float)delta;
		update_sq += delta * delta;
	}
	for (i = 0; i < ad->b_count; i++) {
		double delta = (double)learning_rate * (double)ad->grad_b[i];
		ad->b[i] -= (float)delta;
		update_sq += delta * delta;
	}
	return sqrt(update_sq);
}

double stage4_adapter_adam_step(stage4_adapter_t *ad,
	float learning_rate,
	float beta1,
	float beta2,
	float epsilon,
	float weight_decay,
	uint64_t step_index)
{
	double update_sq = 0.0;
	double b1_pow;
	double b2_pow;
	double one_minus_b1_pow;
	double one_minus_b2_pow;
	size_t i;

	if (ad == NULL || step_index == 0)
		return 0.0;

	b1_pow = pow((double)beta1, (double)step_index);
	b2_pow = pow((double)beta2, (double)step_index);
	one_minus_b1_pow = 1.0 - b1_pow;
	one_minus_b2_pow = 1.0 - b2_pow;

	if (!(one_minus_b1_pow > 0.0) || !(one_minus_b2_pow > 0.0))
		return 0.0;

	for (i = 0; i < ad->a_count; i++) {
		double g = (double)ad->grad_a[i] +
		    (double)weight_decay * (double)ad->a[i];
		double m = (double)beta1 * (double)ad->m1_a[i] +
		    (1.0 - (double)beta1) * g;
		double v = (double)beta2 * (double)ad->m2_a[i] +
		    (1.0 - (double)beta2) * g * g;
		double m_hat = m / one_minus_b1_pow;
		double v_hat = v / one_minus_b2_pow;
		double delta = (double)learning_rate * m_hat /
		    (sqrt(v_hat) + (double)epsilon);

		ad->m1_a[i] = (float)m;
		ad->m2_a[i] = (float)v;
		ad->a[i] -= (float)delta;
		update_sq += delta * delta;
	}

	for (i = 0; i < ad->b_count; i++) {
		double g = (double)ad->grad_b[i] +
		    (double)weight_decay * (double)ad->b[i];
		double m = (double)beta1 * (double)ad->m1_b[i] +
		    (1.0 - (double)beta1) * g;
		double v = (double)beta2 * (double)ad->m2_b[i] +
		    (1.0 - (double)beta2) * g * g;
		double m_hat = m / one_minus_b1_pow;
		double v_hat = v / one_minus_b2_pow;
		double delta = (double)learning_rate * m_hat /
		    (sqrt(v_hat) + (double)epsilon);

		ad->m1_b[i] = (float)m;
		ad->m2_b[i] = (float)v;
		ad->b[i] -= (float)delta;
		update_sq += delta * delta;
	}

	return sqrt(update_sq);
}

double stage4_adapter_max_logit_correction(const float *base_logits,
	const float *final_logits,
	int vocab)
{
	double max_abs = 0.0;
	int i;

	for (i = 0; i < vocab; i++) {
		double d = fabs((double)final_logits[i] - (double)base_logits[i]);
		if (d > max_abs)
			max_abs = d;
	}
	return max_abs;
}

int stage4_adapter_storage_accounting(const stage4_adapter_t *ad,
	stage4_storage_accounting_t *out)
{
	uint64_t param_bytes;
	uint64_t total;

	if (ad == NULL || out == NULL)
		return 0;
	if (!checked_mul_u64((uint64_t)(ad->a_count + ad->b_count),
	    (uint64_t)sizeof(float), &param_bytes))
		return 0;
	if (!checked_add_u64(param_bytes, param_bytes, &total))
		return 0;
	if (!checked_add_u64(total, param_bytes, &total))
		return 0;
	if (!checked_add_u64(total, param_bytes, &total))
		return 0;

	out->adapter_weight_bytes = param_bytes;
	out->gradient_bytes = param_bytes;
	out->adam_first_moment_bytes = param_bytes;
	out->adam_second_moment_bytes = param_bytes;
	out->total_trainable_storage_bytes = total;
	return 1;
}

int stage4_adapter_save_file(const char *path,
	const stage4_adapter_t *ad,
	const stage4_base_identity_t *base_id)
{
	stage4_adapter_header_t hdr;
	FILE *f;
	uint8_t base_sha[32];

	if (ad == NULL || base_id == NULL)
		return 0;
	if (!hex_to_bytes32(base_id->base_sha256, base_sha))
		return 0;

	memset(&hdr, 0, sizeof(hdr));
	memcpy(hdr.magic, STAGE4_ADAPTER_MAGIC, 4);
	hdr.version = STAGE4_ADAPTER_VERSION;
	hdr.rank = (uint32_t)ad->rank;
	hdr.dim = (uint32_t)ad->dim;
	hdr.vocab_size = (uint32_t)ad->vocab;
	hdr.n_heads = 0;
	hdr.n_kv_heads = 0;
	hdr.hidden_dim = 0;
	hdr.n_layers = 0;
	hdr.seq_len = 0;
	hdr.checkpoint_bytes = base_id->base_checkpoint_bytes;
	memcpy(hdr.checkpoint_sha256, base_sha, 32);
	hdr.layout_version = STAGE4_LAYOUT_VERSION;
	hdr.a_params = (uint64_t)ad->a_count;
	hdr.b_params = (uint64_t)ad->b_count;
	hdr.a_bytes = (uint64_t)ad->a_count * sizeof(float);
	hdr.b_bytes = (uint64_t)ad->b_count * sizeof(float);
	hdr.params_count = (uint64_t)(ad->a_count + ad->b_count);
	hdr.data_bytes = hdr.a_bytes + hdr.b_bytes;

	f = fopen(path, "wb");
	if (f == NULL)
		return 0;

	if (fwrite(&hdr, sizeof(hdr), 1, f) != 1 ||
	    fwrite(ad->a, sizeof(float), ad->a_count, f) != ad->a_count ||
	    fwrite(ad->b, sizeof(float), ad->b_count, f) != ad->b_count) {
		fclose(f);
		return 0;
	}

	if (fclose(f) != 0)
		return 0;
	return 1;
}

int stage4_adapter_load_file(const char *path,
	stage4_adapter_t *ad,
	const stage4_base_identity_t *base_id,
	int require_b_zero)
{
	FILE *f;
	stage4_adapter_header_t hdr;
	uint8_t expected_sha[32];
	uint64_t file_bytes;
	uint64_t expected_file;
	size_t i;

	if (ad == NULL || base_id == NULL)
		return 0;
	if (!hex_to_bytes32(base_id->base_sha256, expected_sha))
		return 0;
	if (!stage4_file_size_bytes(path, &file_bytes))
		return 0;

	f = fopen(path, "rb");
	if (f == NULL)
		return 0;
	if (fread(&hdr, sizeof(hdr), 1, f) != 1) {
		fclose(f);
		return 0;
	}

	if (memcmp(hdr.magic, STAGE4_ADAPTER_MAGIC, 4) != 0 ||
	    hdr.version != STAGE4_ADAPTER_VERSION ||
	    hdr.layout_version != STAGE4_LAYOUT_VERSION) {
		fclose(f);
		return 0;
	}

	if (hdr.checkpoint_bytes != base_id->base_checkpoint_bytes ||
	    memcmp(hdr.checkpoint_sha256, expected_sha, 32) != 0) {
		fclose(f);
		return 0;
	}

	if (hdr.rank != (uint32_t)ad->rank ||
	    hdr.dim != (uint32_t)ad->dim ||
	    hdr.vocab_size != (uint32_t)ad->vocab ||
	    hdr.a_params != (uint64_t)ad->a_count ||
	    hdr.b_params != (uint64_t)ad->b_count ||
	    hdr.data_bytes != (uint64_t)(ad->a_count + ad->b_count) * sizeof(float)) {
		fclose(f);
		return 0;
	}

	if (!checked_add_u64((uint64_t)sizeof(hdr), hdr.data_bytes, &expected_file) ||
	    expected_file != file_bytes) {
		fclose(f);
		return 0;
	}

	if (fread(ad->a, sizeof(float), ad->a_count, f) != ad->a_count ||
	    fread(ad->b, sizeof(float), ad->b_count, f) != ad->b_count) {
		fclose(f);
		return 0;
	}

	if (fgetc(f) != EOF) {
		fclose(f);
		return 0;
	}

	for (i = 0; i < ad->a_count; i++) {
		if (!isfinite(ad->a[i])) {
			fclose(f);
			return 0;
		}
	}
	for (i = 0; i < ad->b_count; i++) {
		if (!isfinite(ad->b[i])) {
			fclose(f);
			return 0;
		}
		if (require_b_zero && ad->b[i] != 0.0f) {
			fclose(f);
			return 0;
		}
	}

	if (fclose(f) != 0)
		return 0;
	return 1;
}
