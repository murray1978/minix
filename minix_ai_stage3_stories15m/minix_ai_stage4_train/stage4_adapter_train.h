#ifndef STAGE4_ADAPTER_TRAIN_H
#define STAGE4_ADAPTER_TRAIN_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
	uint64_t adapter_weight_bytes;
	uint64_t gradient_bytes;
	uint64_t adam_first_moment_bytes;
	uint64_t adam_second_moment_bytes;
	uint64_t total_trainable_storage_bytes;
} stage4_storage_accounting_t;

typedef struct {
	int dim;
	int vocab;
	int rank;
	float scale;
	size_t a_count;
	size_t b_count;
	float *a;
	float *b;
	float *grad_a;
	float *grad_b;
	float *m1_a;
	float *m1_b;
	float *m2_a;
	float *m2_b;
	float *u;
	float *dlogits;
} stage4_adapter_t;

int stage4_adapter_init(stage4_adapter_t *ad, int dim, int vocab, int rank,
	float scale);
void stage4_adapter_free(stage4_adapter_t *ad);
void stage4_adapter_zero_grad(stage4_adapter_t *ad);
int stage4_adapter_gradient_is_finite(const stage4_adapter_t *ad);
int stage4_adapter_parameters_are_finite(const stage4_adapter_t *ad);
double stage4_adapter_gradient_norm(const stage4_adapter_t *ad);
void stage4_adapter_clip_gradients(stage4_adapter_t *ad, double max_norm);
void stage4_adapter_scale_gradients(stage4_adapter_t *ad, float scale);
void stage4_adapter_zero_moments(stage4_adapter_t *ad);

void stage4_adapter_fill_a_deterministic(stage4_adapter_t *ad, uint64_t seed,
	float magnitude);
void stage4_adapter_zero_b(stage4_adapter_t *ad);

void stage4_adapter_forward_logits(const stage4_adapter_t *ad,
	const float *hidden,
	const float *base_logits,
	float *final_logits);

int stage4_softmax_loss_and_dlogits(const float *final_logits,
	int vocab,
	int target,
	double *loss_out,
	float *dlogits_out);

void stage4_adapter_backward(stage4_adapter_t *ad,
	const float *hidden,
	const float *dlogits);

double stage4_adapter_sgd_step(stage4_adapter_t *ad, float learning_rate);
double stage4_adapter_adam_step(stage4_adapter_t *ad,
	float learning_rate,
	float beta1,
	float beta2,
	float epsilon,
	float weight_decay,
	uint64_t step_index);

double stage4_adapter_max_logit_correction(const float *base_logits,
	const float *final_logits,
	int vocab);

int stage4_adapter_storage_accounting(const stage4_adapter_t *ad,
	stage4_storage_accounting_t *out);

typedef struct {
	char base_sha256[65];
	uint64_t base_checkpoint_bytes;
} stage4_base_identity_t;

int stage4_hash_file_sha256_hex(const char *path, char out_hex[65]);
int stage4_file_size_bytes(const char *path, uint64_t *size_out);

int stage4_adapter_save_file(const char *path,
	const stage4_adapter_t *ad,
	const stage4_base_identity_t *base_id);

int stage4_adapter_load_file(const char *path,
	stage4_adapter_t *ad,
	const stage4_base_identity_t *base_id,
	int require_b_zero);

#endif
