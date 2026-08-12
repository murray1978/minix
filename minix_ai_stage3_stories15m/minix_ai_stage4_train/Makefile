PROG=stage4_train_adapter
SRCS=stage4_train_adapter.c
MAN=

WARNS?=4
CFLAGS+=-O2
CFLAGS+=-std=c99
CFLAGS+=-D_POSIX_C_SOURCE=200112L
CFLAGS+=-D_LARGEFILE_SOURCE
LDADD+=-lm
DPADD+=${LIBM}

.PHONY: stage4Eval
stage4Eval: stage4_eval.c ../minix_llama.c
	${CC} ${CFLAGS} -o stage4_eval stage4_eval.c ${LDADD}

.PHONY: stage4GradientCheck
stage4GradientCheck: stage4_gradient_check.c stage4_adapter_train.c stage4_adapter_train.h
	${CC} ${CFLAGS} -o stage4_gradient_check stage4_gradient_check.c stage4_adapter_train.c ${LDADD}

.PHONY: stage4OneTokenTrain
stage4OneTokenTrain: stage4_one_token_train.c stage4_adapter_train.c stage4_adapter_train.h ../minix_llama.c
	${CC} ${CFLAGS} -o stage4_one_token_train stage4_one_token_train.c stage4_adapter_train.c ${LDADD}

.PHONY: stage4Train
stage4Train: stage4_train.c stage4_adapter_train.c stage4_adapter_train.h ../minix_llama.c
	${CC} ${CFLAGS} -o stage4_train stage4_train.c stage4_adapter_train.c ${LDADD}

.PHONY: stage4eDatasetCheck
stage4eDatasetCheck: stage4e_dataset_check.c stage4_adapter_train.c stage4_adapter_train.h ../minix_llama.c
	${CC} ${CFLAGS} -o stage4e_dataset_check stage4e_dataset_check.c stage4_adapter_train.c ${LDADD}

.PHONY: stage4eBaselineEval
stage4eBaselineEval: stage4e_baseline_eval.c stage4_adapter_train.c stage4_adapter_train.h ../minix_llama.c
	${CC} ${CFLAGS} -o stage4e_baseline_eval stage4e_baseline_eval.c stage4_adapter_train.c ${LDADD}

.PHONY: stage4eBaselineTop1Eval
stage4eBaselineTop1Eval: stage4e_baseline_top1_eval.c stage4_adapter_train.c stage4_adapter_train.h ../minix_llama.c
	${CC} ${CFLAGS} -o stage4e_baseline_top1_eval stage4e_baseline_top1_eval.c stage4_adapter_train.c ${LDADD}

.PHONY: stage4eBaselineGenerationEval
stage4eBaselineGenerationEval: stage4e_baseline_unadapted_generation.c stage4_adapter_train.c stage4_adapter_train.h ../minix_llama.c
	${CC} ${CFLAGS} -o stage4e_baseline_unadapted_generation stage4e_baseline_unadapted_generation.c stage4_adapter_train.c ${LDADD}

.PHONY: stage4eOneRecordTest
stage4eOneRecordTest: stage4eDatasetCheck
	sh ./test-stage4e-one-record.sh

.PHONY: stage4eCheckCandidate
stage4eCheckCandidate: stage4eDatasetCheck
	./stage4e_dataset_check -d stage4e-candidate.records -z ../tokenizer.bin -o stage4e-candidate-report.txt

.PHONY: stage4eCheckApproved
stage4eCheckApproved: stage4eDatasetCheck
	./stage4e_dataset_check -d stage4e-approved.records -z ../tokenizer.bin -o stage4e-dataset-report.txt

.PHONY: stage4eBaselineUnadapted
stage4eBaselineUnadapted: stage4eBaselineEval
	./stage4e_baseline_eval ../stories15M.bin -d stage4e-approved.records -z ../tokenizer.bin -o stage4e-baseline-unadapted-report.txt

.PHONY: stage4eBaselineTop1
stage4eBaselineTop1: stage4eBaselineTop1Eval
	./stage4e_baseline_top1_eval ../stories15M.bin -d stage4e-approved.records -z ../tokenizer.bin -o stage4e-baseline-unadapted-top1-report.txt

.PHONY: stage4eBaselineGeneration
stage4eBaselineGeneration: stage4eBaselineGenerationEval
	./stage4e_baseline_unadapted_generation ../stories15M.bin -d stage4e-approved.records -z ../tokenizer.bin -o stage4e-baseline-unadapted-generation-report.txt --max-visible-output-tokens 64

.PHONY: stage4C
stage4C: stage4GradientCheck stage4OneTokenTrain
	./stage4_gradient_check
	./stage4_one_token_train ../stories15M.bin -z ../tokenizer.bin

.PHONY: stage4D
stage4D: stage4Train
	./stage4_train ../stories15M.bin -z ../tokenizer.bin -a stories15M.adapter-overfit.bin -d overfit-one.records -r 8 --optimizer adam --learning-rate 0.001 --steps 500 --eval-every 10 --seed 1

.PHONY: zeroAdapterTest
zeroAdapterTest: stage4Eval ${PROG}
	sh ./test-zero-adapter-regression.sh

.include <bsd.prog.mk>
