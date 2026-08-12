#!/bin/sh
set -eu

cd /usr/src/minix_ai_stage3_stories15m/minix_ai_stage4_train
make stage4eBaselineEval CC=clang
./stage4e_baseline_eval ../stories15M.bin -d stage4e-approved.records -z ../tokenizer.bin -o stage4e-baseline-unadapted-report.txt
cat stage4e-baseline-unadapted-report.txt
cksum stage4e-baseline-unadapted-report.txt
