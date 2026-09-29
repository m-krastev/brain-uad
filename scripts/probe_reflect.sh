#!/usr/bin/env bash
# Every 30 min while REFLECT trains: copy its latest checkpoint and score it on
# the CPU with REFLECT's own evaluation (single most-pathological slice per
# test subject). Appends to results/reflect_probe/probe.log.
cd "$(dirname "$0")/../third_party/REFLECT"
R=REFLECT_BraTS_UNet_M_T2_256_kl_f8
P=../../results/reflect_probe
while pgrep -f train_REFLECT.py >/dev/null; do
    ck=$(ls -t $R/*/checkpoints/last.pt | head -1)
    cp "$ck" $P/ckpt/checkpoints/probe.pt
    steps=$(grep -oE "step=[0-9]+" $(dirname $(dirname "$ck"))/log.txt | tail -1)
    res=$(CUDA_VISIBLE_DEVICES="" nice -n 19 .venv/bin/torchrun --standalone --nproc_per_node=1 evaluate_REFLECT.py \
        --data-dir ../../data/reflect_brats21_t2 --model-path $P/ckpt/checkpoints/probe.pt --batch-size 8 --num-workers 2 2>/dev/null \
        | grep -E "^(max Dice|Global max Dice|AUROC)" | tr -s ' ' | tr '\n' ' ')
    echo "$(date '+%F %T') ckpt=$(stat -c %y "$ck" | cut -c12-19) latest_$steps $res" >> $P/probe.log
    sleep 1800
done
