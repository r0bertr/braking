#!/bin/bash

PATH_TO_CFGS=./configs/shadow
PATH_TO_EXPS=../data/kyushu_driving_database/experiments/shadow_round_2

BLACKLIST=(
)

for cfg in $PATH_TO_CFGS/*.yaml; do

exp_name=$(basename $cfg .yaml)
blacklisted=0
for b in "${BLACKLIST[@]}"; do
    if [[ $exp_name == "$b" ]]; then
        blacklisted=1
    fi
done
if [ $blacklisted -eq 1 ]; then
    continue
fi

echo Running $exp_name

python -u src/exp_gbdt.py \
    --path-to-cfg=$cfg \
    --path-to-output=$PATH_TO_EXPS/$exp_name

done
