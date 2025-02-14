#!/bin/bash

PATH_TO_CFGS=./configs/full_data

BLACKLIST=(
accel
gbdt_dist
gbdt_dist_pred_3secs
gbdt_dist_pred_4secs
gbdt_dist_pred_5secs
gbdt_probe
gbdt_probe_pred_3secs
gbdt_probe_pred_4secs
gbdt_probe_pred_5secs
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

python -u src/tools/train_gbdt.py \
    --path-to-cfg=$cfg

done
