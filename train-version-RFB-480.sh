#!/usr/bin/env bash
model_root_path="./models/train-version-RFB-480"
log_dir="$model_root_path/logs"
log="$log_dir/log"
mkdir -p "$log_dir"

python3 -u train.py \
  --datasets \
  ./data/MergedMaskFace_VOC \
  --validation_dataset \
  ./data/MergedMaskFace_VOC \
  --net \
  RFB \
  --num_epochs \
  200 \
  --milestones \
  "60,90,150" \
  --lr \
  1e-2 \
  --batch_size \
  24 \
  --input_size \
  480 \
  --checkpoint_folder \
  ${model_root_path} \
  --num_workers \
  8 \
  --log_dir \
  ${log_dir} \
  --cuda_index \
  0 \
  2>&1 | tee "$log"
