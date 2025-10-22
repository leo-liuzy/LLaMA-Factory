export CUDA_VISIBLE_DEVICES=3,5
# Calculate per device batch size
GLOBAL_BATCH_SIZE=32
PER_DEVICE_BATCH_SIZE=8
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
NUM_ACCUMULATION_STEPS=$((GLOBAL_BATCH_SIZE / NUM_GPUS / PER_DEVICE_BATCH_SIZE))

lr=1e-5

# llamafactory-cli train \
#   --model_name_or_path /u/zliu/datastor1/shared_resources/models/qwen/Qwen3-1.7B-Base \
#   --stage pt \
#   --do_train \
#   --finetuning_type full \
#   --dataset ctrl_re_ood_both \
#   --dataset_dir /u/zliu/datastor1/LLaMA-Factory/data \
#   --cutoff_len 128 \
#   --packing False \
#   --max_steps -1 \
#   --num_train_epochs 4 \
#   --output_dir /u/zliu/datastor1/LLaMA-Factory/saves/pt_on_ctrl_re_ood_both_lr${lr} \
#   --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
#   --gradient_accumulation_steps ${NUM_ACCUMULATION_STEPS} \
#   --learning_rate ${lr} \
#   --lr_scheduler_type constant \
#   --warmup_ratio 0.03 \
#   --weight_decay 0.1 \
#   --max_grad_norm 1.0 \
#   --bf16 \
#   --logging_steps 2 \
#   --overwrite_output_dir \


llamafactory-cli train \
  --model_name_or_path /u/zliu/datastor1/shared_resources/models/qwen/Qwen2.5-1.5B \
  --stage pt \
  --do_train \
  --finetuning_type full \
  --dataset ctrl_re_id \
  --dataset_dir /u/zliu/datastor1/LLaMA-Factory/data \
  --cutoff_len 128 \
  --packing False \
  --max_steps -1 \
  --num_train_epochs 4 \
  --output_dir /u/zliu/datastor1/LLaMA-Factory/saves/pt_on_ctrl_re_id_lr${lr} \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${NUM_ACCUMULATION_STEPS} \
  --learning_rate ${lr} \
  --lr_scheduler_type constant \
  --warmup_ratio 0.03 \
  --weight_decay 0.1 \
  --max_grad_norm 1.0 \
  --logging_steps 2 \
  --overwrite_output_dir \
  --bf16 \