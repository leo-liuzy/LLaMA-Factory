export CUDA_VISIBLE_DEVICES=1

export WANDB_MODE=online

gpu_count=$(awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
# bs=64
# per_device_train_batch_size=32
bs=128
per_device_train_batch_size=8
grad_acc=$((bs / gpu_count / per_device_train_batch_size))

weight_decay=0.1
max_grad_norm=1.0

seed=42
warmup_ratio=0.03
max_seq_length=256
epoch=2
# max_steps=1

lr=1e-5

output_dir=saves/Qwen3-1.7B-Base-eos-sft-lr${lr}
# model_name_or_path=${SCRATCH}/base_models/deepseek/hf/deepseek-coder-1.3b-base

# accelerate launch --config_file="fsdp_config.yaml" \
    # --main_process_port 29601 \
python scripts/lightweight_sft.py \
    --output_dir="${output_dir}" \
    --seed=${seed} \
    --learning_rate=${lr} \
    --lr_scheduler_type=linear \
    --weight_decay=${weight_decay} \
    --per_device_train_batch_size=${per_device_train_batch_size} \
    --per_device_eval_batch_size=1 \
    --gradient_accumulation_steps=${grad_acc} \
    --max_seq_length=${max_seq_length} \
    --max_grad_norm=${max_grad_norm} \
    --optim="adamw_torch" \
    --dataset_text_field="text" \
    --lr_scheduler_type="linear" \
    --warmup_ratio=${warmup_ratio} \
    --eval_strategy="no" \
    --save_strategy="no" \
    --save_total_limit=2 \
    --load_best_model_at_end=False \
    --logging_strategy="steps" \
    --logging_first_step=True \
    --logging_steps=2 \
    --eval_on_start=False \
    --report_to="wandb" \
    --num_train_epochs=${epoch} \
    --run_name="lightweight-eos-sft"