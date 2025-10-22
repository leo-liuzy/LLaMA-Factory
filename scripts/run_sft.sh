export CUDA_VISIBLE_DEVICES=0,1,2,3
# Calculate per device batch size
GLOBAL_BATCH_SIZE=128
PER_DEVICE_BATCH_SIZE=32
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
NUM_ACCUMULATION_STEPS=$((GLOBAL_BATCH_SIZE / NUM_GPUS / PER_DEVICE_BATCH_SIZE))



llamafactory-cli train \
  --model_name_or_path /u/zliu/datastor1/shared_resources/models/qwen/Qwen3-1.7B \
  --stage sft \
  --do_train \
  --finetuning_type full \
  --dataset trivia_wiki \
  --dataset_dir /u/zliu/datastor1/LLaMA-Factory/data \
  --template qwen3 \
  --cutoff_len 2048 \
  --max_steps -1 \
  --num_train_epochs 2 \
  --output_dir /u/zliu/datastor1/LLaMA-Factory/saves/sft_on_trivia_wiki \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${NUM_ACCUMULATION_STEPS} \
  --learning_rate 1e-5 \
  --lr_scheduler_type linear \
  --warmup_ratio 0.03 \
  --weight_decay 0.1 \
  --max_grad_norm 1.0 \
  --bf16 \
  --logging_steps 10 \
  --overwrite_output_dir \