cd /home/jikezhong/Video-R1/src/r1-v

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./home/jikezhong/Video-R1/src/scripts/7b_muma_sft"


CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12349" \
    src/open_r1/sft_video.py \
    --output_dir "/temp/jz/res/7b_muma_sft" \
    --model_name_or_path "Qwen/Qwen2.5-VL-7B-Instruct" \
    --dataset_name "/home/jikezhong/Video-R1/muma_questions.json" \
    --deepspeed local_scripts/zero2.json \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-6 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name 7b_muma_sft \
    --save_steps 1000 \
    --max_grad_norm 5 \
    --save_only_model true \
    --save_total_limit 1