#################### TRAINING  ####################
# (unchanged except we add --vision_mode if you want to train text‑only or video‑only)

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 \
    --master_port 12365 \
    src/open_r1/grpo.py \
    --output_dir "/temp/jz/res/7b_mmtom_rft_1" \
    --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
    --dataset_name "/home/jikezhong/Video-R1/mmtom_questions.json" \
    --vision_mode both \          # ← both | video_only | text_only
    --num_generations 8 \
    ...  # (all other args stay the same)



#################### EVALUATION ###################
# --- evaluate with BOTH modalities (default) ---
cd /home/jikezhong/Video-R1/src/r1-v

CUDA_VISIBLE_DEVICES=0 python src/open_r1/eval_grpo_multimodal.py \
    --model_name_or_path  "/temp/jz/res/7b_muma_sft" \
    --eval_dataset_path   "/home/jikezhong/Video-R1/mmtom_questions.json" \
    --vision_mode both \
    --num_generations 1 \
    --max_new_tokens 2048 & 

CUDA_VISIBLE_DEVICES=1 python src/open_r1/eval_grpo_multimodal.py \
    --model_name_or_path  "Qwen/Qwen2.5-VL-7B-Instruct" \
    --eval_dataset_path   "/home/jikezhong/Video-R1/mmtom_questions.json" \
    --vision_mode both \
    --num_generations 1 \
    --max_new_tokens 2048 &


CUDA_VISIBLE_DEVICES=2 python src/open_r1/eval_grpo_multimodal.py \
    --model_name_or_path  "/temp/jz/res/7b_mmtom_rft_1" \
    --eval_dataset_path   "/home/jikezhong/Video-R1/muma_questions.json" \
    --vision_mode both \
    --num_generations 1 \
    --max_new_tokens 2048 &


CUDA_VISIBLE_DEVICES=3 python src/open_r1/eval_grpo_multimodal.py \
    --model_name_or_path  "Qwen/Qwen2.5-VL-7B-Instruct" \
    --eval_dataset_path   "/home/jikezhong/Video-R1/muma_questions.json" \
    --vision_mode both \
    --num_generations 1 \
    --max_new_tokens 2048




####### video/text only on mmtom

CUDA_VISIBLE_DEVICES=0 python src/open_r1/eval_grpo_multimodal.py \
    --model_name_or_path  "/temp/jz/res/7b_muma_sft" \
    --eval_dataset_path   "/home/jikezhong/Video-R1/mmtom_questions.json" \
    --vision_mode text_only \
    --num_generations 1 \
    --max_new_tokens 2048 & 

CUDA_VISIBLE_DEVICES=1 python src/open_r1/eval_grpo_multimodal.py \
    --model_name_or_path  "Qwen/Qwen2.5-VL-7B-Instruct" \
    --eval_dataset_path   "/home/jikezhong/Video-R1/mmtom_questions.json" \
    --vision_mode text_only \
    --num_generations 1 \
    --max_new_tokens 2048 &

CUDA_VISIBLE_DEVICES=0 python src/open_r1/eval_grpo_multimodal.py \
    --model_name_or_path  "/temp/jz/res/7b_mmtom_sft" \
    --eval_dataset_path   "/home/jikezhong/Video-R1/mmtom_questions_vid_only.json" \
    --vision_mode video_only \
    --num_generations 1 \
    --max_new_tokens 2048 & 

CUDA_VISIBLE_DEVICES=1 python src/open_r1/eval_grpo_multimodal.py \
    --model_name_or_path  "Qwen/Qwen2.5-VL-7B-Instruct" \
    --eval_dataset_path   "/home/jikezhong/Video-R1/mmtom_questions_vid_only.json" \
    --vision_mode video_only \
    --num_generations 1 \
    --max_new_tokens 2048 


#######SFT

CUDA_VISIBLE_DEVICES=0 python src/open_r1/eval_grpo_multimodal.py \
    --model_name_or_path  "/temp/jz/res/7b_muma_sft" \
    --eval_dataset_path   "/home/jikezhong/Video-R1/mmtom_questions.json" \
    --vision_mode both \
    --num_generations 1 \
    --max_new_tokens 2048 &

CUDA_VISIBLE_DEVICES=1 python src/open_r1/eval_grpo_multimodal.py \
    --model_name_or_path  "/temp/jz/res/7b_muma_sft" \
    --eval_dataset_path   "/home/jikezhong/Video-R1/mmtom_questions.json" \
    --vision_mode text_only \
    --num_generations 1 \
    --max_new_tokens 2048 &

CUDA_VISIBLE_DEVICES=2 python src/open_r1/eval_grpo_multimodal.py \
    --model_name_or_path  "/temp/jz/res/7b_muma_sft" \
    --eval_dataset_path   "/home/jikezhong/Video-R1/mmtom_questions_vid_only.json" \
    --vision_mode video_only \
    --num_generations 1 \
    --max_new_tokens 2048 


# --- evaluate TEXT‑ONLY (ignore image/video) ---
CUDA_VISIBLE_DEVICES=0 python src/open_r1/eval_grpo_multimodal.py \
    --model_name_or_path  "/temp/jz/res/7b_mmtom_rft_1" \
    --eval_dataset_path   "/home/jikezhong/Video-R1/muma_questions.json" \
    --vision_mode text_only \
    --num_generations 1


# --- evaluate VIDEO‑ONLY (no question text) ---
CUDA_VISIBLE_DEVICES=0 python src/open_r1/eval_grpo_multimodal.py \
    --model_name_or_path  "/temp/jz/res/7b_mmtom_rft_1" \
    --eval_dataset_path   "/home/jikezhong/Video-R1/muma_questions.json" \
    --vision_mode video_only \
    --num_generations 1
