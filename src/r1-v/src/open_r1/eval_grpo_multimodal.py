#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate a GRPO‑fine‑tuned Qwen‑VL checkpoint on a JSON dataset that
contains both vision (image **or** video) and text.

Options:
    --vision_mode both        # (default) keep vision + text
    --vision_mode video_only  # keep vision, drop question text
    --vision_mode text_only   # keep question text, drop vision
"""

import argparse, copy, json, os
from pathlib import Path
from typing import List, Dict, Any

import torch, numpy as np
from tqdm import tqdm
from datasets import Dataset

from transformers import (
    AutoProcessor,
    GenerationConfig,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template

# ------------- 1.  Reward functions & templates ------------------------
#  (import from your training module; assumes it's on PYTHONPATH)
from grpo import (
    accuracy_reward,
)
from qwen_vl_utils import process_vision_info
import copy
# ----------------------------------------------------------------------

# QUESTION_TEMPLATE = (
#     "Please watch the video and read the description (if any) carefully and answer the question below.\n"
#     "{Question}\n"
#     "Provide your detailed reasoning between the <think> </think> tags, and then give your final answer between the <answer> </answer> tags."
# )
# TYPE_TEMPLATE = {
#     "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
#     "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
#     "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
#     "free-form": " Please provide your text answer within the <answer> </answer> tags.",
#     "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
#     "other": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags."
# }

QUESTION_TEMPLATE = (
    "{Question}\n"
)


TYPE_TEMPLATE = {
    "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.).",
    "other": " Please provide only the single option letter (e.g., A, B, C, D, etc.)."
}
# ------------ 2.  Conversation‑builder with vision_mode ---------------
def build_conversation(example: Dict[str, Any], vision_mode: str) -> Dict[str, Any]:
    """
    Returns the 'prompt' structure expected by Qwen‑VL processors.
    """
    # ---- build question text ----
    if example["problem_type"] == "multiple choice":
        q_text = example["problem"] + "Options:\n" + "\n".join(example["options"])
    else:
        q_text = example["problem"]

    text_msg = {
        "type": "text",
        "text": QUESTION_TEMPLATE.format(Question=q_text)
        + TYPE_TEMPLATE[example["problem_type"]],
    }

    content: List[Dict[str, Any]] = []

    if vision_mode != "text_only":
        content.append({"type": example["data_type"]})  # placeholder for path injection
    # if vision_mode != "video_only":
    content.append(text_msg)

    return {"prompt": [{"role": "user", "content": content}]}


# ----------------------------------------------------------------------


# ------------ 3.  Model / processor loader ----------------------------
def load_model_and_processor(model_name_or_path: str):
    # if "Qwen2.5-VL" in model_name_or_path:
    model_cls = Qwen2_5_VLForConditionalGeneration


    model = model_cls.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto",
    ).eval()

    processor = AutoProcessor.from_pretrained(model_name_or_path)
    processor.pad_token_id = processor.tokenizer.pad_token_id
    processor.eos_token_id = processor.tokenizer.eos_token_id
    return model, processor


# ----------------------------------------------------------------------


# ------------ 4.  Evaluation loop -------------------------------------
def eval_dataset(
    model,
    processor,
    dataset: Dataset,
    vision_mode: str,
    num_generations: int = 1,
    max_new_tokens: int = 768,
) -> Dict[str, Any]:
    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_return_sequences=num_generations,
        pad_token_id=processor.pad_token_id,
    )

    rewards: List[float] = []
    augmented_questions: List[Dict[str, Any]] = []

    for ex in tqdm(dataset, desc=f"Eval ({vision_mode})"):
        conv = build_conversation(ex, vision_mode)
        
        prompts = [conv["prompt"]]
        prompts_text = [maybe_apply_chat_template(conv, processor)["prompt"]]
        # ---- inject real path into the visual token if kept ----
        if vision_mode != "text_only":
            if ex["data_type"] == "image":
                conv["prompt"][0]["content"][0]["image"] = ex["path"]
            else:
                conv["prompt"][0]["content"][0]["video"] = ex["path"]

        # ---- vision preprocessing (skip when text_only) ----
        if vision_mode == "text_only":
            image_inputs = video_inputs = None
        else:
            image_inputs, video_inputs, _ = process_vision_info(
                conv["prompt"], 
                return_video_kwargs=True
            )

        # # ---- text prompt: add_generation_prompt=True mimics training ----
        # prompt_str = processor.apply_chat_template(
        #     conv, tokenize=False, add_generation_prompt=True
        # )

        model_inputs = processor(
            text=copy.deepcopy(prompts_text),
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        ).to(model.device)

        outputs = model.generate(**model_inputs, generation_config=gen_cfg)
        comp_ids = outputs[:, model_inputs["input_ids"].size(1) :]
        comp_text = processor.batch_decode(comp_ids, skip_special_tokens=True)
        print(prompts_text)
        print(comp_text)
        comp_struct = [[{"role": "assistant", "content": c}] for c in comp_text]
        reward_vals = accuracy_reward(
            completions=comp_struct,
            solution=[ex["solution"]] * len(comp_struct),
            problem_type=[ex["problem_type"]] * len(comp_struct),
        )
        print(reward_vals)
                # ── augment and store ────────────────────────────────────────────
        ex_aug = dict(ex)
        ex_aug["model_response"] = comp_text[0]
        ex_aug["reward"] = reward_vals[0]
        augmented_questions.append(ex_aug)

        rewards.extend(reward_vals)
    arr = np.asarray(rewards, dtype=float)
    metrics = {"n": int(arr.size), "mean": float(arr.mean()), "std": float(arr.std(ddof=0))}
    return metrics, augmented_questions


# ----------------------------------------------------------------------


# ------------ 5.  CLI + main ------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--eval_dataset_path", type=str, required=True)
    p.add_argument(
        "--vision_mode",
        type=str,
        default="both",
        choices=["both", "video_only", "text_only"],
        help="Disable/enable vision or text input.",
    )
    p.add_argument("--num_generations", type=int, default=1)
    p.add_argument("--max_new_tokens", type=int, default=768)
    return p.parse_args()


def main():
    args = parse_args()

    model, processor = load_model_and_processor(args.model_name_or_path)

    raw = Dataset.from_json(args.eval_dataset_path)
    # Keep only needed keys; no heavy mapping required
    eval_ds = raw  # mapping is handled on-the-fly inside loop

    metrics, questions = eval_dataset(
        model,
        processor,
        eval_ds,
        vision_mode=args.vision_mode,
        num_generations=args.num_generations,
        max_new_tokens=args.max_new_tokens,
    )

    print("\n========== FINAL EVAL ==========")
    print(f"Vision mode          : {args.vision_mode}")
    print(f"Examples evaluated   : {metrics['n']}")
    print(f"Mean reward/accuracy : {metrics['mean']:.4f}")
    print(f"Std‑dev reward       : {metrics['std']:.4f}")

    # Construct filename from model, dataset, and vision_mode
    model_id = Path(args.model_name_or_path).name
    dataset_id = Path(args.eval_dataset_path).stem
    vision_id = args.vision_mode
    filename = f"{model_id}_{dataset_id}_{vision_id}.json"
    result_path = f"/home/jikezhong/Video-R1/results/{filename}"

    # Ensure results directory exists and save
    os.makedirs(os.path.dirname(result_path), exist_ok=True)

    final_json = {"results": metrics, "questions": questions}
    with open(result_path, "w") as f:
        json.dump(final_json, f, indent=2)
        print(f"\nResults written to {result_path}")


if __name__ == "__main__":
    main()
