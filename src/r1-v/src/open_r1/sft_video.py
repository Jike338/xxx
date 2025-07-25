# Copyright 2024. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Example usage:
accelerate launch \
    --config_file=deepspeed_zero2.yaml \
    train_video_llm.py \
    --dataset_name mfarre/simplevideoshorts \
    --model_name_or_path Qwen/Qwen2-VL-7B-Instruct \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --output_dir video-llm-output \
    --bf16 \
    --torch_dtype bfloat16 \
    --gradient_checkpointing
"""

import os
import json
import random
import requests
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLProcessor,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration
)
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
)
from accelerate import Accelerator
from qwen_vl_utils import process_vision_info

from datasets import Dataset, DatasetDict

import wandb

from typing import List, Dict, Any

def get_current_device():
    """Get the current device. For GPU we return the local process index to enable multiple GPU training."""
    return Accelerator().local_process_index if torch.cuda.is_available() else "cpu"

def download_video(url: str, folder: str = '/tmp/videos/') -> str:
    """Download video if not already present locally."""
    filename = url.split("/")[-1]
    local_path = os.path.join(folder, filename)

    if os.path.exists(local_path):
        return local_path

    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return local_path
    except requests.RequestException as e:
        raise Exception(f"Failed to download video: {e}")

def prepare_dataset(example: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Prepare dataset example for training."""

    

    system_message = "You are a helpful assistant"
    
    
    # QUESTION_TEMPLATE = (
    #     "{Question}\n"
    #     "Please think about this question as if you were a human pondering deeply. "
    #     "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
    #     "It's encouraged to include self-reflection or verification in the reasoning process. "
    #     "Provide your detailed reasoning between the <think> </think> tags, and then give your final answer between the <answer> </answer> tags."
    # )

    QUESTION_TEMPLATE = (
        "{Question}\n"
    )

    # TYPE_TEMPLATE = {
    #     "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
    #     "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
    #     "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
    #     "free-form": " Please provide your text answer within the <answer> </answer> tags.",
    #     "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
    #     "other": " Please provide only the single option letter (e.g., A, B, C, D, etc.)"
    # }

    TYPE_TEMPLATE = {
        "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.).",
        "other": " Please provide only the single option letter (e.g., A, B, C, D, etc.)."
    }

    
    if example["problem_type"] == 'multiple choice':
        question = example['problem'] + "Options:\n"
        for op in example["options"]:
            question += op + "\n"
    else:
        question = example['problem']


    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": example['data_type'],
                    # example['data_type']: os.getcwd() + "/Video-R1-data" + example['path'][1:]
                    example['data_type']: example['path']
                    # "max_pixels": 360*420,
                    # "fps": 1.0
                },
                {
                    "type": "text",
                    "text": QUESTION_TEMPLATE.format(Question=question) + TYPE_TEMPLATE[example['problem_type']]
                }
            ]
        },
        {
            "role": "assistant",
            # "content": [{"type": "text", "text": example['process'] + "\n" + example['solution']}]
            "content": [{"type": "text", "text": example['solution']}]
        }
    ]
    

    return {"messages": messages}

def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate batch of examples for training."""
    texts = []
    # video_inputs = []
    # image_inputs = []

    for i, example in enumerate(examples):
        try:

            texts.append(processor.apply_chat_template(example["messages"], tokenize=False))
            image_inputs, video_inputs, video_kwargs = process_vision_info(example["messages"], return_video_kwargs=True)
            
        except Exception as e:
            raise ValueError(f"Failed to process example {i}: {e}")

    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True
    )

    labels = inputs["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # Handle visual tokens based on processor type
    visual_tokens = [151652, 151653, 151656] if isinstance(processor, Qwen2VLProcessor) else [
        processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    ]

    for visual_token_id in visual_tokens:
        labels[labels == visual_token_id] = -100

    inputs["labels"] = labels
    return inputs

if __name__ == "__main__":
    # Parse arguments
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()
    
    # Configure training args
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    # Load dataset
    if script_args.dataset_name.endswith('.json') or script_args.dataset_name.endswith('.jsonl'):
        dataset =  DatasetDict({"train": Dataset.from_json(script_args.dataset_name)})
    else:
        # Load the dataset
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    # Setup model
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )

    # # Quantization configuration for 4-bit training
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16
    # )

    # Model initialization
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map(),
        # quantization_config=bnb_config,
    )
    
    
    if "Qwen2-VL" in model_config.model_name_or_path:
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    elif "Qwen2.5-VL" in model_config.model_name_or_path:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    else:
        model = AutoModelForVision2Seq.from_pretrained(model_config.model_name_or_path, **model_kwargs)

    processor = AutoProcessor.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code
    )

    # Prepare dataset
    prepared_dataset = [prepare_dataset(example) for example in dataset['train']]

    # Initialize wandb if specified
    if training_args.report_to == "wandb":
        wandb.init(project="video-llm-training")

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=prepared_dataset,
        data_collator=collate_fn,
        peft_config=get_peft_config(model_config),
        # tokenizer=processor.tokenizer
    )

    # Train model
    trainer.train()

    # Save final model

    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)

    if trainer.accelerator.is_main_process:
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    # Cleanup
    del model
    del trainer
    torch.cuda.empty_cache()
    wandb.finish()
