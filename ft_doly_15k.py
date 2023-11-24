import argparse
import json
import os
import random
from functools import partial

import bitsandbytes as bnb
import pandas as pd
import peft
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments, set_seed)

from datasets import load_dataset
from sft_lib.dataset_utils_dolly_ds import *
from sft_lib.model_utils import *

# Reproducibility
SEED = 44
set_seed(SEED)

MODEL_NAME = "NousResearch/Llama-2-7b-chat-hf"
DS_NAME = "databricks/databricks-dolly-15k"
OUTPUT_DIR = f"{MODEL_NAME.split('/')[-1]}_{DS_NAME.split('/')[-1]}"


def load_and_show_dataset(dataset_path):
    dataset = load_dataset(dataset_path, split="train")
    print(f"\nNumber of original dataset: {len(dataset)}")
    print(f"Column names are: {dataset.column_names}")

    print("\n-----Seeing some samples from dataset:")
    nb_samples = 2
    samples = []
    random_indices = [0] + random.sample(range(len(dataset)), nb_samples)
    for idx in random_indices:
        samples.append(dataset[idx])
    for s in samples:
        print(json.dumps(s, indent=4))

    print("\n------prompt example converted from dataset:")
    print(prompterizer_for_dolly_ds(samples[0]))

    return dataset


def train_cli(
    warmup_steps=2,
    max_steps=15,  # 100,
    learning_rate=2e-4,
    accumulate_batch_size=4,
    optimizer="paged_adamw_8bit",
    output_root="results",
):
    bnb_config = create_4bit_bnb_config()
    model, tokenizer = load_model(MODEL_NAME, bnb_config)

    max_length = get_model_max_length(model)
    dataset = load_and_show_dataset(DS_NAME)
    dataset = preprocess_dataset(
        dataset=dataset,
        tokenizer=tokenizer,
        prompterize=prompterizer_for_dolly_ds,
        seed=SEED,
        max_length=max_length,
        do_shuffle=True,
    )
    print(f"preprocessed dataset length: {len(dataset)}")

    model = assemble_trainable_model(model)
    verify_datatypes(model)

    # Training parameters
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=TrainingArguments(
            # per_device_train_batch_size=1加gradient_accumulation_steps=4, 相当于每次更新参数的时候，batch_size=4
            per_device_train_batch_size=1,
            gradient_accumulation_steps=accumulate_batch_size,
            fp16=True,
            logging_steps=1,
            optim=optimizer,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            learning_rate=learning_rate,
            output_dir=os.path.join(output_root, OUTPUT_DIR),
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # Launch training
    print("\nTraining...")
    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    print(metrics)

    # Saving model
    print("Saving last checkpoint of the model...")
    checkpoint_dir = os.path.join(output_root, OUTPUT_DIR, "final_checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)
    trainer.model.save_pretrained(checkpoint_dir)

    # Free memory for merging weights
    del model
    del trainer
    torch.cuda.empty_cache()

    print("Fine-tuning finished!")


if __name__ == "__main__":
    import fire

    fire.Fire(train_cli)
