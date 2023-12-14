import argparse
import json
import os
import random
from functools import partial

import bitsandbytes as bnb
import pandas as pd
import peft
import torch
from accelerate import Accelerator
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments, set_seed)

from datasets import load_dataset
from sft_lib.dataset_utils import *
from sft_lib.model_utils import *

# Reproducibility
SEED = 44
set_seed(SEED)

MODEL_NAME = "NousResearch/Llama-2-7b-chat-hf"


def load_and_show_dataset(ds_name, suffix):
    dataset = get_dataset_from_text_files(ds_name, suffix=suffix)
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
    print(text2prompt(samples[0]["text"]))

    return dataset


def train_cli(
    warmup_steps=4,
    max_steps=30,  # 100,
    learning_rate=2e-4,
    output_root="results",
    ds_name="simple_markdown_with_answer",
):
    output_dir = f"{MODEL_NAME.split('/')[-1]}_{ds_name.split('/')[-1]}"
    # dataset = get_dataset_from_text_files(ds_name, suffix="md")
    dataset = load_and_show_dataset(ds_name, suffix="md")
    bnb_config = create_4bit_bnb_config()
    model, tokenizer = load_model(MODEL_NAME, bnb_config)

    max_length = get_model_max_length(model)
    dataset = generate_dataloader(
        dataset=dataset,
        tokenizer=tokenizer,
        prompterize=text2prompt,
        # for training, max_length can be the max model token length, because the answer is also in the text.
        seed=SEED,
        max_length=max_length,
        batch_size=1,
        do_shuffle=True,
        abandon_long_sent=True,
        with_labels=True,
    )
    print(f"preprocessed dataset length: {len(dataset)}")

    model = assemble_trainable_model(model)
    verify_datatypes(model)

    # if training_args.gradient_checkpointing:
    #     model.gradient_checkpointing_enable()

    accelerator = Accelerator(fp16=True)
    model, optimizer, dataloader = accelerator.prepare(model, adam_bnb_optim, dataloader)

    model.train()
    # for epoch in range(1, 5):
    for i, sample in enumerate(ds, start=1):
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=TrainingArguments(
            # per_device_train_batch_size=1加gradient_accumulation_steps=4, 相当于每次更新参数的时候，batch_size=4
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            fp16=True,
            logging_steps=1,
            optim="paged_adamw_8bit",
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            learning_rate=learning_rate,
            output_dir=os.path.join(output_root, output_dir),
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
    checkpoint_dir = os.path.join(output_root, output_dir, "final_checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)
    trainer.model.save_pretrained(checkpoint_dir)

    # Free memory for merging weights
    del model
    del trainer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    import fire

    fire.Fire(train_cli)
