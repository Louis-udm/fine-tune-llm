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
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

from datasets import load_dataset
from sft_lib.dataset_utils import *
from sft_lib.model_utils import *
from bitsandbytes.optim import AdamW as bnb_AdamW

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
    max_steps=80,  # 100,
    learning_rate=2e-4,
    output_root="results",
    ds_name="simple_markdown_with_answer",
):
    output_dir = f"{MODEL_NAME.split('/')[-1]}_{ds_name.split('/')[-1]}-ft_accelerate"
    # dataset = get_dataset_from_text_files(ds_name, suffix="md")
    dataset = load_and_show_dataset(ds_name, suffix="md")
    bnb_config = create_4bit_bnb_config()
    model, tokenizer = load_model(MODEL_NAME, bnb_config)

    max_length = get_model_max_length(model)
    train_dataloader = generate_dataloader(
        dataset=dataset,
        tokenizer=tokenizer,
        prompterize=text2prompt,
        # for training, max_length can be the max model token length, because the answer is also in the text.
        seed=SEED,
        max_length=max_length,
        batch_size=1,
        do_shuffle=True,
        abandon_long_sample=True,
        with_labels=True,
    )
    print(f"preprocessed dataset length: {len(train_dataloader)}")

    model = assemble_trainable_model(model)
    verify_datatypes(model)

    # if training_args.gradient_checkpointing:
    #     model.gradient_checkpointing_enable()

    accelerator = Accelerator(
        mixed_precision="fp16", cpu=False, gradient_accumulation_steps=4
    )
    accelerator.free_memory()
    # optimizer_kwargs={'lr': 0.0002, 'betas': (0.9, 0.999), 'eps': 1e-08, 'is_paged': True, 'optim_bits': 8}
    # optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate)
    optimizer = bnb_AdamW(
        model.parameters(), lr=learning_rate, optim_bits=8, is_paged=True
    )
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    # Launch training
    print("\nTraining...")
    model.train()
    step = 0
    epoch = 0
    while step <= max_steps:
        for i, batch in enumerate(train_dataloader, start=1):
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    loss = model(**batch).loss

                # backward()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    print("clip")
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                # if lr_scheduler is not None:
                #     lr_scheduler.step()
                optimizer.zero_grad()

                all_loss = accelerator.gather(loss).sum()
                step += 1
                # print(f"step: {step}-{epoch} train loss : {loss.item()}")
                print(f"step: {step}-{epoch} train loss : {all_loss.item()}")
                if step > max_steps:
                    break
        epoch += 1

    # Saving model
    print("Saving last checkpoint of the model...")
    checkpoint_dir = os.path.join(output_root, output_dir, "final_checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.save_pretrained(checkpoint_dir)

    # Free memory for merging weights
    del model
    # del trainer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    import fire

    fire.Fire(train_cli)
