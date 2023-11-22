from datasets import load_dataset
from typing import Callable
from functools import partial
import os
import peft
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def create_prompt_formats(sample):
    """
    special for Databricks Dolly 15k dataset: https://huggingface.co/datasets/databricks/databricks-dolly-15k

    Format various fields of the sample ('instruction', 'context', 'response')
    Then concatenate them using two newline characters
    :param sample: Sample dictionnary
    """
    INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    INSTRUCTION_KEY = "### Instruction:"
    INPUT_KEY = "### Input:"
    RESPONSE_KEY = "### Response:"
    END_KEY = "### End"

    blurb = f"{INTRO_BLURB}"
    instruction = f"{INSTRUCTION_KEY}\n{sample['instruction']}"
    input_context = f"{INPUT_KEY}\n{sample['context']}" if sample["context"] else None
    response = f"{RESPONSE_KEY}\n{sample['response']}"
    end = f"{END_KEY}"

    parts = [
        part for part in [blurb, instruction, input_context, response, end] if part
    ]

    formatted_prompt = "\n\n".join(parts)

    sample["text"] = formatted_prompt

    return sample


def get_model_max_length(model):
    conf = model.config
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            print(f"Found max lenth: {max_length}")
            break
    if not max_length:
        max_length = 1024
        print(f"Using default max length: {max_length}")
    return max_length


def _preprocess_batch(batch, tokenizer, max_length):
    """
    Tokenizing a batch
    """
    return tokenizer(
        batch["text"],
        max_length=max_length + 1, # later dataset.filter will remove samples that exceed max_length
        truncation=True,
    )


# SOURCE https://github.com/databrickslabs/dolly/blob/master/training/trainer.py
def preprocess_dataset(
    tokenizer: AutoTokenizer,
    max_length: int,
    seed,
    dataset,
    create_prompt_formats: Callable,
):
    """Format & tokenize it so it is ready for training
    :param tokenizer (AutoTokenizer): Model Tokenizer
    :param max_length (int): Maximum number of tokens to emit from tokenizer
    """

    # Add prompt to each sample
    print("Preprocessing dataset...")
    dataset = dataset.map(create_prompt_formats)  # , batched=True)

    # Apply preprocessing to each batch of the dataset & and remove 'instruction', 'context', 'response', 'category' fields
    _preprocessing_function = partial(
        _preprocess_batch, max_length=max_length, tokenizer=tokenizer
    )
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        remove_columns=["instruction", "context", "response", "text", "category"],
    )

    # Filter out samples that have input_ids exceeding max_length
    long_sent_ds = dataset.filter(lambda sample: len(sample["input_ids"]) > max_length)
    if len(long_sent_ds) > 0:
        print("abandoning long sentences:")
        print([len(d["input_ids"]) for d in long_sent_ds])
    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) <= max_length)

    # Shuffle dataset
    dataset = dataset.shuffle(seed=seed)

    return dataset
