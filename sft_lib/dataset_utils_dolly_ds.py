import os
from functools import partial
from typing import Callable

import peft
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments, set_seed)

from datasets import load_dataset


def prompterizer_for_dolly_ds(sample):
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

    return formatted_prompt


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


# SOURCE https://github.com/databrickslabs/dolly/blob/master/training/trainer.py
def preprocess_dataset(
    dataset,
    tokenizer: AutoTokenizer,
    prompterize: Callable,
    seed,
    max_length: int,
    do_shuffle=True,
):
    """Format & tokenize it so it is ready for training
    :param tokenizer (AutoTokenizer): Model Tokenizer
    :param max_length (int): Maximum number of tokens to emit from tokenizer
    """

    def _tokenize_batch(batch, tokenizer, max_length):
        """
        Tokenizing a batch
        """
        # batch = {"text": ["sample1","sample2"]} # batch_size default is 1000
        batch_ids = tokenizer(
            batch["text"],
            # later dataset.filter will remove samples that exceed max_length
            max_length=max_length + 1,
            # padding="longest",
            truncation=True,
            # https://huggingface.co/docs/datasets/process#batch-processing
            # return_tensors="pt", # 前面batch_size必须为1，不然默认batch_size=1000，而这里如果没有padding，sample不等长
            # 但是这个地方设置return_tensors没有用，因为dataset会自动转为ist
        )
        return batch_ids

    def _create_prompt_formats(sample):
        sample["text"] = prompterize(sample)
        return sample

    # Add prompt to each sample
    print("Preprocessing dataset...")
    dataset = dataset.map(_create_prompt_formats)  # , batched=True)

    # Apply preprocessing to each batch of the dataset & and remove 'instruction', 'context', 'response', 'category' fields
    _tokenize_function = partial(
        _tokenize_batch, max_length=max_length, tokenizer=tokenizer
    )
    dataset = dataset.map(
        _tokenize_function,
        remove_columns=["instruction", "context", "response", "text", "category"],
        batched=True,
    )

    # Filter out samples that have input_ids exceeding max_length
    long_sent_ds = dataset.filter(lambda sample: len(sample["input_ids"]) > max_length)
    if len(long_sent_ds) > 0:
        print("abandoning long sentences:")
        print([len(d["input_ids"]) for d in long_sent_ds])
    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) <= max_length)

    # Shuffle dataset
    if do_shuffle:
        dataset = dataset.shuffle(seed=seed)

    return dataset
