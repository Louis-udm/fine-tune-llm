from datasets import load_dataset
from typing import Callable
from functools import partial
import glob
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
from sft_lib.prompt_utils import text2prompt
from torch.utils.data.dataloader import DataLoader


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
    batch_size: int,
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
        # print("-----=========--------")
        # print(batch)
        # batch = {"text": ["sample1","sample2"]} # batch_size = 2
        batch_ids = tokenizer(
            batch["text"],
            # later dataset.filter will remove samples that exceed max_length
            max_length=max_length + 1,
            padding="longest",
            truncation=True,
            return_tensors="pt", # 这里虽然可以是tensor，但是最后会变成list，why?
        )
        # print(batch_ids)
        # batch_ids={"input_ids": tensor.size=(2,367), "attention_mask": tensor.size=(2,367)}
        return batch_ids

    def _create_prompt_formats(sample):
        # print("-------****------")
        # print(sample)
        sample["text"] = prompterize(sample["text"])
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
        # remove_columns=["instruction", "context", "response", "text", "category"],
        remove_columns=["text"],
        batched=True,
        batch_size=batch_size,
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


def collate_fn(batch):
    # torch DataLoader batch = 
    # [
    #    {"input_ids": [sample1's input_ids in list], "attention_mask": [sample1's att mask in list]}, 
    #     ...
    #    {"input_ids": [sampleN's input_ids in list], "attention_mask": [sampleN's att mask in list]},
    # ]
    new_batch = {}
    new_batch["input_ids"] = torch.tensor([i["input_ids"] for i in batch])
    new_batch["attention_mask"] = torch.tensor([i["attention_mask"] for i in batch])
    # print(new_batch)
    return new_batch


# -------- create special dataset ----
def get_simple_markdown_dataset():
    # Load text dataset: https://huggingface.co/docs/datasets/nlp_load
    texts = []
    ds_files = glob.glob("datasets/simple_markdown/*.md")
    for f in ds_files:
        with open(f, "rt") as fp:
            texts.append(fp.read())

    ds = load_dataset(
        "text",
        sample_by="document",
        data_files="datasets/simple_markdown/*.md",
        name="simple_markdown",
        split="train",
    )
    return ds


if __name__ == "__main__":
    batch_size=2
    max_length=4096
    SEED=44
    model_name="NousResearch/Llama-2-7b-chat-hf"
    tokenizer=AutoTokenizer.from_pretrained(model_name)

    ds = get_simple_markdown_dataset()
    for sample in ds:
        print(sample)
    # 如果要做suffle，必须收集text的时候做, 因为shuffle后可能导致不同长度的input_ids被放在一起
    # 这个python file可以做到，不同batch的input_ids长度不一样，batch内部的input_ids长度一样
    ds = preprocess_dataset(
        dataset=ds,
        tokenizer=tokenizer,
        prompterize=text2prompt,
        seed=SEED,
        max_length=max_length,
        batch_size=batch_size,
        do_shuffle=False,
    )
    dataloader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    for step, batch in enumerate(dataloader, start=1):
        print(f"step={step}")
        print(batch["input_ids"].shape)
        print(batch["attention_mask"].shape)
        break
