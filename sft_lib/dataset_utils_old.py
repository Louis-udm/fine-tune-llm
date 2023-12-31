import glob
import os
from functools import partial
from typing import Callable

import peft
import torch
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
from sft_lib.prompt_utils import text2prompt


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


tranparent_prompterize = lambda x: x


# SOURCE https://github.com/databrickslabs/dolly/blob/master/training/trainer.py
def preprocess_dataset(
    dataset,
    tokenizer: AutoTokenizer,
    prompterize: Callable,
    seed,
    max_length: int,
    do_shuffle=True,
    abandon_long_sent=True,
):
    """Format & tokenize it so it is ready for training
    :param tokenizer (AutoTokenizer): Model Tokenizer
    :param max_length (int): Maximum number of tokens to emit from tokenizer
    """

    def _create_prompt_formats(sample):
        sample["text"], _ = prompterize(sample["text"])
        return sample

    def _tokenize_batch(batch, tokenizer, max_length, abandon_long_sent):
        """
        Tokenizing a batch
        """
        # batch = {"text": ["sample1","sample2"]} # batch_size default is 1000
        batch_ids = tokenizer(
            batch["text"],
            # later dataset.filter will remove samples that exceed max_length
            max_length=max_length + 1 if abandon_long_sent else max_length,
            # padding="longest",
            truncation=True,
            # https://huggingface.co/docs/datasets/process#batch-processing
            # return_tensors="pt", # 前面batch_size必须为1，不然默认batch_size=1000，而这里如果没有padding，sample不等长
            # 但是这个地方设置return_tensors没有用，因为dataset会自动转为list
        )
        return batch_ids

    # Add prompt to each sample
    print("Preprocessing dataset...")
    dataset = dataset.map(_create_prompt_formats)  # , batched=True)

    # Apply preprocessing to each batch of the dataset & and remove 'instruction', 'context', 'response', 'category' fields
    _tokenize_function = partial(
        _tokenize_batch,
        max_length=max_length,
        tokenizer=tokenizer,
        abandon_long_sent=abandon_long_sent,
    )
    dataset = dataset.map(
        _tokenize_function,
        # remove_columns=["instruction", "context", "response", "text", "category"],
        # remove_columns=["text"],
        batched=True,
        # batch_size=1, # default is 1000
    )

    # Filter out samples that have input_ids exceeding max_length
    # if abandon_long_sent is False, there should not have any sample exceeding max_length (truncated by max_length)
    long_sent_ds = dataset.filter(lambda sample: len(sample["input_ids"]) > max_length)
    if len(long_sent_ds) > 0:
        print("abandoning long sentences:")
        print([len(d["input_ids"]) for d in long_sent_ds])
    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) <= max_length)

    # Shuffle dataset
    if do_shuffle:
        dataset = dataset.shuffle(seed=seed)

    return dataset


# -------- create Special dataset ----
def get_dataset_from_text_files(dir, suffix="txt"):
    # Load text dataset: https://huggingface.co/docs/datasets/nlp_load
    # texts = []
    # ds_files = glob.glob(os.path.join("datasets",dir,f"*.{suffix}"))
    # for f in ds_files:
    #     with open(f, "rt") as fp:
    #         texts.append(fp.read())

    data_files = glob.glob(os.path.join("datasets", dir, f"*.{suffix}"))
    data_files = sorted(data_files)
    ds = load_dataset(
        "text",
        sample_by="document",
        data_files=data_files,
        name="simple_markdown",
        split="train",
    )
    return ds


if __name__ == "__main__":
    ds = get_dataset_from_text_files("simple_markdown_with_answer", suffix="md")
    for sample in ds:
        print(sample)
        break

    tokenizer = AutoTokenizer.from_pretrained(
        "NousResearch/Llama-2-7b-chat-hf", use_auth_token=True
    )

    dataset = preprocess_dataset(
        dataset=ds,
        tokenizer=tokenizer,
        prompterize=text2prompt,
        # for training, max_length can be the max model token length, because the answer is also in the text.
        seed=44,
        max_length=512,
        do_shuffle=True,
    )
    print(f"preprocessed dataset length: {len(dataset)}")
