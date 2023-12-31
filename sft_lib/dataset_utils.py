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
from torch.utils.data.dataloader import DataLoader

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
    batch_size: int,
    do_shuffle=True,
    abandon_long_sample=True,
    with_labels=False,
):
    """Format & tokenize it so it is ready for training
    :param tokenizer (AutoTokenizer): Model Tokenizer
    :param max_length (int): Maximum number of tokens to emit from tokenizer
    """

    def _add_file_name(sample, idx, file_names):
        sample["file_name"] = file_names[idx]
        return sample

    def _create_prompt_formats(sample):
        if with_labels:
            sample["text"], sample["label_text"] = prompterize(sample["text"])
        else:
            sample["text"], _ = prompterize(sample["text"])
        return sample

    def _sample_len(sample, tokenizer, max_length):
        """
        Tokenizing a batch
        """
        input_ids = tokenizer(
            sample["text"],
            # later dataset.filter will remove samples that exceed max_length
            # max_length=max_length + 1,
            max_length=max_length
            + 8096,  # just an enough large size for print real length when abandon_long_sent=True
            padding=False,
            truncation=True,
        )
        sample["token_len"] = len(input_ids["input_ids"])
        return sample

    _sample_len_func = partial(
        _sample_len,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    def _tokenize_batch(batch, tokenizer, max_length):
        """
        Tokenizing a batch
        """
        # batch = {"text": ["sample1","sample2"]} # batch_size default is 1000
        batch_ids = tokenizer(
            batch["text"],
            max_length=max_length,
            padding="longest",
            truncation=True,
            # https://huggingface.co/docs/datasets/process#batch-processing
            return_tensors="pt",  # 默认batch_size=1000，必须有padding，因为sample不等长
            # 但是这个地方设置return_tensors没有用，因为dataset会自动转为list
        )
        if with_labels:
            # This `mask` is for ignore calculating loss when training
            text_for_mask = [
                t[: -len(l)] for t, l in zip(batch["text"], batch["label_text"])
            ]
            ids_for_mask = tokenizer(
                text_for_mask,
                max_length=max_length,
                padding="longest",
                truncation=True,
                return_tensors="pt",
            )
            ids_for_mask = ids_for_mask["input_ids"]
            input_ids = batch_ids["input_ids"]
            label_ids = input_ids[:, ids_for_mask.shape[1] - input_ids.shape[1] :]
            batch_ids["labels"] = torch.cat(
                [torch.full(ids_for_mask.shape, -100, dtype=torch.long), label_ids],
                dim=1,
            )

        return batch_ids

    # Apply preprocessing to each batch of the dataset & and remove 'instruction', 'context', 'response', 'category' fields
    _tokenize_function = partial(
        _tokenize_batch,
        max_length=max_length,
        tokenizer=tokenizer,
    )

    print("Preprocessing dataset...")

    dataset = dataset.map(_create_prompt_formats, batched=False)  # , batched=True)

    if (
        abandon_long_sample
    ):  # abandon too long samples, else only truncate to max_length
        file_names = list(dataset.info.download_checksums.keys())
        _add_file_name_func = partial(
            _add_file_name,
            file_names=file_names,
        )
        dataset = dataset.map(_add_file_name_func, batched=False, with_indices=True)
        dataset = dataset.map(_sample_len_func, batched=False)
        long_sent_ds = dataset.filter(lambda sample: sample["token_len"] > max_length)
        if len(long_sent_ds) > 0:
            print("abandoning too long samples:")
            print(
                "\n".join(
                    [
                        f"len: {d['token_len']}, file name: {d['file_name']}"
                        for d in long_sent_ds
                    ]
                )
            )
            dataset = dataset.filter(lambda sample: sample["token_len"] <= max_length)
        dataset = dataset.remove_columns(["token_len", "file_name"])

    if do_shuffle:
        dataset = dataset.shuffle(seed=seed)

    dataset = dataset.map(
        _tokenize_function,
        # remove_columns=["instruction", "context", "response", "text", "category"],
        # remove_columns=["text"],
        batched=True,
        batch_size=batch_size,  # default is 1000
    )

    if with_labels:
        dataset = dataset.remove_columns(["text", "label_text"])
    else:
        dataset = dataset.remove_columns(["text"])

    return dataset


def collate_fn_for_ids_prepared_with_labels(batch):
    new_batch = {}
    new_batch["input_ids"] = torch.tensor([i["input_ids"] for i in batch])
    new_batch["labels"] = torch.tensor([i["labels"] for i in batch])
    new_batch["attention_mask"] = torch.tensor([i["attention_mask"] for i in batch])
    return new_batch


def generate_dataloader(
    dataset,
    tokenizer: AutoTokenizer,
    prompterize: Callable,
    seed,
    max_length: int,
    batch_size: int,
    do_shuffle=True,
    abandon_long_sample=True,
    with_labels=False,
):
    dataset = preprocess_dataset(
        dataset=dataset,
        tokenizer=tokenizer,
        prompterize=prompterize,
        # for training, max_length can be the max model token length, because the answer is also in the text.
        seed=seed,
        max_length=max_length,
        batch_size=batch_size,
        do_shuffle=do_shuffle,
        abandon_long_sample=abandon_long_sample,
        with_labels=with_labels,
    )
    # shuffle have to be False, because the batch token length is different
    # shuffle ds in preprocess_dataset.
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_for_ids_prepared_with_labels,
    )
    return dataloader


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
        # name="simple_markdown",
        name=dir.split("/")[-1].strip(),
        split="train",
        # save_infos=True, # will put to python3.10/site-packages/datasets/packaged_modules/text/README.md
        cache_dir=None,
        download_mode="force_redownload",
    )
    return ds


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(
        "NousResearch/Llama-2-7b-chat-hf", token=True
    )

    # ds = get_dataset_from_text_files("simple_markdown_with_answer", suffix="md")
    # # for sample in ds:
    # #     print(sample)
    # #     break

    # dataset = preprocess_dataset(
    #     dataset=ds,
    #     tokenizer=tokenizer,
    #     prompterize=text2prompt,
    #     # for training, max_length can be the max model token length, because the answer is also in the text.
    #     seed=44,
    #     max_length=512,
    #     batch_size=2,
    #     do_shuffle=True,
    #     abandon_long_sample=True,
    #     with_labels=True,
    # )
    # print(f"len of ds: {len(dataset)}, each: ")
    # for d in dataset:
    #     print(len(d["input_ids"]))

    # print(f"preprocessed dataset length: {len(dataset)}")

    # dl = generate_dataloader(
    #     dataset=ds,
    #     tokenizer=tokenizer,
    #     prompterize=text2prompt,
    #     # for training, max_length can be the max model token length, because the answer is also in the text.
    #     seed=44,
    #     max_length=512,
    #     batch_size=2,
    #     do_shuffle=True,
    #     abandon_long_sample=True,
    #     with_labels=True,
    # )
    # for batch in dl:
    #     print(batch["input_ids"].shape)
    #     # break

    ds = get_dataset_from_text_files("smd_with_rfp", suffix="md")
    dataset = preprocess_dataset(
        dataset=ds,
        tokenizer=tokenizer,
        prompterize=text2prompt,
        # for training, max_length can be the max model token length, because the answer is also in the text.
        seed=44,
        max_length=1400,
        batch_size=1,
        do_shuffle=True,
        abandon_long_sample=True,
        with_labels=False,
    )
    print(f"len of ds: {len(dataset)}, each: ")
    for d in dataset:
        print(len(d["input_ids"]))
