import glob
import bitsandbytes as bnb
from datasets import load_dataset
from functools import partial
import os
import fire
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
from torch.utils.data.dataloader import DataLoader

from sft_lib.model_utils import load_model_with_adaptor
from sft_lib.prompt_utils import text2prompt
from sft_lib.dataset_utils import preprocess_dataset,get_simple_markdown_dataset

SEED = 44
BATCH_SIZE = 2

def _get_samples_from_ds_files(tokenizer):
    # Load dataset
    texts = []
    ds_files = glob.glob("data/*.md")
    for f in ds_files:
        with open(f, "rt") as fp:
            texts.append(fp.read())

    # texts=["Generate a markdown table, include 2 columns (name, age) and 3 rows (John, 20), (Mary, 30), (Peter, 40)."]
    texts = [text2prompt(t) for t in texts]

    inputs = tokenizer(
        texts[0],
        return_tensors="pt",
        # max_length=max_length,
        truncation=True,
    )

    return texts, inputs


def predict_cli(
    lora_adaptor_dir=None,
    model_name="NousResearch/Llama-2-7b-chat-hf",
    output_root="predictions",
    max_length=4096,
    # num_beams=1,
    # num_return_sequences=1,
    temperature=0.3,
    top_k=5,
    top_p=0.9,
    repetition_penalty=1.0,
    # length_penalty=1.0,
    # no_repeat_ngram_size=3,
    # num_beam_groups=1,
    # diversity_penalty=0.0,
    do_sample=True,
    # do_sample=False,
):
    # Reproducibility
    set_seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    model, tokenizer = load_model_with_adaptor(
        model_name=model_name, lora_adaptor_dir=lora_adaptor_dir
    )

    ds = get_simple_markdown_dataset()
    ds = preprocess_dataset(
        dataset=ds,
        tokenizer=tokenizer,
        prompterize=text2prompt,
        seed=SEED,
        max_length=max_length,
        do_shuffle=False,
    )

    for i, sample in enumerate(ds, start=1):
        # transformers/generation/utils.py GenerationMixin.generate.generation_config
        # https://huggingface.co/docs/transformers/main_classes/text_generation
        print(f"\n--------- LLM generation for sample {i}:")
        input_ids=torch.tensor(sample["input_ids"]).to(device)
        att_mask=torch.tensor(sample["attention_mask"]).to(device)
        outputs = model.generate(
            input_ids=input_ids.view(1,-1),
            attention_mask=att_mask.view(1,-1),
            max_new_tokens=max_length,
            pad_token_id=tokenizer.eos_token_id,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
        )
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    fire.Fire(predict_cli)
