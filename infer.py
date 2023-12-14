import gc
import glob
import os
from functools import partial

import bitsandbytes as bnb
import fire
import peft
import torch
from torch.utils.data.dataloader import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    TextStreamer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from datasets import Dataset, load_dataset
from sft_lib.dataset_utils import (
    get_dataset_from_text_files,
    preprocess_dataset,
    get_model_max_length,
)
from sft_lib.model_utils import (
    create_4bit_bnb_config,
    load_model_with_adaptor,
    load_model,
)
from sft_lib.prompt_utils import text2prompt

SEED = 44
# Reproducibility
set_seed(SEED)

# BATCH_SIZE = 2


def predict_cli(
    # lora_adaptor_dir="Llama-2-7b-chat-hf_databricks-dolly-15k",
    # lora_adaptor_dir=None,
    lora_adaptor_dir="Llama-2-7b-chat-hf_simple_markdown_with_answer",
    # model_name="microsoft/phi-2",
    model_name="NousResearch/Llama-2-7b-chat-hf",
    output_root="predictions",
    max_prompt_length=1400,
    max_new_length=1400,
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
    # ds_name="dir:simple_markdown",
    ds_name="dir:smd_with_rfp",
    # ds_name=["What is OVHcloud?", "Where is Montreal?", "What is Markdown?"],
    # in cli: --ds-name "eg:I am here||you are there ||bla bla"
):
    if isinstance(ds_name, str) and ds_name.startswith("dir:"):
        ds = get_dataset_from_text_files(ds_name[4:], suffix="md")
    if isinstance(ds_name, str) and ds_name.startswith("eg:"):
        ds = ds_name[3:].split("||")
        ds = Dataset.from_dict({"text": ds})
    if isinstance(ds_name, list):
        ds = ds_name
        ds = Dataset.from_dict({"text": ds})

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Load model and tokenizer
    if lora_adaptor_dir:
        model, tokenizer = load_model_with_adaptor(
            model_name=model_name, lora_adaptor_dir=lora_adaptor_dir
        )
    else:
        bnb_config = create_4bit_bnb_config()
        model, tokenizer = load_model(model_name, bnb_config)

    streamer = TextStreamer(tokenizer)

    max_length = get_model_max_length(model)
    print(
        f"max_model_length: {max_length}, setting max_prompt_length to {max_prompt_length}"
    )

    ds = preprocess_dataset(
        dataset=ds,
        tokenizer=tokenizer,
        prompterize=text2prompt,
        # prompterize=tranparent_prompterize, # 如果想重现rola训练dolly_ds的风格，不要加prompterize
        seed=SEED,
        # for inference, max_length shoud be less than max model token length, and the subtraction is for generation.
        max_length=max_prompt_length,
        batch_size=1,
        do_shuffle=False,
        abandon_long_sent=True,
        with_labels=False,
    )

    for i, sample in enumerate(ds, start=1):
        # transformers/generation/utils.py GenerationMixin.generate.generation_config
        # https://huggingface.co/docs/transformers/main_classes/text_generation
        print(f"\n--------- LLM generation for sample {i}-{len(ds)}:")
        # print(f"input text: {sample['text']}")
        input_ids = torch.tensor(sample["input_ids"]).to(device)
        att_mask = torch.tensor(sample["attention_mask"]).to(device)
        outputs = model.generate(
            input_ids=input_ids.view(1, -1),
            attention_mask=att_mask.view(1, -1),
            max_new_tokens=max_new_length,
            # max_new_tokens=50,
            pad_token_id=tokenizer.eos_token_id,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            streamer=streamer,
        )
        del input_ids
        del att_mask
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
        # print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    fire.Fire(predict_cli)
