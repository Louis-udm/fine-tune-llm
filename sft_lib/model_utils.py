import os
from functools import partial
from typing import Callable

import bitsandbytes as bnb
import peft
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments, set_seed)

from datasets import load_dataset


def create_4bit_bnb_config():
    bnb_config = BitsAndBytesConfig(
        # 对于7b模型，使用4bit量化, 那就是7G/2=3.5G, 大概还有1.3G的内核开销
        # 如果是8bit量化，那就是7G+1.3G=8.3G
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    return bnb_config


def load_model(model_name, bnb_config):
    n_gpus = torch.cuda.device_count()
    max_memory = f"{40960}MB"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",  # dispatch efficiently the model on the available ressources
        max_memory={i: max_memory for i in range(n_gpus)},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)

    # Needed for LLaMA tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def load_model_with_adaptor(
    model_name="NousResearch/Llama-2-7b-chat-hf", lora_adaptor_dir=None
):
    n_gpus = torch.cuda.device_count()
    max_memory = f"{40960}MB"
    if lora_adaptor_dir:
        lora_adaptor_path = os.path.join(
            "results", lora_adaptor_dir, "final_checkpoint"
        )
        model = peft.AutoPeftModelForCausalLM.from_pretrained(
            lora_adaptor_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            # 如果使用rola的配置（加quantization_config, max_memory），生成的text好像不是训练rola的风格，why?
            # 这是采用adapter，也可以测试merge_and_unload后的模型会不会这样。
            # quantization_config=create_4bit_bnb_config(),
            # max_memory={i: max_memory for i in range(n_gpus)},
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    else:
        model, tokenizer = load_model(model_name, create_4bit_bnb_config())

    return model, tokenizer


def create_peft_config(modules):
    """
    Create Parameter-Efficient Fine-Tuning config for your model
    :param modules: Names of the modules to apply Lora to
    """
    peft_config = peft.LoraConfig(
        r=16,  # dimension of the updated matrices
        lora_alpha=64,  # parameter for scaling
        target_modules=modules,
        lora_dropout=0.1,  # dropout probability for layers
        bias="none",
        task_type="CAUSAL_LM",
    )

    return peft_config


# SOURCE https://github.com/artidoro/qlora/blob/main/qlora.py
def find_all_linear_names(model):
    cls = (
        bnb.nn.Linear4bit
    )  # if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if (
        "lm_head" in lora_module_names
    ):  # needed for 16-bit; this layer is for downstream tasks
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def print_trainable_parameters(model, use_4bit=False):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    if use_4bit:
        trainable_params /= 2
    print(
        f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}"
    )


def assemble_trainable_model(model):
    # Apply preprocessing to the model to prepare it by
    # 1 - Enabling gradient checkpointing to reduce memory usage during fine-tuning
    model.gradient_checkpointing_enable()

    # 2 - Using the prepare_model_for_kbit_training method from PEFT
    model = peft.prepare_model_for_kbit_training(model)

    # Get lora module names
    modules = find_all_linear_names(model)

    # Create PEFT config for these modules and wrap the model to PEFT
    peft_config = create_peft_config(modules)
    model = peft.get_peft_model(model, peft_config)

    model.config.use_cache = (
        False  # re-enable for inference to speed up predictions for similar inputs
    )

    return model


def verify_datatypes(assembled_model):
    ### SOURCE https://github.com/artidoro/qlora/blob/main/qlora.py
    # Verifying the datatypes before training

    # Print information about the percentage of trainable parameters
    print_trainable_parameters(assembled_model)

    dtypes = {}
    for _, p in assembled_model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes:
            dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items():
        total += v
    for k, v in dtypes.items():
        print(k, v, v / total)
