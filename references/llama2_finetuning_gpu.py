# %%
# Set up Python environment
# !pip install -r requirements.txt

# %%
# Import libraries
import argparse
import os
from functools import partial

import bitsandbytes as bnb
import peft
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments, set_seed)

from datasets import load_dataset

# Reproducibility
seed = 42
set_seed(seed)

# %% [markdown]
# ### Download LLaMA 2 model
# As mentioned before, LLaMA 2 models come in different flavors which are 7B, 13B, and 70B. Your choice can be influenced by your computational resources. Indeed, larger models require more resources, memory, processing power, and training time.

# %% [markdown]
# To download the model you have been granted access to, **make sure you are logged in to the Hugging Face model hub**. As mentioned in the requirements step, you need to use the `huggingface-cli login` command.
#
# The following function will help us to download the model and its tokenizer. It requires a bitsandbytes configuration that we will define later.


# %%
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


# %%
# Load the databricks dataset from Hugging Face
from datasets import load_dataset

dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

# %% [markdown]
# ### Explore dataset
#
# Once the dataset is downloaded, we can take a look at it to understand what it contains:

# %%
print(f"Number of prompts: {len(dataset)}")
print(f"Column names are: {dataset.column_names}")

# %%
i = 1
import json

for d in dataset:
    print(json.dumps(d, indent=4))
    i += 1
    if i > 3:
        break

# %%
import random

import pandas as pd

# Generate random indices
nb_samples = 3
random_indices = random.sample(range(len(dataset)), nb_samples)
samples = []

for idx in random_indices:
    sample = dataset[idx]

    sample_data = {
        "instruction": sample["instruction"],
        "context": sample["context"],
        "response": sample["response"],
        "category": sample["category"],
    }
    samples.append(sample_data)

# Create a DataFrame and display it
df = pd.DataFrame(samples)
# display(df)

# %% [markdown]
# As we can see, each sample is a dictionary that contains:
# - **An instruction**: What could be entered by the user, such as a question
# - **A context**: Help to interpret the sample
# - **A response**: Answer to the instruction
# - **A category**: Classify the sample between Open Q&A, Closed Q&A, Extract information from Wikipedia, Summarize information from Wikipedia, Brainstorming, Classification, Creative writing

# %%
print(sample)


# %%
def create_prompt_formats(sample):
    """
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


print(create_prompt_formats(sample)["text"])


# %%
# SOURCE https://github.com/databrickslabs/dolly/blob/master/training/trainer.py
def get_max_length(model):
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


def preprocess_batch(batch, tokenizer, max_length):
    """
    Tokenizing a batch
    """
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
    )


# SOURCE https://github.com/databrickslabs/dolly/blob/master/training/trainer.py
def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int, seed, dataset: str):
    """Format & tokenize it so it is ready for training
    :param tokenizer (AutoTokenizer): Model Tokenizer
    :param max_length (int): Maximum number of tokens to emit from tokenizer
    """

    # Add prompt to each sample
    print("Preprocessing dataset...")
    dataset = dataset.map(create_prompt_formats)  # , batched=True)

    # Apply preprocessing to each batch of the dataset & and remove 'instruction', 'context', 'response', 'category' fields
    _preprocessing_function = partial(
        preprocess_batch, max_length=max_length, tokenizer=tokenizer
    )
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        remove_columns=["instruction", "context", "response", "text", "category"],
    )

    # Filter out samples that have input_ids exceeding max_length
    long_sent_ds = dataset.filter(lambda sample: len(sample["input_ids"]) >= max_length)
    print("abandoning long sentences:")
    print([len(d["input_ids"]) for d in long_sent_ds])
    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_length)

    # Shuffle dataset
    dataset = dataset.shuffle(seed=seed)

    return dataset


# %%
def create_bnb_config():
    bnb_config = BitsAndBytesConfig(
        # 对于7b模型，使用4bit量化, 那就是7G/2=3.5G, 大概还有1.3G的内核开销
        # 如果是8bit量化，那就是7G+1.3G=8.3G
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    return bnb_config


# %%
def create_peft_config(modules):
    """
    Create Parameter-Efficient Fine-Tuning config for your model
    :param modules: Names of the modules to apply Lora to
    """
    config = peft.LoraConfig(
        r=16,  # dimension of the updated matrices
        lora_alpha=64,  # parameter for scaling
        target_modules=modules,
        lora_dropout=0.1,  # dropout probability for layers
        bias="none",
        task_type="CAUSAL_LM",
    )

    return config


# %%
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

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


# %%
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


# %%
# Load model from HF with user's token and with bitsandbytes config
# model_name = "meta-llama/Llama-2-7b-hf"
model_name = "NousResearch/Llama-2-7b-chat-hf"
bnb_config = create_bnb_config()
model, tokenizer = load_model(model_name, bnb_config)

# %%
## Preprocess dataset
print(f"original dataset length: {len(dataset)}")
max_length = get_max_length(model)
dataset = preprocess_dataset(tokenizer, max_length, seed, dataset)
print(f"preprocessed dataset length: {len(dataset)}")


# %%
def train(model, tokenizer, dataset, output_dir):
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

    # Print information about the percentage of trainable parameters
    print_trainable_parameters(model)

    # Training parameters
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=TrainingArguments(
            # per_device_train_batch_size=1加gradient_accumulation_steps=4, 相当于每次更新参数的时候，batch_size=4
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=2,
            max_steps=15,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=1,
            output_dir="outputs",
            optim="paged_adamw_8bit",
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    model.config.use_cache = (
        False  # re-enable for inference to speed up predictions for similar inputs
    )

    ### SOURCE https://github.com/artidoro/qlora/blob/main/qlora.py
    # Verifying the datatypes before training

    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes:
            dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items():
        total += v
    for k, v in dtypes.items():
        print(k, v, v / total)

    do_train = True

    # Launch training
    print("Training...")

    if do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        print(metrics)

    ###

    # Saving model
    print("Saving last checkpoint of the model...")
    os.makedirs(output_dir, exist_ok=True)
    trainer.model.save_pretrained(output_dir)

    # Free memory for merging weights
    del model
    del trainer
    torch.cuda.empty_cache()


output_dir = "results/llama2/final_checkpoint"
train(model, tokenizer, dataset, output_dir)

# %%
model = peft.AutoPeftModelForCausalLM.from_pretrained(
    output_dir, device_map="auto", torch_dtype=torch.bfloat16
)
# 需要清空vram
model = model.merge_and_unload()

output_merged_dir = "results/llama2/final_merged_checkpoint"
os.makedirs(output_merged_dir, exist_ok=True)
model.save_pretrained(output_merged_dir, safe_serialization=True)

# save tokenizer for easy inference
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(output_merged_dir)

# %%
# Specify input
text = "What is OVHcloud?"

# Specify device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Tokenize input text
inputs = tokenizer(text, return_tensors="pt").to(device)

# Get answer
# (Adjust max_new_tokens variable as you wish (maximum number of tokens the model can generate to answer the input))
outputs = model.generate(
    input_ids=inputs["input_ids"].to(device),
    attention_mask=inputs["attention_mask"],
    max_new_tokens=50,
    pad_token_id=tokenizer.eos_token_id,
)

# Decode output & print it
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
