import os
from functools import partial

import bitsandbytes as bnb
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

old_model_name = "NousResearch/Llama-2-7b-chat-hf"

# bnb_config = BitsAndBytesConfig(
#     # 对于7b模型，使用4bit量化, 那就是7G/2=3.5G, 大概还有1.3G的内核开销
#     # 如果是8bit量化，那就是7G+1.3G=8.3G
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
# )
# model = AutoModelForCausalLM.from_pretrained(
#     old_model_name,
#     quantization_config=bnb_config,
#     device_map="auto",  # dispatch efficiently the model on the available ressources
# )


lora_adaptor = "results/llama2/final_checkpoint"
model = peft.AutoPeftModelForCausalLM.from_pretrained(
    lora_adaptor, device_map="auto", torch_dtype=torch.bfloat16
)

# 需要清空vram
model = model.merge_and_unload()

output_merged_dir = "results/llama2/final_merged_checkpoint"
os.makedirs(output_merged_dir, exist_ok=True)
model.save_pretrained(output_merged_dir, safe_serialization=True)

# save tokenizer for easy inference
tokenizer = AutoTokenizer.from_pretrained(old_model_name)
tokenizer.save_pretrained(output_merged_dir)

tokenizer = AutoTokenizer.from_pretrained(old_model_name)


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
