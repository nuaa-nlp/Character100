import os
import torch
import transformers
from datasets import load_dataset
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer
from transformers import LlamaForCausalLM, LlamaTokenizer

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
    PrefixTuningConfig,
    TaskType
)

train_dataset = load_dataset('json', data_files='/Data/discriminator_train.json', split='train')

def formatting_func(example):
    output_texts = []
    for i in range(len(example['input'])):
        text = f"{example['input'][i]}\n{example['output'][i]}"
        output_texts.append(text)
    return output_texts

base_model_name = '/llama2-7b-chat-hf'

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    use_auth_token=True
)
base_model.config.use_cache = False

base_model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

output_dir = "./llama-2-7b-chat-discriminator-lora"

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    learning_rate=2e-4,
    logging_steps=50,
    num_train_epochs=10,
    logging_dir="./logs",        # Directory for storing logs
    save_strategy="steps",       # Save the model checkpoint every logging step
    save_steps=50,                # Save checkpoints every 50 steps
    # evaluation_strategy="steps", # Evaluate the model every logging step
    # eval_steps=50,               # Evaluate and save checkpoints every 50 steps
    # do_eval=True                 # Perform evaluation at the end of training
)

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=8,
    bias="none",
    task_type="CAUSAL_LM",
)

max_seq_length = 512
trainer = SFTTrainer(
    model=base_model,
    train_dataset=train_dataset,
    # eval_dataset=eval_dataset,
    peft_config=peft_config,
    formatting_func=formatting_func,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_args,
)

# pass in resume_from_checkpoint=True to resume from a checkpoint
trainer.train()