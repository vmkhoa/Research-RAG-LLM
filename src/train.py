import os
import torch
from transformer import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModelling
    )
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from bitsandbytes.config import BitsAndBytesConfig

#Configs
model_name = 'mistralai/Mistral-7B-v0.1'
dataset_path = 'data/train.jsonl'
output_dir = './output'

#Load tokenizer and dataset
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code)
tokenizer.pad_token = tokenizer.eos_token
dataset = load_dataset('json', data_files={'train': dataset_path}['train'])

#Format dataset for training
def format_paper(example):
    return {
        "text": f"Title: {example['title']}\n\nAbstract: {example['abstract']}\n\n{example['text']}"
    }

dataset = dataset.map(format_paper)

#Tokenize
def tokenize(example):
    return tokenizer(
        example['text'],
        truncation=True,
        padding='max_length',
        max_length=1024
        )

tokenized_dataset = dataset.map(tokenize, batched=True)

#Load model with QLoRA
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.float32,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map='auto',
    trust_remote_code=True
)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=['q_proj','k_proj', 'v_proj', 'o_proj']
    lora_dropout=0.05,
    bias='none',
    task_type='CAUSAL_LM'
)

model = get_peft_model(model, lora_config)

#Training args and trainer
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=False,
    logging_dir=f'{output_dir}/logs'
    save_total_limit=2,
    save_strategy='epoch',
    logging_steps=10,
    disable_tqdm=False,
    evaluation_strategy='no',
    report_to='none'
)

data_collator = DataCollatorForLanguageModelling(tokenizer=tokenizer, mlm=False)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)