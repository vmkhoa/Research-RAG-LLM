import os
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling
    )
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

#Configs
model_name = 'microsoft/phi-2'
dataset_path = 'data/train.jsonl'
output_dir = './output'

#Load tokenizer and dataset
tokenizer = AutoTokenizer.from_pretrained(model_name)
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

#Load model with LoRA

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32
)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=['q_proj','k_proj', 'v_proj', 'o_proj'],
    lora_dropout=0.05,
    bias='none',
    task_type='CAUSAL_LM'
)

model = get_peft_model(model, lora_config)

#Training args and trainer
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    learning_rate=2e-4,
    logging_dir=f'{output_dir}/logs',
    save_total_limit=1,
    save_strategy='epoch',
    logging_steps=10,
    disable_tqdm=False,
    evaluation_strategy='no',
    report_to='none'
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
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