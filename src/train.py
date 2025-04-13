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
from tqdm import tqdm

#Configs
model_name = 'microsoft/phi-2'
train_path = 'data/train.jsonl'
val_path = 'data/validation.jsonl'
test_path = 'data/test.jsonl'
output_dir = './output'

#Load tokenizer and dataset
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
train_dataset = load_dataset('json', data_files={'train': train_path})['train']
val_dataset = load_dataset('json', data_files={'validation': val_path})['validation']
test_dataset = load_dataset('json', data_files={'test': test_path})['test']

#Format dataset for training
def format_paper(example):
    return {
        "text": f"Title: {example['title']}\n\nAbstract: {example['abstract']}\n\n{example['text']}"
    }

train_dataset = train_dataset.map(format_paper)
val_dataset = val_dataset.map(format_paper)
test_dataset = test_dataset.map(format_paper)

#Tokenize
def tokenize(example):
    tokenized = tokenizer(
        example["text"],
        truncation=True,
        padding='max_length',
        max_length=1024
    )
    return tokenized

tokenized_train = train_dataset.map(tokenize, batched=True, remove_columns=train_dataset.column_names)
tokenized_val = val_dataset.map(tokenize, batched=True, remove_columns=val_dataset.column_names)
tokenized_test = test_dataset.map(tokenize, batched=True, remove_columns=test_dataset.column_names)

#Load model with LoRA

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  
    torch_dtype=torch.float16  
)



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
    num_train_epochs=6,
    learning_rate=2e-4,
    logging_dir=f'{output_dir}/logs',
    save_total_limit=1,
    save_strategy='epoch',
    logging_steps=10,
    evaluation_strategy='epoch',
    report_to='none',
    fp16=True,                    #Enable mixed precision for speed
    bf16=False,                   
    dataloader_num_workers=2,     #Speed up data loading
    remove_unused_columns=False   #Required for PEFT/LoRA training
)



data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Predict on Test Set
print("\nRunning prediction on test set...")
model.eval()
decoded_outputs = []
for i, example in enumerate(tqdm(tokenized_test, desc="Generating")):
    input_ids = torch.tensor([example['input_ids']]).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    
    decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    decoded_outputs.append(decoded)

#Print and save
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, 'test_predictions.txt')

with open(output_file, 'w') as f:
    for i, output in enumerate(decoded_outputs):
        print(f"\n[Sample {i+1}]\n{'='*40}\n{output}\n")
        f.write(f"[Sample {i+1}]\n{'='*40}\n{output}\n\n")

print(f"\n Test predictions saved to: {output_file}")
