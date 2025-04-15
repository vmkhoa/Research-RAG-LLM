import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def load_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
    base_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype=torch.float32)
    base_model.resize_token_embeddings(len(tokenizer))  #Resize is necessary for proper model loading
    model = PeftModel.from_pretrained(model=base_model, model_id=model_path)
    model.eval()
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def rag_generate(model, tokenizer, query: str, context_chunks: list, max_new_tokens: int = 100):
    context = '\n\n'.join(chunk['chunk'] for chunk in context_chunks)
    prompt = f'{query}\n\nRelevant Context:\n{context}\n\nAnswer:'
    inputs = tokenizer(
        prompt,
        return_tensors='pt',
        truncation=True,
        padding=True,
        max_length=1024
    ).to(model.device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,  
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id  
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
