import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def load_model(model_path: str):
    base = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype=torch.float16, device_map="auto")
    model = PeftModel.from_pretrained(base, model_path)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def rag_generate(model, tokenizer, query: str, context_chunks: list, max_new_tokens: int = 100):
    context = '\n\n'.join(chunk['chunk'] for chunk in context_chunks)
    prompt = f'{query}\n\nRelevant Context:\n{context}\n\nAnswer:'
    input_ids = tokenizer(prompt, return_tensors='pt', truncation=True, padding=True, max_length=1024).input_ids.to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
