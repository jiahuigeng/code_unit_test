import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    GemmaTokenizer, 
    BitsAndBytesConfig,
    pipeline
)

def get_text_after(text:str, target:str) -> str:
    index = text.find(target)
    if index != -1:
        return text[index + len(target):]
    return text

def get_open_model(model_name, quantization=None):
    """
    Load a pre-trained code generation model and its tokenizer.
    """
    model_configs = {
        "codegemma": {
            "class": GemmaTokenizer,
            "model_class": AutoModelForCausalLM,
            "path": "google/codegemma-7b-it"
        },
        "codellama": {
            "class": AutoTokenizer,
            "model_class": AutoModelForCausalLM,
            "path": "codellama/CodeLlama-7b-hf"
        },
        "starcoder3b": {
            "class": AutoTokenizer,
            "model_class": AutoModelForCausalLM,
            "path": "bigcode/starcoder2-3b"
        },
        "starcoder7b": {
            "class": AutoTokenizer,
            "model_class": AutoModelForCausalLM,
            "path": "bigcode/starcoder2-7b"
        },
        "qwencoder7b": {
            "class": AutoTokenizer,
            "model_class": AutoModelForCausalLM,
            "path": "Qwen/Qwen2.5-Coder-7B-Instruct"
        },
        "qwencoder3b": {
            "class": AutoTokenizer,
            "model_class": AutoModelForCausalLM,
            "path": "Qwen/Qwen2.5-Coder-3B-Instruct"
        },
        "llama": {
            "class": AutoTokenizer,
            "model_class": AutoModelForCausalLM,
            "path": "meta-llama/Llama-3.1-8B"
        },
        "qwen": {
            "class": AutoTokenizer,
            "model_class": AutoModelForCausalLM,
            "path": "Qwen/Qwen2.5-7B-Instruct"
        },
        "deepseek7b": {
            "class": AutoTokenizer,
            "model_class": AutoModelForCausalLM,
            "path": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        },
        "deepseek8b": {
            "class": AutoTokenizer,
            "model_class": AutoModelForCausalLM,
            "path": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
        },


    }

    if model_name.lower() not in model_configs:
        raise ValueError(f"Unsupported model: {model_name}. "
                         f"Supported models: {list(model_configs.keys())}")

    config = model_configs[model_name.lower()]
    kwargs = {}
    if quantization:
        if quantization == '8bit':
            kwargs['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True)
        elif quantization == '4bit':
            kwargs['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True)
        else:
            raise ValueError("Quantization must be '8bit' or '4bit'")
        kwargs['device_map'] = 'auto'
        kwargs['torch_dtype'] = torch.float16

    tokenizer = config['class'].from_pretrained(config['path'])
    model = config['model_class'].from_pretrained(config['path'], **kwargs)
    model = model.eval()
    return model, tokenizer


def prompt_open_model(
    model_name, 
    model, 
    tokenizer, 
    prompt, 
    max_new_tokens=2048, 
    temperature=0.1, 
    system_message=None
):
    """
    Generate code or text using a pre-loaded model and tokenizer.
    """
    
    if 'gemma' in model_name.lower() or 'deepseek' in model_name.lower():
        input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**input_ids, max_new_tokens=max_new_tokens)
        return get_text_after(tokenizer.decode(outputs[0], skip_special_tokens=True), prompt)

    elif 'llama' in model_name.lower():
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        results = generator(
            prompt,
            do_sample=True,
            top_k=10,
            temperature=temperature,
            top_p=0.95,
            num_return_sequences=1,
            max_new_tokens=max_new_tokens,  # Changed from max_length
        )

        return get_text_after(results[0]['generated_text'], prompt)

    elif 'starcoder' in model_name.lower():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
        outputs = model.generate(inputs, max_new_tokens=max_new_tokens, temperature=temperature)
        return get_text_after(tokenizer.decode(outputs[0], skip_special_tokens=True), prompt)

    elif 'qwen' in model_name.lower():
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        return get_text_after(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0], prompt)

    raise ValueError(f"Unsupported model: {model_name}")

if __name__ == "__main__":
    
    prompt = "Write a Python function to calculate the nth Fibonacci number."
    # "codegemma", "codellama", "starcoder3b", "qwencoder3b",  "qwen", "llama", "starcoder7b", "qwencoder7b"
    for model_name in ["deepseek7b", "deepseek8b"]:
        print(f"********************************{model_name}****************************************")
        model, tokenizer = get_open_model(model_name, quantization='4bit')
        output = prompt_open_model(model_name, model, tokenizer, prompt)
        print(output)
