import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
import time
import pandas as pd
import matplotlib.pyplot as plt

def generate_text(prompt, model, tokenizer, num_iterations=1, new_tokens_per_iter=50):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    for _ in range(num_iterations):
        # Generate attention mask
        attention_mask = (input_ids != model.config.pad_token_id).long().to(model.device)

        # Generate new tokens with controlled diversity
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=input_ids.shape[1] + new_tokens_per_iter,
                attention_mask=attention_mask,
                do_sample=True,          # Enable sampling for more natural text
                top_k=50,                # Limit to top 50 tokens by probability
                top_p=0.95,              # Nucleus sampling for controlled randomness
                temperature=0.7,
                eos_token_id=model.config.eos_token_id
            )

        # Extract newly generated tokens and decode them
        new_tokens = output[0][input_ids.shape[1]:]
        prompt += tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Update input_ids with new tokens for the next generation iteration
        input_ids = torch.cat([input_ids, new_tokens.unsqueeze(0)], dim=1)

    return prompt

def calculate_perplexity(prompt, model, tokenizer):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=input_ids)
    loss = outputs.loss
    return torch.exp(loss).item()

def measure_latency(prompt, model, tokenizer):
    start_time = time.time()
    _ = generate_text(prompt, model, tokenizer)
    return time.time() - start_time

def measure_gpu_memory_usage():
    return torch.cuda.memory_allocated() / (1024 ** 2)  # in MB

def run_benchmark_suite(prompts, model, tokenizer):
    results = []
    for prompt in prompts:
        perplexity = calculate_perplexity(prompt, model, tokenizer)
        latency = measure_latency(prompt, model, tokenizer)
        gpu_memory = measure_gpu_memory_usage()

        results.append({
            "prompt": prompt,
            "perplexity": perplexity,
            "latency": latency,
            "gpu_memory": gpu_memory
        })
    return results

def analyze_results(results_unquantized, results_quantized):
    df_unquantized = pd.DataFrame(results_unquantized)
    df_quantized = pd.DataFrame(results_quantized)
    
    df_unquantized['model'] = 'unquantized'
    df_quantized['model'] = 'quantized'
    
    df = pd.concat([df_unquantized, df_quantized], ignore_index=True)
    
    # Print descriptive statistics
    print(df.groupby('model').describe())

    # Plot comparison
    metrics = ['perplexity', 'latency', 'gpu_memory']
    for metric in metrics:
        df_pivot = df.pivot(index='prompt', columns='model', values=metric)
        df_pivot.plot(kind='bar', title=f'Comparison of {metric}')
        plt.ylabel(metric)
        plt.show()

def main():
    # Load model and tokenizer
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Unquantized model
    model_unquantized = AutoModelForCausalLM.from_pretrained(model_name).cuda()
    model_unquantized.config.pad_token_id = model_unquantized.config.eos_token_id

    # Quantized model using bitsandbytes 8-bit quantization
    # Ensure bitsandbytes is installed: pip install bitsandbytes
    model_quantized = AutoModelForCausalLM.from_pretrained(
        model_name, 
        load_in_8bit=True, 
        device_map='auto'
    )
    model_quantized.config.pad_token_id = model_quantized.config.eos_token_id

    # Define benchmark prompts
    prompts = [
        "public static void",
        "for i in range(10):",
        "Hello, how are you?",
        "Once upon a time"
    ]

    # Run benchmark suite for unquantized model
    results_unquantized = run_benchmark_suite(prompts, model_unquantized, tokenizer)

    # Run benchmark suite for quantized model
    results_quantized = run_benchmark_suite(prompts, model_quantized, tokenizer)

    # Analyze results
    analyze_results(results_unquantized, results_quantized)

if __name__ == "__main__":
    main()
