import torch
from torchinfo import summary
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer

def generate_text(prompt, model, tokenizer, num_iterations=1, new_tokens_per_iter=50):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

    for _ in range(num_iterations):
        # Generate attention mask
        attention_mask = (input_ids != model.config.pad_token_id).long().cuda()

        # Generate new tokens with controlled diversity
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=input_ids.shape[1] + new_tokens_per_iter,
                attention_mask=attention_mask,
                do_sample=True,          # Enable sampling for more natural text
                top_k=50,                 # Limit to top 50 tokens by probability
                top_p=0.95,               # Nucleus sampling for controlled randomness
                temperature=0.7,
                eos_token_id=model.config.eos_token_id
            )

        # Extract newly generated tokens and decode them
        new_tokens = output[0][input_ids.shape[1]:]
        prompt += tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Update input_ids with new tokens for the next generation iteration
        input_ids = torch.cat([input_ids, new_tokens.unsqueeze(0)], dim=1)

    return prompt

def run_benchmark_suite(prompts, model, tokenizer):
    results = []
    for prompt in prompts:
        generated_text = generate_text(prompt, model, tokenizer)
        results.append(generated_text)
    return results

def analyze_results(results_unquantized, results_quantized):
    print("Unquantized Model Results:")
    for i, text in enumerate(results_unquantized):
        print(f"Prompt {i+1}: {text}\n")

    print("\nQuantized Model Results:")
    for i, text in enumerate(results_quantized):
        print(f"Prompt {i+1}: {text}\n")

def main():
    # Load model and tokenizer for LLaMA 2
    model_name = "meta-llama/Llama-2-7b"  # Adjust size if needed (e.g., Llama-2-13b)
    # tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)  # LLaMA 2 may require `use_fast=False`
    # tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    
    tokenizer = LlamaTokenizer.from_pretrained(model_name)


    # Unquantized model
    model_unquantized = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda()
    model_unquantized.config.pad_token_id = model_unquantized.config.eos_token_id

    # Quantized model using bitsandbytes 8-bit quantization
    model_quantized = AutoModelForCausalLM.from_pretrained(
        model_name, 
        load_in_8bit=True, 
        device_map='auto'
    )
    model_quantized.config.pad_token_id = model_quantized.config.eos_token_id

    # Define benchmark prompts
    prompts = [
        "public static void"
        # ,
        # "for i in range(10):",
        # "Hello, how are you?",
        # "Once upon a time"
    ]

    # Run benchmark suite for unquantized model
    results_unquantized = run_benchmark_suite(prompts, model_unquantized, tokenizer)

    # Run benchmark suite for quantized model
    results_quantized = run_benchmark_suite(prompts, model_quantized, tokenizer)

    # Analyze results
    analyze_results(results_unquantized, results_quantized)

if __name__ == "__main__":
    main()
