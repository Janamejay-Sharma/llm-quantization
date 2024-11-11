import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def setup_model(model_name="mistralai/Mistral-7B-v0.1", device="cuda"):
    """Setup model and tokenizer with proper configuration."""
    print(f"Loading model {model_name}...")
    
    # Initialize tokenizer with padding and special tokens
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left",
        trust_remote_code=True
    )
    
    # Add pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with optimized settings
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Use half precision for memory efficiency
        device_map="auto",  # Automatically handle model placement
        trust_remote_code=True
    )
    
    return model, tokenizer

def generate_text(
    prompt,
    model,
    tokenizer,
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    num_return_sequences=1,
    device="cuda"
):
    """Generate text using the model with advanced parameters."""
    
    # Tokenize input
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048  # Mistral's context window
    ).to(device)
    
    # Generate with advanced parameters
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=num_return_sequences,
            do_sample=True,  # Enable sampling for more creative outputs
            no_repeat_ngram_size=3,  # Prevent repetition of 3-grams
            length_penalty=1.0,  # Balanced length penalty
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode and clean up the generated text
    generated_texts = [
        tokenizer.decode(output, skip_special_tokens=True)
        for output in outputs
    ]
    
    return generated_texts

def main():
    # Initialize model and tokenizer
    model, tokenizer = setup_model()
    
    # Example prompt in Chinese
    prompt = """请继续以下的故事：
    那天早上，当我打开窗户的时候，看到了一个意想不到的景象..."""
    
    # Generate multiple variations
    generated_texts = generate_text(
        prompt,
        model,
        tokenizer,
        max_new_tokens=200,  # Generate longer responses
        temperature=1,  # Slightly more creative
        num_return_sequences=1  # Generate 3 different continuations
    )
    
    # Print results
    print("\nOriginal prompt:", prompt)
    print("\nGenerated continuations:")
    for i, text in enumerate(generated_texts, 1):
        print(f"\nVersion {i}:")
        print(text)

if __name__ == "__main__":
    main()