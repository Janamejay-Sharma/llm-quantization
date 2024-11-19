import logging
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Suppress logging
logging.getLogger("transformers").setLevel(logging.ERROR)

def generate_text(prompt, model, tokenizer, new_tokens_per_iter=50):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

    # Generate attention mask
    attention_mask = (input_ids != model.config.pad_token_id).long().cuda()

    # Generate new tokens with controlled diversity
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=input_ids.shape[1] + new_tokens_per_iter,
            attention_mask=attention_mask,
            do_sample=True,          
            top_k=50,                
            top_p=0.95,              
            temperature=0.7,
            eos_token_id=model.config.eos_token_id
        )

    new_tokens = output[0][input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)

def load_model(model_name="gpt2", quantization=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load model with quantization options if specified
    if quantization == "8bit":
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            load_in_8bit=True,
            device_map="auto"
        )
    elif quantization == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name).cuda()

    model.config.pad_token_id = model.config.eos_token_id
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser(description="Run a local LLM chatbot")
    parser.add_argument("model_name", nargs="?", default="gpt2", help="Name of the model (default: 'gpt2')")
    parser.add_argument("--quantization", choices=["8bit", "4bit"], help="Quantization level: 8bit or 4bit")
    args = parser.parse_args()

    # Load the model and tokenizer with specified model name and quantization
    model, tokenizer = load_model(model_name=args.model_name, quantization=args.quantization)

    print("Chatbot ready! Type your message or 'exit' to quit.")
    while True:
        prompt = input("You: ")
        if prompt.lower() == "exit":
            break
        response = generate_text(prompt, model, tokenizer)
        print(f"Bot: {response}")

if __name__ == "__main__":
    main()
