import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import argparse

# Model configuration
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

# Load model and tokenizer
print("Loading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Print model information
print(f"Model: {model_name}")
print(f"Quantization Level: 4-bit with NF4")

def chat(context: list) -> str:
    """Generate a response based on the conversation context."""
    # Tokenize the context
    tokenized_message = tokenizer.apply_chat_template(
        context, tokenize=True, add_generation_prompt=True, 
        return_tensors="pt", return_dict=True
    )
    
    # Generate response
    response_token_ids = model.generate(
        tokenized_message['input_ids'].cuda(),
        attention_mask=tokenized_message['attention_mask'].cuda(),
        max_new_tokens=256,  # Limit response length
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode the response
    generated_tokens = response_token_ids[:, len(tokenized_message['input_ids'][0]):]
    generated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return generated_text

def main():
    parser = argparse.ArgumentParser(description="Interactive Chatbot using Mistral Model with Context")
    parser.add_argument("--prompt", type=str, help="The initial prompt for the chatbot.")
    args = parser.parse_args()

    # Initialize conversation context
    context = []

    if args.prompt:
        print("user:", args.prompt)
        context.append({"role": "user", "content": args.prompt})
        response = chat(context)
        print("bot:", response)
        context.append({"role": "assistant", "content": response})
    else:
        print("Welcome to the Chatbot! Type 'exit' to quit.")
        try:
            while True:
                print("-"*20)
                prompt = input("user: ")
                if prompt.lower() in {"exit", "quit"}:
                    print("Exiting the chatbot. Goodbye!")
                    break
                # Append user message to context
                context.append({"role": "user", "content": prompt})
                response = chat(context)
                print("bot:", response)
                # Append bot response to context
                context.append({"role": "assistant", "content": response})
        except KeyboardInterrupt:
            print("\nReceived keyboard interrupt.")


if __name__ == "__main__":
    main()
