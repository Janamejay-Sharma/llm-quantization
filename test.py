import torch
from torchinfo import summary
from transformers import AutoModelForCausalLM, AutoTokenizer

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

def main():
    # Load model and tokenizer
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()

    # To avoid generation warnings
    model.config.pad_token_id = model.config.eos_token_id

    # summary(model, input_size=(1, 1), dtypes=[torch.long])

    initial_prompt = "public static void"
    generated_text = generate_text(initial_prompt, model, tokenizer)

    print(generated_text)

if __name__ == "__main__":
    main()