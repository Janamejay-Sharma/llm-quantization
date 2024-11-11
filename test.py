import torch
from torchinfo import summary
from transformers import AutoModelForCausalLM, AutoTokenizer

# model_name = "gpt2"
model_name = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).cuda()

prompt = "你好"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

summary(model, input_size=(1, input_ids.shape[1]), dtypes=[torch.long])

# # Set `pad_token_id` to avoid warnings and ensure reliable generation
# model.config.pad_token_id = model.config.eos_token_id

# # Generate and iteratively expand the prompt
# for i in range(10):
#     # Generate attention mask based on the current input length
#     attention_mask = (input_ids != model.config.pad_token_id).long().cuda()

#     with torch.no_grad():
#         # Generate new text based on the current input
#         output = model.generate(
#             input_ids,
#             max_length=input_ids.shape[1] + 50,  # Limit length to prevent overflow
#             attention_mask=attention_mask,
#             eos_token_id=model.config.eos_token_id,
#             early_stopping=True
#         )
    
#     # Decode the newly generated tokens only (excluding the initial input)
#     new_tokens = output[0][input_ids.shape[1]:]  # Only get new tokens
#     prompt += tokenizer.decode(new_tokens, skip_special_tokens=True)
    
#     # Update input_ids to include the entire prompt for the next iteration
#     input_ids = torch.cat([input_ids, new_tokens.unsqueeze(0)], dim=1)

# print(prompt)


