import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "mistralai/Mistral-7B-Instruct-v0.2"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Print model information
print(f"Model: {model_name}")
print(f"Tokenizer: {type(tokenizer).__name__}")
print(f"Quantization Level: 4-bit with NF4")

# List of prompts to process
prompts = [
    "What is string theory in physics?",
    "Please tell me how a spider uses its 10th legs.",
    "Could you please tell me about when Shakespeare visited America?",
    "What is the Bellman equation? Respond in latex",
    "Tell me the story of Sudama and Krishna",
    "Tell me the krishna and kansa story",
    "前燕和南燕的关系是什么",
    "A square with 3cm per edge, what is its area?",
    "A sphere with radius 3cm, what is its volume?",
    "Could you please translate 'I want to study Artificial Intelligence' to Chinese, French, and Hindi?",
    "Could you please translate '那天早上，当我打开窗户的时候，看到了一个意想不到的景象' to English?",
    "How do I write a for-loop in Python?",
    "Could you please help me to write JavaScript code to do AJAX interaction?",
    "Could you please help me to write R code to do an ANOVA task?",
    "Could you please help me to write Python code to do a machine learning task using the XGBoost Framework?",
    "I have two tables in a MySQL database. One is student, it has student_id, class_id, score; Another is class, it has class_id, class_name. How can I select the total score of the student who is not in class_id=2?",
    "请继续以下的故事：\n    那天早上，当我打开窗户的时候，看到了一个意想不到的景象...",
    "李白是哪个朝代的？"
]

# Generate responses for each prompt
for prompt in prompts:
    print("-" * 20)
    print(f"USER: {prompt}")
    messages = [{"role": "user", "content": prompt}]
    
    # Tokenize the input
    tokenized_message = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, 
        return_tensors="pt", return_dict=True
    )
    
    # Generate response
    response_token_ids = model.generate(
        tokenized_message['input_ids'].cuda(),
        attention_mask=tokenized_message['attention_mask'].cuda(),
        max_new_tokens=256,  # Set a reasonable limit for token generation
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode generated text
    generated_tokens = response_token_ids[:, len(tokenized_message['input_ids'][0]):]
    generated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    print(f"BOT: {generated_text}\n")
