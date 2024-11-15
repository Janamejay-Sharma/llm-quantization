import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "mistralai/Mistral-7B-Instruct-v0.2"

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

tokenizer = AutoTokenizer.from_pretrained(model_name)

# prompt = "What is string theory in physics?"
# prompt = "Please tell me how a spider use its 10th legs."
# prompt = "Could you please tell me about when Shakespeare visited America?"
# prompt = "What is the Bellman equation? Respond in latex"
# prompt = "Tell me the story of Sudama and Krishna"
prompt = "Tell me the krishna and kansa story"

# prompt = "A square with 3cm per edge, what is its area?"
# prompt = "A sphere with radius 3cm, what is its volume?"

# prompt = "Could you please translate 'I want to study Artificial Inteligence' to Chinese, French, and Hindi?"
# prompt = "Could you please translate '那天早上，当我打开窗户的时候，看到了一个意想不到的景象' to English?"

# prompt = "How do I write a for-loop in Python?"
# prompt = "Could you please help me to write javascript code to do ajax interaction?"
# prompt = "Could you please help me to write R code to do ANOVA task?"
# prompt = "Could you please help me to write python code to do mechaine learing task use XGBoost Framework?"
# prompt = "I have two tables is Mysql database. One is student, it has student_id, class_id, score; Another is class, it has class_id, class_name. How can I" + "\n"
# "select the total score of the student who is not in class_id=2?"

# prompt = """请继续以下的故事：
#     那天早上，当我打开窗户的时候，看到了一个意想不到的景象..."""
# prompt = "李白是哪个朝代的？"

messages = [{"role": "user", "content": prompt}]

tokenized_message = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True)
response_token_ids = model.generate(tokenized_message['input_ids'].cuda(),
                                    attention_mask=tokenized_message['attention_mask'].cuda(),
                                    max_new_tokens=4096,
                                    pad_token_id = tokenizer.eos_token_id)
generated_tokens =response_token_ids[:, len(tokenized_message['input_ids'][0]):]
generated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
print(generated_text)
