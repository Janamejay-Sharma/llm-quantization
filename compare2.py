import time
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def evaluate_model(model, tokenizer, dataset, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for sample in dataset:
            inputs = tokenizer(sample['text'], return_tensors='pt', truncation=True, max_length=512)
            input_ids = inputs['input_ids'].to(device)

            # Skip samples with empty input_ids
            if input_ids.size(1) == 0:
                continue
            
            labels = input_ids.clone()
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss.item()
            losses.append(loss)

    avg_loss = np.mean(losses) if losses else float('inf')
    perplexity = np.exp(avg_loss)
    return perplexity

def main():
    # Set random seed for reproducibility
    set_seed(42)

    # Use GPU for evaluation and quantization compatibility
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model and tokenizer
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load unquantized model
    print("Loading unquantized model...")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    # Quantize the model with bitsandbytes
    print("Quantizing the model with bitsandbytes...")
    
    # Configure 4-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )

    quantized_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto"
    )

    # Load evaluation dataset
    print("Loading evaluation dataset...")
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    # Limit the dataset for quick evaluation (adjust as needed)
    dataset = dataset.select(range(100))

    # Evaluate unquantized model
    print("\nEvaluating unquantized model...")
    start_time = time.time()
    perplexity_unquantized = evaluate_model(model, tokenizer, dataset, device)
    time_unquantized = time.time() - start_time

    # Evaluate quantized model
    print("\nEvaluating quantized model...")
    start_time = time.time()
    perplexity_quantized = evaluate_model(quantized_model, tokenizer, dataset, device)
    time_quantized = time.time() - start_time

    # Calculate model sizes in megabytes
    size_unquantized = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
    size_quantized = sum(p.numel() * p.element_size() for p in quantized_model.parameters()) / (1024 ** 2)

    # Print evaluation results
    print(f"\nUnquantized Model:")
    print(f"Perplexity: {perplexity_unquantized:.2f}")
    print(f"Inference Time: {time_unquantized:.2f} seconds")
    print(f"Model Size: {size_unquantized:.2f} MB")

    print(f"\nQuantized Model:")
    print(f"Perplexity: {perplexity_quantized:.2f}")
    print(f"Inference Time: {time_quantized:.2f} seconds")
    print(f"Model Size: {size_quantized:.2f} MB")

    # Visualization
    labels = ['Perplexity', 'Inference Time (s)', 'Model Size (MB)']
    unquantized_metrics = [perplexity_unquantized, time_unquantized, size_unquantized]
    quantized_metrics = [perplexity_quantized, time_quantized, size_quantized]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, unquantized_metrics, width, label='Unquantized')
    rects2 = ax.bar(x + width/2, quantized_metrics, width, label='Quantized')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Values')
    ax.set_title('Comparison of Unquantized and Quantized Models')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Attach a text label above each bar, displaying its height
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()