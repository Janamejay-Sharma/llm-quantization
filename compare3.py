import random
import numpy as np
import torch
import time
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple
import psutil
import gc

class ModelEvaluator:
    def __init__(self, model_name: str, seed: int = 42):
        self.model_name = model_name
        self.seed = seed
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.set_seed()
        
        # Initialize metrics storage
        self.metrics_history = {
            'unquantized': {'perplexity': [], 'latency': [], 'memory': [], 'gpu_memory': []},
            'quantized': {'perplexity': [], 'latency': [], 'memory': [], 'gpu_memory': []}
        }
        
    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
    
    def measure_gpu_memory(self) -> float:
        """Measure GPU memory usage in MB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 ** 2)
        return 0.0
    
    def measure_system_memory(self) -> float:
        """Measure system memory usage in MB"""
        return psutil.Process().memory_info().rss / (1024 ** 2)
    
    def calculate_model_size(self, model) -> float:
        """Calculate model size in MB"""
        return sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
    
    def evaluate_model(self, model, tokenizer, texts: List[str], batch_size: int = 8) -> Dict[str, float]:
        model.eval()
        losses = []
        latencies = []
        gpu_memories = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Measure latency and memory for each batch
            start_time = time.time()
            gpu_mem_start = self.measure_gpu_memory()
            
            try:
                with torch.no_grad():
                    inputs = tokenizer(batch_texts, 
                                     return_tensors='pt', 
                                     padding=True, 
                                     truncation=True, 
                                     max_length=512)
                    input_ids = inputs['input_ids'].to(self.device)
                    attention_mask = inputs['attention_mask'].to(self.device)
                    
                    if input_ids.size(1) == 0:
                        continue
                    
                    labels = input_ids.clone()
                    outputs = model(input_ids, 
                                  attention_mask=attention_mask, 
                                  labels=labels)
                    
                    loss = outputs.loss.item()
                    losses.append(loss)
            
                latency = time.time() - start_time
                gpu_mem_used = self.measure_gpu_memory() - gpu_mem_start
                
                latencies.append(latency)
                gpu_memories.append(gpu_mem_used)
            
            except Exception as e:
                print(f"Error processing batch {i}: {str(e)}")
                continue
            
            # Clear cache periodically
            if i % (batch_size * 4) == 0:
                torch.cuda.empty_cache()
                gc.collect()
        
        if not losses:
            return {
                'perplexity': float('inf'),
                'latency': 0.0,
                'gpu_memory': 0.0
            }
            
        return {
            'perplexity': np.exp(np.mean(losses)),
            'latency': np.mean(latencies),
            'gpu_memory': np.mean(gpu_memories)
        }
    
    def run_evaluation(self, dataset_name: str = 'wikitext', 
                      dataset_config: str = 'wikitext-2-raw-v1', 
                      split: str = 'test',
                      sample_size: int = 100) -> Tuple[Dict, Dict]:
        """Run comprehensive evaluation on both models"""
        print(f"Loading dataset: {dataset_name}")
        dataset = load_dataset(dataset_name, dataset_config, split=split)
        dataset = dataset.select(range(min(sample_size, len(dataset))))
        
        # Extract texts from dataset
        texts = dataset['text']
        # Filter out empty strings and very short texts
        texts = [text for text in texts if len(text.strip()) > 10]
        
        print(f"Loaded {len(texts)} text samples")
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # Set the padding token to the end-of-sequence token if it doesn’t exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Evaluate unquantized model
        print("\nEvaluating unquantized model...")
        model_unquantized = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
        metrics_unquantized = self.evaluate_model(model_unquantized, tokenizer, texts)
        metrics_unquantized['model_size'] = self.calculate_model_size(model_unquantized)
        
        # Free up memory
        del model_unquantized
        torch.cuda.empty_cache()
        gc.collect()
        
        # Evaluate quantized model
        print("\nEvaluating quantized model...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        
        model_quantized = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto"
        )
        
        metrics_quantized = self.evaluate_model(model_quantized, tokenizer, texts)
        metrics_quantized['model_size'] = self.calculate_model_size(model_quantized)
        
        return metrics_unquantized, metrics_quantized
    
    def visualize_results(self, metrics_unquantized: Dict, metrics_quantized: Dict):
        # Prepare data for visualization
        metrics = ['perplexity', 'latency', 'gpu_memory', 'model_size']
        unquantized_values = [metrics_unquantized[m] for m in metrics]
        quantized_values = [metrics_quantized[m] for m in metrics]
        units = ['', '(s)', '(MB)', '(MB)']  # Units for each metric
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        fig.suptitle('Quantized vs Unquantized Model Comparison', fontsize=20, fontweight='bold')
        
        # Flatten axes for easier iteration
        axes = axes.flatten()
        
        # Create vertical bar plots for each metric
        for idx, (metric, unit, ax) in enumerate(zip(metrics, units, axes)):
            values = [metrics_unquantized[metric], metrics_quantized[metric]]
            bars = ax.bar(['Unquantized', 'Quantized'], values, color=['#1f77b4', '#ff7f0e'], width=0.5)
            
            # Set title and label with units
            ax.set_title(f'{metric.replace("_", " ").title()} {unit}', fontsize=14, fontweight='bold')
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height * 1.05,
                        f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
            
            # Calculate and display percentage difference
            if values[0] != 0:
                pct_diff = ((values[1] - values[0]) / values[0]) * 100
                ax.text(0.5, 0.85, f'Difference: {pct_diff:.1f}%',
                        ha='center', va='top', transform=ax.transAxes,
                        fontsize=12, fontweight='bold',
                        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
            else:
                ax.text(0.5, 0.85, 'N/A (division by zero)',
                        ha='center', va='top', transform=ax.transAxes,
                        fontsize=12, fontweight='bold',
                        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
            
            # Set y-axis label with units
            ax.set_ylabel(unit)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit title
        plt.show()
        
        # Create a summary DataFrame
        summary_df = pd.DataFrame({
            'Metric': metrics,
            'Unquantized': unquantized_values,
            'Quantized': quantized_values,
            'Improvement (%)': [
                (q - u) / u * 100 if u != 0 else float('inf')
                for u, q in zip(unquantized_values, quantized_values)
            ]
        })
        
        print("\nSummary Statistics:")
        print(summary_df.to_string(index=False))



def main():
    # Initialize evaluator
    evaluator = ModelEvaluator("gpt2")
    
    # Run evaluation
    metrics_unquantized, metrics_quantized = evaluator.run_evaluation(
        sample_size=100,  # Adjust based on your needs
    )
    
    # Visualize results
    evaluator.visualize_results(metrics_unquantized, metrics_quantized)

if __name__ == "__main__":
    main()