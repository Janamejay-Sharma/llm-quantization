import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForSequenceClassification, 
    AutoModel
)
from transformers import BitsAndBytesConfig
from datasets import load_dataset
import numpy as np
import random
import time
import gc
import psutil
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Tuple
from bert_score import score as bert_score
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    def __init__(self, model_name: str, seed: int = 42):
        self.model_name = model_name
        self.seed = seed
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.set_seed()
        
        # Initialize metrics storage
        self.metrics_history = {
            'unquantized': {},
            'quantized': {}
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
    
    def evaluate_model(self, model, tokenizer, prompts: List[str], references: List[str],
                       batch_size: int = 1) -> Dict[str, float]:
        model.eval()
        latencies = []
        gpu_memories = []
        generated_texts = []
        max_length = getattr(model.config, 'max_position_embeddings', 512)
        
        # Initialize metrics
        bert_scores = []
        self_bleu_scores = []
        response_relevance_scores = []
        fluency_scores = []
        
        # For computing sentence embeddings
        sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # For fluency evaluation
        fluency_tokenizer = AutoTokenizer.from_pretrained('gpt2', padding_side='left')
        fluency_model = AutoModelForCausalLM.from_pretrained('gpt2').to(self.device)
        fluency_model.eval()
        
        # Process in batches
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_references = references[i:i + batch_size]
            
            # Measure latency and memory for each batch
            start_time = time.time()
            gpu_mem_start = self.measure_gpu_memory()
            max_new_tokens = 50
            try:
                with torch.no_grad():
                    inputs = tokenizer(batch_prompts, 
                                       return_tensors='pt', 
                                       padding=True, 
                                       truncation=True, 
                                       max_length=1024 - max_new_tokens,
                                       padding_side='left').to(self.device)
                    
                    prompt_length = inputs['input_ids'].shape[1]
                    max_allowed_length = 1024
                    max_new_tokens = max(1, max_allowed_length - prompt_length)

                    # Generate outputs
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        top_k=50,
                        top_p=0.95,
                        num_return_sequences=1
                    )
                    
                    # Decode generated texts
                    generated_batch = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    generated_texts.extend(generated_batch)
                
                latency = time.time() - start_time
                gpu_mem_used = self.measure_gpu_memory() - gpu_mem_start
                
                latencies.append(latency)
                gpu_memories.append(gpu_mem_used)
            
            except Exception as e:
                print(f"Error processing batch {i}: {str(e)}")
                continue
            
            # Evaluate additional metrics
            # BERTScore
            P, R, F1 = bert_score(generated_batch, batch_references, lang='en', rescale_with_baseline=True)
            bert_scores.extend(F1.numpy())
            
            # Self-BLEU (Diversity)
            for gen_text in generated_batch:
                tokens = gen_text.split()
                self_bleu = sentence_bleu([tokens], tokens)
                self_bleu_scores.append(self_bleu)
            
            # Response Relevance
            prompt_embeddings = sbert_model.encode(batch_prompts)
            response_embeddings = sbert_model.encode(generated_batch)
            relevance_scores = cosine_similarity(prompt_embeddings, response_embeddings).diagonal()
            response_relevance_scores.extend(relevance_scores)
            
            # Fluency (Perplexity)
            for gen_text in generated_batch:
                encodings = fluency_tokenizer(gen_text, return_tensors='pt')
                input_ids = encodings.input_ids.to(self.device)
                with torch.no_grad():
                    outputs = fluency_model(input_ids, labels=input_ids)
                    loss = outputs.loss
                    perplexity = torch.exp(loss).item()
                    fluency_scores.append(perplexity)
            
            # Clear cache periodically
            if i % (batch_size * 4) == 0:
                torch.cuda.empty_cache()
                gc.collect()
        
        # Compile metrics
        metrics = {
            'latency': np.mean(latencies),
            'gpu_memory': np.mean(gpu_memories),
            'bert_score': np.mean(bert_scores),
            'self_bleu': np.mean(self_bleu_scores),
            'response_relevance': np.mean(response_relevance_scores),
            'fluency': np.mean(fluency_scores),
        }
        
        return metrics
    
    def run_evaluation(self, dataset_name: str = 'conv_ai_2', 
                       split: str = 'train',
                       sample_size: int = 100) -> Tuple[Dict, Dict]:
        """Run comprehensive evaluation on both models"""
        print(f"Loading dataset: {dataset_name}")
        dataset = load_dataset(dataset_name, split=split)
        dataset = dataset.select(range(min(sample_size, len(dataset))))
        
        # Extract prompts and references
        prompts = []
        references = []
        for sample in dataset:
            # Reconstruct the conversation
            dialog = sample['dialog']
            bot_profile = sample['bot_profile']
            user_profile = sample['user_profile']
            
            # Build the conversation context
            context = ''
            # Include bot and user profiles
            context += 'Bot persona: ' + ' '.join(map(str, bot_profile)) + '\n'
            context += 'User persona: ' + ' '.join(map(str, user_profile)) + '\n'
            context += 'Conversation:\n'
            for turn in dialog[:-1]:
                sender = 'Bot' if turn['sender_class'] == 'Bot' else 'User'
                context += f"{sender}: {turn['text']}\n"
            # The last turn is the reference response
            last_turn = dialog[-1]
            prompt = context
            prompts.append(prompt)
            references.append(last_turn['text'])
        
        print(f"Loaded {len(prompts)} samples")
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side='left')
        # Set the padding token to the end-of-sequence token if it doesn’t exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Evaluate unquantized model
        print("\nEvaluating unquantized model...")
        model_unquantized = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
        metrics_unquantized = self.evaluate_model(model_unquantized, tokenizer, prompts, references)
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
            bnb_4bit_quant_type="nf4"
        )
        
        model_quantized = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto"
        )
        
        metrics_quantized = self.evaluate_model(model_quantized, tokenizer, prompts, references)
        metrics_quantized['model_size'] = self.calculate_model_size(model_quantized)
        
        return metrics_unquantized, metrics_quantized
    
    def visualize_results(self, metrics_unquantized: Dict, metrics_quantized: Dict):
        # Prepare data for visualization
        metrics = ['bert_score', 'fluency', 'self_bleu', 'response_relevance', 'latency', 'gpu_memory', 'model_size']
        unquantized_values = [metrics_unquantized[m] for m in metrics]
        quantized_values = [metrics_quantized[m] for m in metrics]
        units = ['', '', '', '', '(s)', '(MB)', '(MB)']  # Units for each metric
        
        # Create figure with subplots
        num_metrics = len(metrics)
        num_rows = (num_metrics + 2) // 3
        fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))
        fig.suptitle('Quantized vs Unquantized Model Comparison', fontsize=20, fontweight='bold')
        
        # Flatten axes for easier iteration
        axes = axes.flatten()
        
        # Create vertical bar plots for each metric
        for idx, (metric, unit, ax) in enumerate(zip(metrics, units, axes)):
            values = [unquantized_values[idx], quantized_values[idx]]
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
        
        # Hide any unused subplots
        for idx in range(len(metrics), len(axes)):
            fig.delaxes(axes[idx])
        
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
        dataset_name='conv_ai_2',
        sample_size=50,  # Adjust based on your needs
    )
    
    # Visualize results
    evaluator.visualize_results(metrics_unquantized, metrics_quantized)

if __name__ == "__main__":
    main()
