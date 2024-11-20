
# LLM Quantization and Evaluation Project

This project demonstrates the use of quantization techniques to optimize large language models (LLMs) for inference on hardware with limited resources. It includes implementations for interactive chatbots powered by GPT-2 and Mistral-7B-Instruct-v0.2, showcasing 8-bit and 4-bit quantization.

---
## 📖 **Overview**

### What is Quantization?
Quantization is a technique used to reduce the size and computational requirements of deep learning models by representing weights and activations with lower precision. For instance, instead of using 32-bit floating-point numbers, a model can use 8-bit or even 4-bit representations. 

#### Benefits of Quantization
- **Smaller Model Size:** Reduces storage and memory requirements making deployment feasible on edge devices.
- **Faster Inference:** Speeds up computations by using lower precision operations, although there is a quantization-associated overhead.

Quantization allowed us to deploy the **Mistral-7B** (a 15 GB, 7 billion parameter model) on consumer-grade GPUs with limited VRAM, such as the NVIDIA RTX 2060 (6 GB).

---
## 🚀 **Features**
- **Text Completion (`completion.py`):** A script for interactive text generation using GPT-2 or other Hugging Face-supported models.
- **Interactive Chatbot (`chatbot.py`):** A conversational chatbot using the Mistral-7B-Instruct-v0.2 model, optimized with 4-bit quantization.

---
## 📦 **Installation**

1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo-name/llm-quantization.git
   cd llm-quantization
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. **Important:** To use the Mistral-7B model, accept the model’s terms of use on the [Hugging Face page](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2), and authenticate with your Hugging Face token.

---
## 🛠 **Usage**

### Text Completion (`completion.py`)

The `completion.py` script generates text based on a given prompt. By default, it uses the GPT-2 model, a ~470 MB download.

```bash
python completion.py --model_name <model_name> --quantization <8bit|4bit>
```

**Examples:**
- Default (GPT-2 without quantization): 
  ```bash
  python completion.py
  ```
- GPT-2 with 8-bit quantization: 
  ```bash
  python completion.py --model_name gpt2 --quantization 8bit
  ```

### Interactive Chatbot (`chatbot.py`)

The `chatbot.py` script runs a conversational chatbot using the **Mistral-7B-Instruct-v0.2** model (~15 GB download).

**Examples**

- Zero-shot prompt:
    ```bash
    python chatbot.py --prompt "Hello, how are you?"
    ```
- Conversation chain:
    ```bash
    python chatbot.py
    ```

Despite its 15 GB model size, the 4-bit quantized version ran successfully on an RTX 2060 (6 GB VRAM).

---
## 🖥️ **Hardware Requirements**

| Model          | Model Size | Quantization | VRAM Requirement |
|-----------------|---------------------|--------------|-------------------|
| GPT-2          | 470 MB             | 8-bit        | ~2 GB            |
| Mistral-7B     | 15 GB              | 4-bit        | ~6 GB            |

The programs were only tested on CUDA-compatible GPUs. CPU inference is likely to be prohibitively slow.

---
### 📬 **Acknowledgments**
- [Hugging Face Transformers](https://huggingface.co/docs/transformers) for providing pre-trained models and tools.
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) for quantization support.