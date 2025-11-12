# Domain-Adaptive & Instruction-Tuned LLM using TinyLlama + LoRA

Fine-tunes **TinyLlama (1.1B)** for **domain-specific adaptation** and **instruction-based alignment** using **LoRA (Low-Rank Adaptation)** — achieving efficient specialization of large language models on limited GPU resources.

---

## 🚀 Project Overview
This project demonstrates **end-to-end fine-tuning of a Large Language Model (LLM)** for both **domain-specific learning** and **instruction-following behavior** using **TinyLlama (1.1B parameters)** and **LoRA**.

The goal was to build a **custom conversational model** capable of understanding **pharmaceutical research documents** and **responding to structured human instructions** — similar to ChatGPT or Alpaca.

Built end-to-end using **Hugging Face Transformers**, **PEFT**, and **Datasets**, this project showcases applied expertise in **LLM adaptation**, **data preprocessing**, and **training optimization**.

---

## 🗂️ Repository Structure
```
├── non_instruction_pretrain_finetune.py   # Domain fine-tuning on pharmaceutical PDFs
├── instruction_finetune.py                # Instruction-based fine-tuning (LoRA)
├── pharma_instruction_data.csv            # Example dataset (if provided)
├── requirements.txt                       # Dependencies
└── README.md                              # Project documentation
```

---

## 🎯 Problem Statement
While base LLMs like Llama or GPT are powerful, they often lack:
- **Industry-specific understanding** (e.g., medical or pharmaceutical terms)
- **Instruction comprehension** (structured human prompts)

This project bridges that gap by:
- Teaching **domain knowledge** through fine-tuning on pharma PDFs.
- Training **instruction adherence** through structured prompt-response pairs.

---

## ⚙️ Technical Stack

| Component | Technology |
|------------|-------------|
| **Base Model** | TinyLlama (1.1B parameters) |
| **Fine-tuning Technique** | LoRA (Parameter Efficient Fine-Tuning) |
| **Frameworks** | Hugging Face Transformers, PEFT, Datasets, Accelerate |
| **Data Sources** | Custom PDFs (Pharma), Alpaca / Domain-specific datasets |
| **Hardware** | GPU (A100 / Colab Pro / Cloud VM) |

---

## 🧠 Model Architecture Overview

### 🔹 TinyLlama (Base Model)
Decoder-only transformer, GPT-style architecture, optimized for lightweight fine-tuning.

### 🔹 LoRA (Low-Rank Adaptation)
Introduces small trainable matrices (`q_proj`, `v_proj`) — drastically reducing compute & memory requirements.

### 🔹 Training Stages
1. **Domain Fine-Tuning** — Teaches model pharma-specific terms from PDFs.
2. **Instruction Fine-Tuning** — Trains model to follow structured human prompts.

---

## 🧩 Implementation Workflow

### **Step 1: Setup Environment**
```bash
pip install -r requirements.txt
```

### **Step 2: Domain Fine-Tuning**
Fine-tune TinyLlama on raw text extracted from pharmaceutical PDFs.

#### Example Command
```bash
python non_instruction_pretrain_finetune.py   --model_name TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T   --pdf_dir ./pdfs   --output_dir ./outputs/domain_model   --epochs 2 --batch_size 2 --use_lora
```

### **Step 3: Instruction Fine-Tuning**
Fine-tune the domain model on instruction datasets for conversational ability.

#### Example Command
```bash
python instruction_finetune.py   --model_name TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T   --data_path ./pharma_instruction_data.csv   --output_dir ./outputs/instruction_model   --epochs 3 --batch_size 1 --use_lora
```

Dataset format should include columns:
```
instruction | input | output
```

---

## 🧪 Inference Example
```python
prompt = "Explain the mechanism of action of Metformin."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

**Sample Output:**
> Metformin activates AMP-activated protein kinase (AMPK), increasing glucose uptake and reducing hepatic glucose production.

---

## 📈 Results Comparison

| Question | Domain Model Response | Instruction-Tuned Response |
|-----------|----------------------|----------------------------|
| Explain Metformin mechanism | Generic description | Detailed pharmacological explanation |
| What is Ezetimibe? | Irrelevant output | Accurate pharmacological definition |
| Summarize mRNA vaccines | Unfocused text | Clear, concise medical summary |

---

## 🔍 Key Learnings
- **LoRA** enables efficient fine-tuning on 1B+ models using limited GPUs.
- **Domain data** enhances factual precision.
- **Instruction tuning** improves coherence and structured responses.
- **Response masking** helps isolate output prediction.

---

## 🚧 Challenges
- GPU memory management for long sequences.
- Cleaning unstructured PDF text.
- Maintaining prompt consistency across datasets.

---

## 🔮 Future Work
- Integrate **RLHF** for alignment.
- Expand datasets to **multi-domain** (finance, legal, healthcare).
- Experiment with **QLoRA (4-bit fine-tuning)** for larger models.
- Deploy using **Gradio** or **Streamlit** for interactive demos.

---

## 📚 References
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [PEFT Library](https://huggingface.co/docs/peft)
- [TinyLlama Model Card](https://huggingface.co/TinyLlama)
- [Alpaca Dataset](https://huggingface.co/datasets/tatsu-lab/alpaca)
- 
---

## 🧾 Key Skills Demonstrated
- LLM fine-tuning with **LoRA / PEFT**
- Domain adaptation & instruction alignment
- Efficient training with **FP16 & gradient accumulation**
- Data preprocessing (PDF parsing, dataset formatting)
- Reproducible ML project structuring for GitHub portfolios

---

## 🏁 Summary
This project showcases a complete LLM fine-tuning workflow — from **domain adaptation** to **instruction alignment** — efficiently implemented using **LoRA** on **TinyLlama**.  
It demonstrates practical expertise in customizing open-source LLMs for **domain-specialized conversational AI**.

