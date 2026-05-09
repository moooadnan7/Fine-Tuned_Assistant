# 🤖 LoRA Fine-tuning — Qwen2.5-1.5B on Customer Order Data

Fine-tuning Qwen2.5-1.5B using LoRA on 100,000 customer order records with HuggingFace Transformers & PEFT.

---

## 🧠 What is This?

This project fine-tunes a local LLM (Qwen2.5-1.5B) on customer order data using LoRA (Low-Rank Adaptation) — a lightweight fine-tuning technique that trains only a small fraction of the model's parameters.

```
100k Order Records
        ↓
  Data Preparation
        ↓
  LoRA Fine-tuning
        ↓
  Fine-tuned Model
        ↓
  Local Inference
```

---

## 🗂️ Project Structure

```
lora-finetune/
├── 📁 data/
│   └── data.csv                  ← training data
├── 📁 src/
│   ├── prepare_data.py           ← data preparation
│   ├── train.py                  ← training script
│   └── inference.py              ← test the model
├── 📁 notebooks/
│   └── finetune.ipynb            ← full Colab notebook
├── 📁 outputs/                   ← saved model (gitignored)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

| Component | Tool |
|-----------|------|
| Base Model | `Qwen/Qwen2.5-1.5B` |
| Fine-tuning | LoRA (r=16, alpha=32) |
| Framework | HuggingFace Transformers |
| PEFT | `peft` library |
| Trainer | `trl` SFTTrainer |
| Data | HuggingFace Datasets |
| Language | Python |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/lora-finetune.git
cd lora-finetune
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add Your Data

Place your `data.csv` file in the `data/` folder with these columns:

```
CustomerName, City, ProductName, Quantity, TotalAmount, PaymentMethod, OrderStatus
```

### 4. Prepare Data

```bash
python src/prepare_data.py
```

### 5. Train (on Colab recommended)

```bash
python src/train.py
```

### 6. Test the Model

```bash
python src/inference.py
```

---

## 📊 Training Config

| Parameter | Value |
|-----------|-------|
| **Model** | Qwen2.5-1.5B |
| **Method** | LoRA |
| **Rank (r)** | 16 |
| **Alpha** | 32 |
| **Epochs** | 3 |
| **Batch Size** | 4 |
| **Learning Rate** | 2e-4 |
| **Max Seq Length** | 512 |
| **Records** | 100,000 |
| **Train Split** | 90% |
| **Eval Split** | 10% |

---

## 📁 Data Format

Your CSV should follow this structure:

```
CustomerName, City, ProductName, Quantity, TotalAmount, PaymentMethod, OrderStatus
John, Cairo, Laptop, 2, 1200, Credit Card, Delivered
Sara, Austin, Phone, 1, 800, Debit Card, Pending
```

Each row is converted to natural language for training:

```
Customer John from Cairo ordered 2 units of Laptop. 
Total amount was 1200 USD. Payment method was Credit Card 
and the order was Delivered.
```

---

## 🔧 LoRA Config

```python
LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

---

## 💻 Hardware Requirements

| Platform | GPU | Recommended |
|----------|-----|-------------|
| Google Colab | T4 15GB | ✅ Free |
| Kaggle | P100 16GB | ✅ Free |
| Local PC | None | ❌ Too slow |
| RunPod | A100 40GB | ✅ Fastest |

---

## 📦 Requirements

```
torch>=2.0.0
transformers>=4.40.0
datasets>=2.0.0
peft>=0.10.0
trl>=0.8.0
accelerate>=0.27.0
bitsandbytes>=0.43.0
pandas>=2.0.0
```

Install all:
```bash
pip install -r requirements.txt
```

---

## 📈 Expected Results

| Metric | Value |
|--------|-------|
| Training Loss | 2.5 → 0.5 |
| Eval Loss | ~0.6 - 0.8 |
| Trainable Params | ~2M (0.14%) |
| Training Time (Colab T4) | ~4-6 hours |

---

## 🔄 LoRA vs QLoRA

| | LoRA | QLoRA |
|--|------|-------|
| **Quantization** | ❌ No | ✅ 4-bit |
| **Memory** | 12GB+ | 6GB+ |
| **Speed** | Slower | Faster |
| **Quality** | Slightly better | Almost same |

---

## 🙌 Credits

Built with:
- [HuggingFace Transformers](https://huggingface.co/transformers)
- [PEFT](https://github.com/huggingface/peft)
- [TRL](https://github.com/huggingface/trl)
- [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-1.5B)
