<div align="center">

<!-- Animated banner using shields + typing SVG -->
<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=32&duration=3000&pause=1000&color=FF4B4B&center=true&vCenter=true&width=700&lines=🔍+Fake+News+Detector;Fine-tuned+RoBERTa+on+LIAR;Real+%2F+Fake+%E2%80%94+Let+the+model+decide." alt="Typing SVG" />

<br/>

<!-- Badges row 1 -->
[![HuggingFace Space](https://img.shields.io/badge/🤗%20Live%20Demo-Spaces-orange?style=for-the-badge)](https://huggingface.co/spaces/ManasMehta/BERT-FakeNews-Detector)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-181717?style=for-the-badge&logo=github)](https://github.com/ManasMehta1110/BERT-FakeNews-Detector)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<!-- Badges row 2 -->
![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/🤗-Transformers-FFD21E?style=flat-square)
![Gradio](https://img.shields.io/badge/Gradio-UI-F97316?style=flat-square)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-LIAR-8B5CF6?style=flat-square)

<br/>

> **Fine-tuned `roberta-base` on the LIAR dataset for binary fake news classification.**  
> Achieves **71.4% F1** and **78.3% recall** on the test split — built and deployed with a live Gradio demo.

</div>

---

## 📌 Table of Contents

- [🚀 Live Demo](#-live-demo)
- [💡 Project Overview](#-project-overview)
- [🗂️ Dataset — LIAR](#️-dataset--liar)
- [🧠 Model Architecture](#-model-architecture)
- [📊 Results](#-results)
- [🗃️ Project Structure](#️-project-structure)
- [⚙️ Setup & Installation](#️-setup--installation)
- [🏋️ Training](#️-training)
- [🔮 Inference](#-inference)
- [🐳 Docker](#-docker)
- [🔭 Potential Improvements](#-potential-improvements)
- [📄 License](#-license)

---

## 🚀 Live Demo

<div align="center">

[![Open in HuggingFace Spaces](https://img.shields.io/badge/▶%20Try%20it%20Live-HuggingFace%20Spaces-FF4B4B?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/spaces/ManasMehta/BERT-FakeNews-Detector)

</div>

The demo runs directly in your browser — paste any political statement and the model returns:
- ✅ `REAL` or ❌ `FAKE`
- Confidence score (e.g. `87.32%`)

**Example inputs to try:**
```
"The unemployment rate hit a record low last quarter."
"The president signed a bill banning all imports from China."
"Scientists confirmed the Earth is flat based on new satellite data."
```

---

## 💡 Project Overview

Political misinformation is a growing threat to public discourse. This project fine-tunes **RoBERTa-base** — a robustly optimized BERT variant — on the **LIAR benchmark dataset** to classify political statements as real or fake.

The pipeline covers everything end-to-end:

```
Raw LIAR Dataset  →  Tokenization  →  RoBERTa Fine-tuning  →  Evaluation  →  Gradio Demo
```

| Property | Value |
|---|---|
| Base Model | `roberta-base` |
| Task | Binary Classification (REAL / FAKE) |
| Dataset | LIAR (`load_dataset("liar")`) |
| Max Token Length | 256 |
| Training Epochs | 5 |
| Batch Size | 16 (train) / 32 (eval) |
| Optimizer | AdamW |
| Loss | Cross-entropy (class-weighted) |

---

## 🗂️ Dataset — LIAR

The [LIAR dataset](https://huggingface.co/datasets/liar) contains ~12,800 short political statements labelled across 6 veracity classes, collected from PolitiFact.

### Original 6-Class Labels

| Label | Meaning |
|---|---|
| `pants-fire` | Completely false |
| `false` | False |
| `barely-true` | Mostly false |
| `half-true` | Mixed |
| `mostly-true` | Mostly true |
| `true` | True |

### Binary Mapping Used in This Project

```
FAKE (0)  ←  pants-fire  |  false  |  barely-true
REAL (1)  ←  half-true   |  mostly-true  |  true
```

### Dataset Splits

| Split | Size |
|---|---|
| Train | 10,240 |
| Validation | 1,284 |
| Test | 1,267 |

---

## 🧠 Model Architecture

```
Input Statement (text)
        │
        ▼
 RobertaTokenizerFast
  (max_length = 256)
        │
        ▼
 roberta-base encoder
  (12 layers, 768 hidden)
        │
        ▼
  [CLS] representation
        │
        ▼
  Dropout (0.1)
        │
        ▼
  Linear(768 → 2)
        │
        ▼
  Softmax → P(FAKE), P(REAL)
```

RoBERTa improves on BERT by training longer, on more data, removing the Next Sentence Prediction objective, and using dynamic masking — making it a stronger baseline for downstream classification.

---

## 📊 Results

Trained with **class-weighted loss** to handle label imbalance in LIAR.

### Validation (Epoch 5)

| Metric | Score |
|---|---|
| 🎯 Accuracy | **66.98%** |
| 📈 F1 Score | **71.88%** |
| 🔬 Precision | 64.52% |
| 🔁 Recall | 81.14% |

### Test Set

| Metric | Score |
|---|---|
| 🎯 Accuracy | **64.54%** |
| 📈 F1 Score | **71.44%** |
| 🔬 Precision | 65.70% |
| 🔁 Recall | **78.27%** |

> **Note:** LIAR is a notoriously hard benchmark. A high recall (78–81%) is intentional — in fake news detection, missing a fake claim (false negative) is more costly than a false alarm.

### Why These Numbers are Solid for LIAR

- Most baseline BERT/RoBERTa models on LIAR binary report **62–68% accuracy**
- This model sits in that range while pushing **recall above 78%** via class weighting
- The gap between validation F1 (71.88%) and test F1 (71.44%) is tiny, indicating no significant overfitting

---

## 🗃️ Project Structure

```
BERT-FakeNews-Detector/
│
├── src/
│   ├── data.py          # Load LIAR, apply binary mapping, return DatasetDict
│   ├── model.py         # RoBERTa classifier wrapper + config
│   ├── train.py         # Fine-tuning script (HuggingFace Trainer API)
│   └── evaluate.py      # Metrics: accuracy, F1, confusion matrix
│
├── app.py               # Gradio inference UI
├── requirements.txt     # Pinned dependencies
├── Dockerfile           # Container for HF Spaces / local deploy
└── README.md
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.10+
- pip
- (Optional) CUDA-capable GPU for faster training

### Install

```bash
git clone https://github.com/ManasMehta1110/BERT-FakeNews-Detector.git
cd BERT-FakeNews-Detector
pip install -r requirements.txt
```

<details>
<summary>📦 Click to see key dependencies</summary>

```
transformers>=4.40.0
datasets>=2.19.0
torch>=2.2.0
scikit-learn>=1.4.0
gradio>=4.0.0
accelerate>=0.29.0
```

</details>

---

## 🏋️ Training

### Standard run (HuggingFace datasets)

```bash
python src/train.py \
  --model_name roberta-base \
  --output_dir outputs/roberta-liar \
  --num_train_epochs 5 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 32
```

### With class-weighted loss (recommended — matches reported results)

```bash
python src/train.py \
  --model_name roberta-base \
  --output_dir outputs/roberta-liar \
  --num_train_epochs 5 \
  --per_device_train_batch_size 16 \
  --use_class_weights true
```

### Using local TSV files instead of HuggingFace datasets

```bash
python src/train.py \
  --train_file data/train.tsv \
  --validation_file data/valid.tsv \
  --test_file data/test.tsv
```

After training, the following are saved to `--output_dir`:

| File | Contents |
|---|---|
| `eval_metrics.json` | Validation accuracy, F1, precision, recall per epoch |
| `test_report.json` | Test set full classification report + confusion matrix |
| `config.json` | Model config (with `id2label` mapping) |
| `model.safetensors` | Fine-tuned weights |

---

## 🔮 Inference

### Run the Gradio app locally

```bash
MODEL_NAME_OR_PATH=outputs/roberta-liar python app.py
```

Then open `http://127.0.0.1:7860` in your browser.

### Run inference in Python directly

```python
from transformers import RobertaForSequenceClassification, RobertaTokenizerFast
import torch

model = RobertaForSequenceClassification.from_pretrained("outputs/roberta-liar")
tokenizer = RobertaTokenizerFast.from_pretrained("outputs/roberta-liar")
model.eval()

statement = "The unemployment rate hit a record low last quarter."
inputs = tokenizer(statement, return_tensors="pt", truncation=True, max_length=256)

with torch.no_grad():
    probs = torch.softmax(model(**inputs).logits, dim=-1)[0]

label = "REAL" if probs.argmax().item() == 1 else "FAKE"
confidence = probs.max().item() * 100
print(f"{label} ({confidence:.1f}% confidence)")
```

---

## 🐳 Docker

Build and run locally in a container:

```bash
docker build -t fake-news-detector .
docker run -p 7860:7860 -e MODEL_NAME_OR_PATH=outputs/roberta-liar fake-news-detector
```

The provided `Dockerfile` is also compatible with **HuggingFace Spaces** — just push the repo and Spaces will handle the build automatically.

---

## 🔭 Potential Improvements

| Improvement | Why It Helps on LIAR |
|---|---|
| **Metadata features** (speaker, party, state) | LIAR includes rich context — political affiliation is highly predictive of statement veracity |
| **`roberta-large`** | Same code, ~3–4% accuracy boost; needs more VRAM |
| **`deberta-v3-base`** | Consistently outperforms RoBERTa on classification benchmarks |
| **Domain-adaptive pretraining** | Further pretraining on political news text aligns language patterns before fine-tuning |
| **Early stopping** | Prevents overfitting on LIAR's noisy labels |
| **Data augmentation** | Paraphrase generation for minority classes to reduce imbalance |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built by [Manas Mehta](https://github.com/ManasMehta1110)**

[![HuggingFace](https://img.shields.io/badge/🤗%20Live%20Demo-Try%20it%20Now-FF4B4B?style=for-the-badge)](https://huggingface.co/spaces/ManasMehta/BERT-FakeNews-Detector)

<sub>If this project helped you, consider giving it a ⭐ on GitHub!</sub>

</div>
