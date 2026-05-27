<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=28&duration=3000&pause=1000&color=2F80ED&center=true&vCenter=true&width=700&lines=BERT+Fake+News+Detector;Fine-tuned+RoBERTa+on+LIAR;Binary+Classification+%E2%80%94+REAL+vs+FAKE" alt="Typing SVG" />

<br/>

[![HuggingFace Space](https://img.shields.io/badge/Live%20Demo-HuggingFace%20Spaces-orange?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/spaces/ManasMehta/BERT-FakeNews-Detector)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github)](https://github.com/ManasMehta1110/BERT-FakeNews-Detector)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![Gradio](https://img.shields.io/badge/Gradio-Demo-F97316?style=flat-square)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-LIAR-8B5CF6?style=flat-square)

<br/>

> Fine-tuned `roberta-base` on the LIAR benchmark for binary fake news classification.  
> Achieves **71.4% F1** and **78.3% recall** on the held-out test split.

</div>

---

## Table of Contents

- [Live Demo](#live-demo)
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Training](#training)
- [Inference](#inference)
- [Docker](#docker)
- [Potential Improvements](#potential-improvements)
- [License](#license)

---

## Live Demo

<div align="center">

[![Open in HuggingFace Spaces](https://img.shields.io/badge/Try%20the%20Live%20Demo-HuggingFace%20Spaces-FF4B4B?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/spaces/ManasMehta/BERT-FakeNews-Detector)

</div>

The demo runs in-browser. Paste any political statement and the model returns a `REAL` or `FAKE` label alongside a confidence score.

**Example statements to test:**

```
The unemployment rate hit a record low last quarter.
The president signed a bill banning all imports from China.
Scientists confirmed the Earth is flat based on new satellite data.
```

---

## Project Overview

Political misinformation is a measurable problem with real consequences for public discourse. This project fine-tunes **RoBERTa-base** — a robustly optimized BERT variant — on the **LIAR benchmark** to produce a binary classifier for political statement veracity.

The pipeline is fully end-to-end:

```
LIAR Dataset  →  Tokenization  →  RoBERTa Fine-tuning  →  Evaluation  →  Gradio Demo
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
| Loss | Cross-entropy with class weights |

---

## Dataset

The [LIAR dataset](https://huggingface.co/datasets/liar) contains approximately 12,800 short political statements labelled across six veracity classes, sourced from PolitiFact fact-checks.

### Original Label Space

| Label | Description |
|---|---|
| `pants-fire` | Completely false |
| `false` | False |
| `barely-true` | Mostly false |
| `half-true` | Mixed |
| `mostly-true` | Mostly true |
| `true` | True |

### Binary Mapping

The six classes are collapsed into two for this project:

```
FAKE  (0)  <--  pants-fire  |  false  |  barely-true
REAL  (1)  <--  half-true   |  mostly-true  |  true
```

### Splits

| Split | Size |
|---|---|
| Train | 10,240 |
| Validation | 1,284 |
| Test | 1,267 |

---

## Model Architecture

```
Input Statement (text)
        |
        v
 RobertaTokenizerFast
  (max_length = 256)
        |
        v
 roberta-base encoder
  (12 layers, 768 hidden dim)
        |
        v
  [CLS] token representation
        |
        v
  Dropout (p = 0.1)
        |
        v
  Linear (768 -> 2)
        |
        v
  Softmax  ->  P(FAKE),  P(REAL)
```

RoBERTa improves on the original BERT pre-training by using larger batches, more data, longer sequences, and dynamic masking — while removing the Next Sentence Prediction objective. These changes produce a stronger general-purpose encoder for downstream classification tasks.

---

## Results

All runs use **class-weighted cross-entropy** to address label imbalance in the LIAR dataset.

### Validation — Epoch 5

| Metric | Score |
|---|---|
| Accuracy | **66.98%** |
| F1 Score | **71.88%** |
| Precision | 64.52% |
| Recall | 81.14% |

### Test Set

| Metric | Score |
|---|---|
| Accuracy | **64.54%** |
| F1 Score | **71.44%** |
| Precision | 65.70% |
| Recall | **78.27%** |

> LIAR is a notoriously difficult benchmark due to label noise and short, context-free statements. Most baseline BERT/RoBERTa binary classifiers on LIAR report accuracy in the 62–68% range. This model sits within that range while pushing recall above 78%, which is the more important metric for fake news detection — a missed false claim carries greater real-world cost than a false alarm.

The gap between validation F1 (71.88%) and test F1 (71.44%) is negligible, indicating the model generalizes without significant overfitting.

---

## Project Structure

```
BERT-FakeNews-Detector/
|
|-- src/
|   |-- data.py          # Load LIAR, apply binary label mapping, return DatasetDict
|   |-- model.py         # RoBERTa classifier wrapper and config
|   |-- train.py         # Fine-tuning script using HuggingFace Trainer API
|   `-- evaluate.py      # Metrics: accuracy, F1, confusion matrix
|
|-- app.py               # Gradio inference interface
|-- requirements.txt     # Pinned dependencies
|-- Dockerfile           # Container for HuggingFace Spaces or local deployment
`-- README.md
```

---

## Setup and Installation

### Prerequisites

- Python 3.10 or higher
- pip
- A CUDA-capable GPU is optional but recommended for training

### Clone and Install

```bash
git clone https://github.com/ManasMehta1110/BERT-FakeNews-Detector.git
cd BERT-FakeNews-Detector
pip install -r requirements.txt
```

<details>
<summary>Key dependencies</summary>

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

## Training

### Standard run using HuggingFace datasets

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

### Using local TSV files

```bash
python src/train.py \
  --train_file data/train.tsv \
  --validation_file data/valid.tsv \
  --test_file data/test.tsv
```

The following files are written to `--output_dir` after training completes:

| File | Contents |
|---|---|
| `eval_metrics.json` | Validation accuracy, F1, precision, recall per epoch |
| `test_report.json` | Full classification report and confusion matrix on the test split |
| `config.json` | Model configuration including `id2label` mapping |
| `model.safetensors` | Fine-tuned model weights |

---

## Inference

### Launch the Gradio app locally

```bash
MODEL_NAME_OR_PATH=outputs/roberta-liar python app.py
```

Open `http://127.0.0.1:7860` in your browser.

### Run inference directly in Python

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

## Docker

Build and run in a container:

```bash
docker build -t fake-news-detector .
docker run -p 7860:7860 -e MODEL_NAME_OR_PATH=outputs/roberta-liar fake-news-detector
```

The provided `Dockerfile` is also compatible with HuggingFace Spaces deployment. Push the repository and Spaces will build and serve the container automatically.

---

## Potential Improvements

| Improvement | Rationale |
|---|---|
| Metadata features (speaker, party, state) | LIAR includes structured speaker context; political affiliation is empirically predictive of statement veracity |
| `roberta-large` | Drop-in replacement with a ~3–4% accuracy gain at the cost of increased VRAM |
| `deberta-v3-base` | DeBERTa consistently outperforms RoBERTa on GLUE and classification benchmarks |
| Domain-adaptive pretraining | Additional pretraining on political news text before fine-tuning improves in-domain language alignment |
| Early stopping | Reduces overfitting on LIAR's noisy, short-statement labels |
| Data augmentation | Paraphrase generation for minority classes mitigates the label imbalance |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

Built by [Manas Mehta](https://github.com/ManasMehta1110)

[![Live Demo](https://img.shields.io/badge/Live%20Demo-HuggingFace%20Spaces-FF4B4B?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/spaces/ManasMehta/BERT-FakeNews-Detector)

</div>
