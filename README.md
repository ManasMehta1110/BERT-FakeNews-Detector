# BERT-Fake-News-Detector (RoBERTa on LIAR)

Fine-tune `roberta-base` on the LIAR dataset for binary fake news classification and deploy a lightweight Gradio demo.

## Overview
- **Task:** Binary classification (REAL vs FAKE)
- **Dataset:** LIAR (`load_dataset("liar")`)
- **Model:** RoBERTa-base with a sequence classification head
- **Stack:** PyTorch, HuggingFace Transformers/Datasets, Gradio

## Label Mapping
Binary mapping used throughout the codebase:
- **FAKE (0):** pants-fire, false, barely-true
- **REAL (1):** half-true, mostly-true, true

## Project Structure
```
BERT-FakeNews_Detector/
├── src/
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── app.py
├── requirements.txt
├── Dockerfile
└── README.md
```

## Setup
```bash
pip install -r requirements.txt
```

This project is designed to run locally in VS Code (GPU if available).

## User-Configurable Values (Fill These In)
- **Model output path:** set `--output_dir` in training (example below).
- **Gradio load path:** set `MODEL_NAME_OR_PATH` when running `app.py`.
- **HuggingFace Hub repo (optional):** `HF_HUB_REPO_ID=your-username/roberta-liar`.
- **HuggingFace Spaces URL:** add your live demo link in Results.

## Training
```bash
python src/train.py \
  --model_name roberta-base \
  --output_dir outputs/roberta-liar \
  --num_train_epochs 3 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 32
```

Optional (class imbalance handling):
```bash
python src/train.py --use_class_weights true
```

### Using Local TSV Files (Optional)
```bash
python src/train.py \
  --train_file train.tsv \
  --validation_file valid.tsv \
  --test_file test.tsv
```

## Evaluation Outputs
After training, the following files are saved to the output directory:
- `eval_metrics.json`
- `test_report.json` (includes confusion matrix)

## Gradio Demo
```bash
MODEL_NAME_OR_PATH=outputs/roberta-liar python app.py
```

## Results
Class-weighted run (`--use_class_weights true`):

Validation (eval_metrics.json, epoch 5.0):
- **Accuracy:** 0.6698
- **F1 Score:** 0.7188
- **Precision:** 0.6452
- **Recall:** 0.8114

Test (test_report.json):
- **Accuracy:** 0.6454
- **F1 Score:** 0.7144
- **Precision:** 0.6570
- **Recall:** 0.7827

- **HuggingFace Spaces Demo:** TODO (add URL)

## Model Architecture
RoBERTa-base is a 12-layer transformer encoder. We add a classification head on top of the `[CLS]` representation and fine-tune end-to-end for binary labels.

## Potential Improvements (Why They Help on LIAR)
- **Class-weighted loss:** LIAR has label imbalance; weighting reduces bias toward the majority class.
- **Metadata features:** LIAR includes speaker/party/context fields; incorporating them can boost accuracy because political context matters.
- **Longer training with early stopping:** Statements are short, so more epochs with early stopping often improves stability.
- **Domain-adaptive pretraining:** Further pretraining on political/news text can better align language patterns before fine-tuning.

## Deployment
Use the provided `Dockerfile` for HuggingFace Spaces or local containerized runs. Update `MODEL_NAME_OR_PATH` to your fine-tuned model path or HF Hub repo.

## TODOs to Fill In
- Add your HuggingFace Spaces URL.
- Optionally link to your saved model on the HF Hub.
