from __future__ import annotations

import os
from typing import Dict, Tuple

import gradio as gr
import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizerFast

MODEL_NAME_OR_PATH = os.getenv("MODEL_NAME_OR_PATH", "outputs/roberta-liar")
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "256"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_id2label(config) -> Dict[int, str]:
    if config.id2label:
        return {int(k): v for k, v in config.id2label.items()}
    return {0: "FAKE", 1: "REAL"}


def load_model_and_tokenizer() -> Tuple[RobertaForSequenceClassification, RobertaTokenizerFast, Dict[int, str]]:
    tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME_OR_PATH)
    model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME_OR_PATH)
    model.to(DEVICE)
    model.eval()
    id2label = _resolve_id2label(model.config)
    return model, tokenizer, id2label


try:
    MODEL, TOKENIZER, ID2LABEL = load_model_and_tokenizer()
except (OSError, Exception) as e:
    raise RuntimeError(
        f"Failed to load model from '{MODEL_NAME_OR_PATH}'. "
        f"Ensure the path exists and contains a valid model. Error: {e}"
    ) from e


@torch.no_grad()
def predict(statement: str) -> Tuple[str, str]:
    if not statement or statement.strip() == "":
        return "", ""

    inputs = TOKENIZER(
        statement,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
    outputs = MODEL(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
    pred_id = int(probs.argmax())
    label = ID2LABEL.get(pred_id, "FAKE")
    confidence = probs[pred_id] * 100.0
    return label, f"{confidence:.2f}%"


EXPLANATION = (
    "RoBERTa is a transformer encoder that reads the full statement at once, "
    "learns contextual word meanings, and then classifies the statement as real or fake "
    "based on patterns it learned during fine-tuning."
)

with gr.Blocks(title="Fake News Detection - RoBERTa") as demo:
    gr.Markdown("# Fake News Detection (RoBERTa on LIAR)")
    gr.Markdown(
        "Enter a short political statement. The model outputs REAL/FAKE with confidence."
    )

    statement_input = gr.Textbox(
        label="Statement",
        placeholder="Type a political claim...",
        lines=3,
    )
    predict_button = gr.Button("Predict")

    label_output = gr.Textbox(label="Prediction", interactive=False)
    confidence_output = gr.Textbox(label="Confidence", interactive=False)

    gr.Markdown(f"**What RoBERTa is doing:** {EXPLANATION}")

    predict_button.click(
        predict,
        inputs=statement_input,
        outputs=[label_output, confidence_output],
    )


if __name__ == "__main__":
    demo.launch(share=True)
