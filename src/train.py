from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed

from data import DatasetConfig, get_label_mappings, load_liar_dataset
from evaluate import compute_metrics, evaluate_predictions
from model import ModelConfig, get_data_collator, get_model, get_tokenizer, tokenize_dataset

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "roberta-liar"


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**model_inputs)
        logits = outputs.get("logits")
        if self.class_weights is not None and self.class_weights.device != logits.device:
            self.class_weights = self.class_weights.to(logits.device)
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune RoBERTa on LIAR (binary).")
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--num_train_epochs", type=float, default=5)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use_class_weights", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--validation_file", type=str, default=None)
    parser.add_argument("--test_file", type=str, default=None)

    return parser.parse_args()


def _resolve_splits(dataset) -> Dict[str, str]:
    splits = {"train": "train"}
    if "validation" in dataset:
        splits["validation"] = "validation"
    elif "valid" in dataset:
        splits["validation"] = "valid"
    elif "dev" in dataset:
        splits["validation"] = "dev"

    if "test" in dataset:
        splits["test"] = "test"
    return splits


def _compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    counts = np.bincount(labels, minlength=2)
    counts = np.maximum(counts, 1)
    total = counts.sum()
    weights = total / (2.0 * counts)
    return torch.tensor(weights, dtype=torch.float)


def _build_training_arguments(
    args: argparse.Namespace,
    *,
    has_validation: bool,
    fp16: bool,
) -> TrainingArguments:
    eval_strategy = "epoch" if has_validation else "no"
    kwargs = {
        "output_dir": args.output_dir,
        "evaluation_strategy": eval_strategy,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "logging_steps": 50,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "num_train_epochs": args.num_train_epochs,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "load_best_model_at_end": has_validation,
        "metric_for_best_model": "f1",
        "greater_is_better": True,
        "report_to": "none",
        "fp16": fp16,
    }

    params = inspect.signature(TrainingArguments.__init__).parameters
    if "evaluation_strategy" not in params:
        kwargs.pop("evaluation_strategy", None)
        if "eval_strategy" in params:
            kwargs["eval_strategy"] = eval_strategy
        elif "do_eval" in params:
            kwargs["do_eval"] = has_validation

    allowed = set(params.keys()) - {"self"}
    filtered = {key: value for key, value in kwargs.items() if key in allowed}
    return TrainingArguments(**filtered)


def _build_trainer_kwargs(
    *,
    model,
    training_args: TrainingArguments,
    train_dataset,
    eval_dataset,
    tokenizer,
) -> Dict[str, object]:
    kwargs: Dict[str, object] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": get_data_collator(tokenizer),
        "compute_metrics": compute_metrics,
    }

    params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in params:
        kwargs["processing_class"] = tokenizer
    else:
        kwargs["tokenizer"] = tokenizer
    return kwargs


def main() -> None:
    args = _parse_args()
    set_seed(args.seed)

    data_files: Optional[Dict[str, str]] = None
    if args.train_file or args.validation_file or args.test_file:
        data_files = {}
        if args.train_file:
            data_files["train"] = args.train_file
        if args.validation_file:
            data_files["validation"] = args.validation_file
        if args.test_file:
            data_files["test"] = args.test_file

    dataset = load_liar_dataset(DatasetConfig(), data_files=data_files)
    splits = _resolve_splits(dataset)
    has_validation = "validation" in splits

    model_cfg = ModelConfig(model_name=args.model_name, max_length=args.max_length)
    tokenizer = get_tokenizer(model_cfg.model_name)
    tokenized = tokenize_dataset(
        dataset,
        tokenizer,
        max_length=model_cfg.max_length,
        text_field=model_cfg.text_field,
    )

    mappings = get_label_mappings()
    model = get_model(
        model_cfg.model_name,
        num_labels=2,
        id2label=mappings["id2label"],
        label2id=mappings["label2id"],
    )

    fp16 = args.fp16 if args.fp16 is not None else torch.cuda.is_available()

    training_args = _build_training_arguments(
        args,
        has_validation=has_validation,
        fp16=fp16,
    )

    class_weights = None
    if args.use_class_weights:
        train_labels = np.array(tokenized[splits["train"]]["labels"])
        class_weights = _compute_class_weights(train_labels)

    trainer_cls = WeightedTrainer if args.use_class_weights else Trainer
    eval_dataset = tokenized[splits["validation"]] if "validation" in splits else None
    trainer_kwargs = _build_trainer_kwargs(
        model=model,
        training_args=training_args,
        train_dataset=tokenized[splits["train"]],
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    if args.use_class_weights:
        trainer_kwargs["class_weights"] = class_weights

    trainer = trainer_cls(**trainer_kwargs)

    if has_validation:
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=2))

    trainer.train()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    if has_validation:
        eval_metrics = trainer.evaluate()
    else:
        eval_metrics = {"note": "No validation split provided."}

    (output_dir / "eval_metrics.json").write_text(
        json.dumps(eval_metrics, indent=2), encoding="utf-8"
    )

    if "test" in splits:
        predictions = trainer.predict(tokenized[splits["test"]])
        preds = np.argmax(predictions.predictions, axis=-1)
        test_report = evaluate_predictions(predictions.label_ids, preds)
        (output_dir / "test_report.json").write_text(
            json.dumps(test_report, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
