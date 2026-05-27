from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from datasets import DatasetDict
from transformers import (
    DataCollatorWithPadding,
    RobertaForSequenceClassification,
    RobertaTokenizerFast,
)


@dataclass(frozen=True)
class ModelConfig:
    model_name: str = "roberta-base"
    max_length: int = 256
    text_field: str = "statement"


def get_tokenizer(model_name: str) -> RobertaTokenizerFast:
    return RobertaTokenizerFast.from_pretrained(model_name)


def get_model(
    model_name: str,
    num_labels: int,
    id2label: Dict[int, str],
    label2id: Dict[str, int],
) -> RobertaForSequenceClassification:
    return RobertaForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        problem_type="single_label_classification",
    )


def get_data_collator(tokenizer: RobertaTokenizerFast) -> DataCollatorWithPadding:
    return DataCollatorWithPadding(tokenizer=tokenizer)


def tokenize_dataset(
    dataset: DatasetDict,
    tokenizer: RobertaTokenizerFast,
    *,
    max_length: int,
    text_field: str,
    remove_columns: Optional[list] = None,
) -> DatasetDict:
    def _tokenize(batch):
        return tokenizer(
            batch[text_field],
            truncation=True,
            max_length=max_length,
        )

    if remove_columns is None:
        remove_columns = [
            col for col in dataset["train"].column_names if col != "labels"
        ]

    return dataset.map(_tokenize, batched=True, remove_columns=remove_columns)
