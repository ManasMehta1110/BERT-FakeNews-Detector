from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from datasets import ClassLabel, DatasetDict, load_dataset

LABEL_TO_BINARY: Dict[str, int] = {
    "pants-fire": 0,
    "false": 0,
    "barely-true": 0,
    "half-true": 1,
    "mostly-true": 1,
    "true": 1,
}

BINARY_ID2LABEL: Dict[int, str] = {0: "FAKE", 1: "REAL"}
BINARY_LABEL2ID: Dict[str, int] = {"FAKE": 0, "REAL": 1}


@dataclass(frozen=True)
class DatasetConfig:
    dataset_name: str = "liar"
    text_field: str = "statement"
    label_field: str = "label"
    trust_remote_code: bool = True


def get_label_mappings() -> Dict[str, Dict]:
    return {"id2label": BINARY_ID2LABEL, "label2id": BINARY_LABEL2ID}


def _normalize_label(label_value: object, label_names: Optional[List[str]]) -> str:
    if isinstance(label_value, int):
        if not label_names:
            raise ValueError("Label is int but label names are unavailable.")
        return label_names[label_value]
    if isinstance(label_value, str):
        return label_value
    raise ValueError(f"Unsupported label type: {type(label_value)}")


def _map_to_binary_labels(
    example: Dict,
    *,
    label_field: str,
    label_names: Optional[List[str]],
) -> Dict[str, object]:
    label_text = _normalize_label(example[label_field], label_names)
    if label_text not in LABEL_TO_BINARY:
        raise ValueError(f"Unexpected label: {label_text}")
    return {
        "labels": LABEL_TO_BINARY[label_text],
        "label_text": label_text,
    }


def load_liar_dataset(
    config: DatasetConfig = DatasetConfig(),
    data_files: Optional[Dict[str, str]] = None,
) -> DatasetDict:
    if data_files:
        dataset = load_dataset("csv", data_files=data_files, delimiter="\t")
    else:
        dataset = load_dataset(
            config.dataset_name,
            trust_remote_code=config.trust_remote_code,
        )

    label_names: Optional[List[str]] = None
    label_feature = dataset["train"].features.get(config.label_field)
    if isinstance(label_feature, ClassLabel):
        label_names = label_feature.names

    dataset = dataset.map(
        _map_to_binary_labels,
        fn_kwargs={
            "label_field": config.label_field,
            "label_names": label_names,
        },
    )

    dataset = dataset.filter(
        lambda ex: ex[config.text_field] is not None
        and str(ex[config.text_field]).strip() != ""
    )

    return dataset
