"""
aligned_dataset.py

Map-style dataset that reads the intermediate aligned VLA dataset produced by
`data_post_processing` and exposes it in the format expected by OpenVLA-OFT.
"""

from __future__ import annotations

import json
import math
import random
from bisect import bisect_right
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import IGNORE_INDEX, NUM_ACTIONS_CHUNK, NormalizationType


@lru_cache(maxsize=128)
def _load_npz(path_str: str) -> Dict[str, np.ndarray]:
    with np.load(path_str, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


@lru_cache(maxsize=512)
def _load_json(path_str: str) -> Dict:
    with open(path_str, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


@lru_cache(maxsize=4096)
def _load_rgb_image(path_str: str) -> Image.Image:
    return Image.open(path_str).convert("RGB")


@dataclass(frozen=True)
class EpisodeRecord:
    episode_dir: Path
    task_name: str
    data_path: Path
    meta_path: Path
    num_samples: int
    language_command: str
    third_view_paths: Tuple[str, ...]
    wrist_paths: Tuple[str, ...]


class AlignedVLADataset(Dataset):
    def __init__(
        self,
        dataset_root: Path,
        dataset_name: str,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
        *,
        normalization_type: NormalizationType,
        use_wrist_image: bool = False,
        use_proprio: bool = False,
        train: bool = True,
        val_split: float = 0.05,
        seed: int = 7,
        dataset_statistics: Optional[Dict[str, Dict]] = None,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.dataset_name = dataset_name
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn
        self.normalization_type = normalization_type
        self.use_wrist_image = use_wrist_image
        self.use_proprio = use_proprio
        self.train = train
        self.val_split = val_split
        self.seed = seed

        all_records = self._discover_records(self.dataset_root)
        self.records = self._split_records(all_records, train=train, val_split=val_split, seed=seed)
        if not self.records:
            split_name = "train" if train else "val"
            raise ValueError(f"No aligned episodes found for split={split_name} under {self.dataset_root}")

        self._cumulative_lengths = self._build_cumulative_lengths(self.records)
        if dataset_statistics is None:
            self.dataset_statistics = {self.dataset_name: self._compute_statistics(self.records)}
        else:
            self.dataset_statistics = dataset_statistics

    @staticmethod
    def _build_cumulative_lengths(records: List[EpisodeRecord]) -> List[int]:
        cumulative = []
        total = 0
        for record in records:
            total += record.num_samples
            cumulative.append(total)
        return cumulative

    @staticmethod
    def _discover_records(dataset_root: Path) -> List[EpisodeRecord]:
        records: List[EpisodeRecord] = []
        for episode_dir in sorted(path for path in dataset_root.rglob("episode_*") if path.is_dir()):
            data_path = episode_dir / "data.npz"
            meta_path = episode_dir / "meta.json"
            if not data_path.exists() or not meta_path.exists():
                continue

            episode_data = _load_npz(str(data_path))
            meta = _load_json(str(meta_path))
            num_samples = int(episode_data["action"].shape[0])
            image_streams = meta.get("image_streams", {})
            third_view_paths = tuple(image_streams.get("third_view", []))
            wrist_paths = tuple(image_streams.get("wrist", []))
            if len(third_view_paths) != num_samples:
                raise ValueError(
                    f"{episode_dir} has {num_samples} action steps but {len(third_view_paths)} third-view images."
                )
            if wrist_paths and len(wrist_paths) != num_samples:
                raise ValueError(f"{episode_dir} has mismatched wrist image count.")

            records.append(
                EpisodeRecord(
                    episode_dir=episode_dir,
                    task_name=episode_dir.parent.name,
                    data_path=data_path,
                    meta_path=meta_path,
                    num_samples=num_samples,
                    language_command=str(meta.get("language_command", "")).strip(),
                    third_view_paths=third_view_paths,
                    wrist_paths=wrist_paths,
                )
            )
        return records

    @staticmethod
    def _split_records(
        records: List[EpisodeRecord], *, train: bool, val_split: float, seed: int
    ) -> List[EpisodeRecord]:
        grouped: Dict[str, List[EpisodeRecord]] = {}
        for record in records:
            grouped.setdefault(record.task_name, []).append(record)

        selected: List[EpisodeRecord] = []
        for group_idx, task_name in enumerate(sorted(grouped)):
            task_records = sorted(grouped[task_name], key=lambda record: record.episode_dir.as_posix())
            rng = random.Random(seed + group_idx)
            rng.shuffle(task_records)

            if val_split <= 0 or len(task_records) <= 1:
                num_val = 0
            else:
                num_val = max(1, int(round(len(task_records) * val_split)))
                num_val = min(num_val, len(task_records) - 1)

            if train:
                chosen = task_records[num_val:]
            else:
                chosen = task_records[:num_val]
            selected.extend(chosen)

        return sorted(selected, key=lambda record: record.episode_dir.as_posix())

    def _compute_statistics(self, records: List[EpisodeRecord]) -> Dict:
        all_actions = []
        all_proprios = []
        num_transitions = 0
        for record in records:
            episode_data = _load_npz(str(record.data_path))
            actions = episode_data["action"].astype(np.float32)
            proprios = episode_data["observation_state"].astype(np.float32)
            all_actions.append(actions)
            all_proprios.append(proprios)
            num_transitions += int(actions.shape[0])

        actions = np.concatenate(all_actions, axis=0)
        proprios = np.concatenate(all_proprios, axis=0)
        return {
            "action": {
                "mean": actions.mean(axis=0).astype(np.float32),
                "std": actions.std(axis=0).astype(np.float32),
                "max": actions.max(axis=0).astype(np.float32),
                "min": actions.min(axis=0).astype(np.float32),
                "q01": np.quantile(actions, 0.01, axis=0).astype(np.float32),
                "q99": np.quantile(actions, 0.99, axis=0).astype(np.float32),
            },
            "proprio": {
                "mean": proprios.mean(axis=0).astype(np.float32),
                "std": proprios.std(axis=0).astype(np.float32),
                "max": proprios.max(axis=0).astype(np.float32),
                "min": proprios.min(axis=0).astype(np.float32),
                "q01": np.quantile(proprios, 0.01, axis=0).astype(np.float32),
                "q99": np.quantile(proprios, 0.99, axis=0).astype(np.float32),
            },
            "num_transitions": int(num_transitions),
            "num_trajectories": int(len(records)),
        }

    def _normalize(self, array: np.ndarray, key: str) -> np.ndarray:
        stats = self.dataset_statistics[self.dataset_name][key]
        array = array.astype(np.float32)

        if self.normalization_type == NormalizationType.NORMAL:
            return (array - stats["mean"]) / (stats["std"] + 1e-8)

        if self.normalization_type == NormalizationType.BOUNDS:
            low, high = stats["min"], stats["max"]
        elif self.normalization_type == NormalizationType.BOUNDS_Q99:
            low, high = stats["q01"], stats["q99"]
        else:
            raise ValueError(f"Unsupported normalization type: {self.normalization_type}")

        normalized = np.clip(2.0 * (array - low) / (high - low + 1e-8) - 1.0, -1.0, 1.0)
        zeros_mask = low == high
        normalized[..., zeros_mask] = 0.0
        return normalized.astype(np.float32)

    @staticmethod
    def _build_action_chunk(actions: np.ndarray, index: int) -> np.ndarray:
        max_index = actions.shape[0] - 1
        chunk_indices = np.clip(
            np.arange(index, index + NUM_ACTIONS_CHUNK, dtype=np.int64),
            0,
            max_index,
        )
        return actions[chunk_indices].astype(np.float32)

    def __len__(self) -> int:
        return self._cumulative_lengths[-1]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        episode_idx = bisect_right(self._cumulative_lengths, idx)
        prev_total = 0 if episode_idx == 0 else self._cumulative_lengths[episode_idx - 1]
        step_idx = idx - prev_total
        record = self.records[episode_idx]

        episode_data = _load_npz(str(record.data_path))
        proprio_raw = episode_data["observation_state"][step_idx].astype(np.float32)
        actions_raw = self._build_action_chunk(episode_data["action"], step_idx)
        actions = self._normalize(actions_raw, "action")
        proprio = self._normalize(proprio_raw, "proprio")

        language_command = record.language_command.lower()
        primary_image = _load_rgb_image(str(record.episode_dir / record.third_view_paths[step_idx]))
        pixel_values = self.image_transform(primary_image)

        prompt_builder = self.prompt_builder_fn("openvla")
        current_action_string = self.action_tokenizer(actions[0])
        future_actions_string = "".join(self.action_tokenizer(actions[1:]))
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)

        conversation = [
            {"from": "human", "value": f"What action should the robot take to {language_command}?"},
            {"from": "gpt", "value": action_chunk_string},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)
        input_ids = torch.tensor(input_ids)
        labels = torch.tensor(labels)
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX

        item = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "labels": labels,
            "dataset_name": self.dataset_name,
            "actions": torch.tensor(actions, dtype=torch.float32),
        }

        if self.use_wrist_image:
            if not record.wrist_paths:
                raise ValueError(f"{record.episode_dir} does not contain wrist images.")
            wrist_image = _load_rgb_image(str(record.episode_dir / record.wrist_paths[step_idx]))
            item["pixel_values_wrist"] = self.image_transform(wrist_image)

        if self.use_proprio:
            item["proprio"] = torch.tensor(proprio, dtype=torch.float32)

        return item
