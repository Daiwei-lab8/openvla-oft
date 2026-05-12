from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np
from PIL import Image
import tensorflow_datasets as tfds


CONTROL_MODE_PREFIX = "<control mode> joint <control mode>"


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _clean_language_instruction(raw_instruction: str) -> str:
    instruction = raw_instruction.strip()
    lowered = instruction.lower()
    prefix = CONTROL_MODE_PREFIX.lower()
    if lowered.startswith(prefix):
        return instruction[len(CONTROL_MODE_PREFIX) :].strip()
    return instruction


def _stratified_episode_split(
    episode_dirs: List[Path],
    *,
    val_split: float,
    seed: int,
) -> Dict[str, List[Path]]:
    grouped: Dict[str, List[Path]] = {}
    for episode_dir in episode_dirs:
        grouped.setdefault(episode_dir.parent.name, []).append(episode_dir)

    train_paths: List[Path] = []
    val_paths: List[Path] = []
    for group_idx, task_name in enumerate(sorted(grouped)):
        task_episodes = sorted(grouped[task_name])
        rng = random.Random(seed + group_idx)
        rng.shuffle(task_episodes)

        if val_split <= 0 or len(task_episodes) <= 1:
            num_val = 0
        else:
            num_val = max(1, int(round(len(task_episodes) * val_split)))
            num_val = min(num_val, len(task_episodes) - 1)

        val_paths.extend(task_episodes[:num_val])
        train_paths.extend(task_episodes[num_val:])

    return {
        "train": sorted(train_paths),
        "val": sorted(val_paths),
    }


class FrankaDisassembly(tfds.core.GeneratorBasedBuilder):
    """TFDS/RLDS builder for the aligned Franka disassembly dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial RLDS export from aligned Franka disassembly dataset.",
    }

    def __init__(
        self,
        *args,
        source_dir: str | Path | None = None,
        val_split: float = 0.05,
        split_seed: int = 7,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.source_dir = Path(source_dir).resolve() if source_dir is not None else None
        self.val_split = val_split
        self.split_seed = split_seed

    def _info(self) -> tfds.core.DatasetInfo:
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "image": tfds.features.Image(
                                        shape=(None, None, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Primary third-view RGB observation.",
                                    ),
                                    "wrist_image": tfds.features.Image(
                                        shape=(None, None, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Wrist RGB observation.",
                                    ),
                                    "state": tfds.features.Tensor(
                                        shape=(8,),
                                        dtype=np.float32,
                                        doc="Robot proprio state: 7 joint positions + gripper width.",
                                    ),
                                }
                            ),
                            "action": tfds.features.Tensor(
                                shape=(8,),
                                dtype=np.float32,
                                doc="Absolute joint-position target plus gripper width target.",
                            ),
                            "discount": tfds.features.Scalar(
                                dtype=np.float32,
                                doc="Discount, always 1.0 for demonstrations.",
                            ),
                            "reward": tfds.features.Scalar(
                                dtype=np.float32,
                                doc="Reward, 1.0 at the final step and 0.0 otherwise.",
                            ),
                            "is_first": tfds.features.Scalar(dtype=np.bool_, doc="True on first step."),
                            "is_last": tfds.features.Scalar(dtype=np.bool_, doc="True on last step."),
                            "is_terminal": tfds.features.Scalar(dtype=np.bool_, doc="True on last step."),
                            "language_instruction": tfds.features.Text(doc="Task language instruction."),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "episode_path": tfds.features.Text(doc="Path to aligned episode directory."),
                            "task_name": tfds.features.Text(doc="Task group name."),
                        }
                    ),
                }
            )
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        if self.source_dir is None:
            raise ValueError("source_dir must be provided when building FrankaDisassembly.")

        episode_dirs = sorted(path for path in self.source_dir.rglob("episode_*") if path.is_dir())
        if not episode_dirs:
            raise FileNotFoundError(f"No episode_* directories found under {self.source_dir}")

        split_paths = _stratified_episode_split(
            episode_dirs,
            val_split=self.val_split,
            seed=self.split_seed,
        )
        return {
            split_name: self._generate_examples(paths)
            for split_name, paths in split_paths.items()
        }

    def _generate_examples(self, episode_dirs: List[Path]) -> Iterator[Tuple[str, Any]]:
        for episode_dir in episode_dirs:
            meta = _load_json(episode_dir / "meta.json")
            with np.load(episode_dir / "data.npz", allow_pickle=False) as episode_data:
                states = episode_data["observation_state"].astype(np.float32)
                actions = episode_data["action"].astype(np.float32)

            language_instruction = _clean_language_instruction(str(meta["language_command"]))
            third_view_paths = meta["image_streams"]["third_view"]
            wrist_paths = meta["image_streams"]["wrist"]

            episode = []
            num_steps = int(actions.shape[0])
            for idx in range(num_steps):
                image = np.asarray(Image.open(episode_dir / third_view_paths[idx]).convert("RGB"))
                wrist_image = np.asarray(Image.open(episode_dir / wrist_paths[idx]).convert("RGB"))
                episode.append(
                    {
                        "observation": {
                            "image": image,
                            "wrist_image": wrist_image,
                            "state": states[idx],
                        },
                        "action": actions[idx],
                        "discount": np.float32(1.0),
                        "reward": np.float32(float(idx == (num_steps - 1))),
                        "is_first": idx == 0,
                        "is_last": idx == (num_steps - 1),
                        "is_terminal": idx == (num_steps - 1),
                        "language_instruction": language_instruction,
                    }
                )

            sample = {
                "steps": episode,
                "episode_metadata": {
                    "episode_path": str(episode_dir),
                    "task_name": episode_dir.parent.name,
                },
            }
            yield str(episode_dir), sample
