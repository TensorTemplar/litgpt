# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union, Dict

import torch
from torch.utils.data import DataLoader, random_split

from litgpt.prompts import PromptStyle
from litgpt.data import DataModule, SFTDataset, LazySFTDataset, get_sft_collate_fn
from litgpt.tokenizer import Tokenizer

class JSONDataSource:
    """A data source that lazily reads samples from a JSON or JSONL file.

    Args:
        file_path: Path to the JSON or JSONL file.
    """

    def __init__(self, file_path: Path, indices: Optional[List[int]] = None) -> None:
        self.file_path = file_path
        self.file_format = self.file_path.suffix
        self.indices = indices
        self._length = None
        self._build_index()

    def _build_index(self):
        if self.file_format == ".jsonl":
            self.sample_offsets = []
            with open(self.file_path, "r", encoding="utf-8") as f:
                offset = 0
                for i, line in enumerate(f):
                    if self.indices is None or i in self.indices:
                        self.sample_offsets.append(offset)
                    offset += len(line.encode("utf-8"))
            self._length = len(self.sample_offsets)
        elif self.file_format == ".json":
            with open(self.file_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
            if self.indices is not None:
                self.data = [self.data[i] for i in self.indices]
            self._length = len(self.data)
        else:
            raise ValueError(f"Unsupported file format: '{self.file_format}'.")

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.file_format == ".jsonl":
            with open(self.file_path, "r", encoding="utf-8") as f:
                f.seek(self.sample_offsets[idx])
                line = f.readline()
                example = json.loads(line)
        elif self.file_format == ".json":
            example = self.data[idx]
        else:
            raise ValueError(f"Unsupported file format: '{self.file_format}'. Expected '.json' or '.jsonl'.")
        return example
    
@dataclass
class JSON(DataModule):
    """Loads JSON or JSONL data for supervised finetuning."""

    json_path: Path
    """A path to a JSON file or a directory with `train.json` and `val.json` containing the data.
    The file(s) should contain a list of samples (dicts). Each dict must have the keys 'instruction' and 'output',
    and can optionally have a key 'input' (see Alpaca)."""
    mask_prompt: bool = False
    """Whether to mask the prompt section from the label (with ``ignore_index``)."""
    val_split_fraction: Optional[float] = None
    """The fraction of the dataset to use for the validation dataset. The rest is used for training.
    Only applies if you passed in a single file to `json_path`."""
    prompt_style: Union[str, PromptStyle] = "alpaca"
    """The style to apply to instruction prompts. See `litgpt.prompts` for a list of available styles."""
    ignore_index: int = -100
    """The index to use for elements to be ignored in the label."""
    seed: int = 42
    """The random seed for creating the train/val splits and shuffling the dataset."""
    num_workers: int = 4
    """How many DataLoader processes to use for loading."""
    use_lazy_loading: bool = False
    """Whether to use lazy loading of the data from disk, instead of storing the data in RAM."""
    shuffle_training_data: bool = True
    """Whether to shuffle the training data every epoch."""
    pin_memory: bool = False
    """Whether to pin memory for the DataLoader."""

    tokenizer: Optional[Tokenizer] = field(default=None, init=False, repr=False)
    batch_size: int = field(default=1, init=False, repr=False)
    max_seq_length: int = field(default=-1, init=False, repr=False)
    train_dataset: Optional[SFTDataset] = field(default=None, init=False, repr=False)
    val_dataset: Optional[SFTDataset] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        super().__init__()
        if self.json_path.is_file() and self.val_split_fraction is None:
            raise ValueError(
                "If `json_path` is a file, you must set `val_split_fraction` to a value between 0 and 1 to split the"
                " data into train and validation sets."
            )
        if self.json_path.is_dir() and self.val_split_fraction is not None:
            raise ValueError(
                "If `json_path` is a directory, it must contain 'train.json' and 'val.json' files and"
                f" hence `val_split_fraction` should not be set. Got `{self.val_split_fraction=}`."
            )
        if not self.json_path.exists():
            raise FileNotFoundError(
                "The `json_path` must be a file or a directory containing 'train.json' and 'val.json' files,"
                f" but '{self.json_path!s}' does not exist."
            )
        if isinstance(self.prompt_style, str):
            self.prompt_style = PromptStyle.from_name(self.prompt_style)

    def connect(
        self, tokenizer: Optional[Tokenizer] = None, batch_size: int = 1, max_seq_length: Optional[int] = None
    ) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_length = -1 if max_seq_length is None else max_seq_length

    def get_data_files(self) -> Tuple[Path, Path]:
        """Return paths to the train and validation data files."""
        if self.json_path.is_file():
            # Single file; we'll use the same file for both train and val datasets
            return self.json_path, self.json_path
        elif self.json_path.is_dir():
            train_file = self.find_split("train")
            val_file = self.find_split("val")
            if not train_file or not val_file:
                raise FileNotFoundError(
                    f"Training and validation files not found in directory '{self.json_path}'."
                )
            return train_file, val_file
        else:
            raise FileNotFoundError(f"'{self.json_path}' does not exist.")

    def setup(self, stage: str = "") -> None:
        common_dataset_kwargs = dict(
            tokenizer=self.tokenizer,
            prompt_style=self.prompt_style,
            max_seq_length=self.max_seq_length,
            mask_prompt=self.mask_prompt,
            ignore_index=self.ignore_index,
        )

        if self.use_lazy_loading:
            DatasetClass = LazySFTDataset

            # RFC: Consider requiring separate parameters for train and val files (breaking change)
            # to avoid the overhead of looking for hardcoded file names,
            # inspection of paths, checking of extensions, and downstream branching in multiple places
            if self.json_path.is_file():
                # Single file to be split into train and val
                total_length = self.get_total_length(self.json_path)
                train_indices, val_indices = self.get_split_indices(total_length)
                train_data_source = JSONDataSource(self.json_path, indices=train_indices)
                val_data_source = JSONDataSource(self.json_path, indices=val_indices)
            elif self.json_path.is_dir():
                # Directory containing 'train.json' and 'val.json'
                train_file = self.find_split("train")
                val_file = self.find_split("val")
                if train_file and val_file:
                    train_data_source = JSONDataSource(train_file)
                    val_data_source = JSONDataSource(val_file)
                else:
                    raise FileNotFoundError(
                        "The `json_path` must be a file or a directory containing 'train.json' and 'val.json' files."
                    )
            else:
                raise FileNotFoundError(
                    "The `json_path` must be a file or a directory containing 'train.json' and 'val.json' files."
                )
            
            train_dataset_kwargs = {
                "data_source": train_data_source,
                **common_dataset_kwargs,
            }
            val_dataset_kwargs = {
                "data_source": val_data_source,
                **common_dataset_kwargs,
            }
        else:
            # we load all data into RAM
            train_data, val_data = self.get_splits()
            DatasetClass = SFTDataset
            train_dataset_kwargs = {
                "data": train_data,
                **common_dataset_kwargs,
            }
            val_dataset_kwargs = {
                "data": val_data,
                **common_dataset_kwargs,
            }

        self.train_dataset = DatasetClass(**train_dataset_kwargs)
        self.val_dataset = DatasetClass(**val_dataset_kwargs)

    def get_total_length(self, file_path: Path) -> int:
        if file_path.suffix == ".jsonl":
            with open(file_path, "r", encoding="utf-8") as f:
                return sum(1 for _ in f)
        elif file_path.suffix == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return len(data)
        else:
            raise ValueError(f"Unsupported file format: '{file_path.suffix}'.")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_training_data,
            generator=torch.Generator().manual_seed(self.seed),
            num_workers=self.num_workers if not self.use_lazy_loading else 0,
            collate_fn=get_sft_collate_fn(max_seq_length=self.max_seq_length, ignore_index=self.ignore_index),
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers if not self.use_lazy_loading else 0,
            collate_fn=get_sft_collate_fn(max_seq_length=self.max_seq_length, ignore_index=self.ignore_index),
            pin_memory=self.pin_memory,
        )

    def get_split_indices(self, total_length: int) -> Tuple[List[int], List[int]]:
        indices = list(range(total_length))
        if self.shuffle_training_data:
            generator = torch.Generator().manual_seed(self.seed)
            indices = torch.randperm(total_length, generator=generator).tolist()
        split = int(total_length * (1 - self.val_split_fraction))
        return indices[:split], indices[split:]

    def get_splits(self) -> Tuple:
        # A single file (gets split into train and test)
        if self.json_path.is_file():
            data = load_split(self.json_path)
            total_length = len(data)
            # Partition the dataset into train and test
            train_indices, val_indices = self.get_split_indices(total_length)
            train_data = [data[i] for i in train_indices]
            val_data = [data[i] for i in val_indices]
            return train_data, val_data

        # A directory containing train.json and val.json
        if (train_file := self.find_split("train")) and (val_file := self.find_split("val")):
            train_data = load_split(train_file)
            val_data = load_split(val_file)
            return train_data, val_data

        raise FileNotFoundError(
            "The `json_path` must be a file or a directory containing 'train.json' and 'val.json' files."
        )

    def find_split(self, split_name: str) -> Optional[Path]:
        for suffix in (".json", ".jsonl"):
            if (file := self.json_path / f"{split_name}{suffix}").is_file():
                return file
        return None


def load_split(json_path: Path) -> Any:
    if json_path.suffix == ".json":
        with open(json_path, "r", encoding="utf-8") as file:
            return json.load(file)
    if json_path.suffix == ".jsonl":
        with open(json_path, "r", encoding="utf-8") as file:
            return [json.loads(line) for line in file]
    else:
        raise ValueError(f"Unsupported file format: {json_path.suffix}. Expected `.json` or `.jsonl`.")
