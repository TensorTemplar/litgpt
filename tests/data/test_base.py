# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import pytest
import torch

from litgpt.data import SFTDataset, LazySFTDataset, get_sft_collate_fn
from litgpt.prompts import PromptStyle


# Mock data source for LazySFTDataset
class MockDataSource:
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


@pytest.mark.parametrize("dataset_cls", [SFTDataset, LazySFTDataset])
@pytest.mark.parametrize("mask_prompt", [True, False])
@pytest.mark.parametrize("ignore_index", [-1, -100])
@pytest.mark.parametrize("max_seq_length", [1000, 5, -1])
def test_sft_datasets(dataset_cls, max_seq_length, ignore_index, mask_prompt, mock_tokenizer):
    class Style(PromptStyle):
        def apply(self, prompt, **kwargs):
            return f"In: {prompt} Out:"

    i = ignore_index
    data = [{"instruction": "Foo", "output": "Bar"}, {"instruction": "Boo", "output": "Ahh"}]

    if dataset_cls == LazySFTDataset:
        data_source = MockDataSource(data)
        dataset = dataset_cls(
            data_source=data_source,
            tokenizer=mock_tokenizer,
            prompt_style=Style(),
            mask_prompt=mask_prompt,
            ignore_index=ignore_index,
            max_seq_length=max_seq_length,
        )
    else:
        dataset = dataset_cls(
            data=data,
            tokenizer=mock_tokenizer,
            prompt_style=Style(),
            mask_prompt=mask_prompt,
            ignore_index=ignore_index,
            max_seq_length=max_seq_length,
        )
    assert len(dataset) == len(data)

    expected_input_ids = torch.tensor([73, 110, 58, 32, 70, 111, 111, 32, 79, 117, 116, 58, 66, 97, 114, 1])
    expected_labels = (
        torch.tensor([i, i, i, i, i, i, i, i, i, i, i, i, 66, 97, 114, 1]) if mask_prompt else expected_input_ids
    )

    if max_seq_length == -1:
        assert torch.equal(dataset[0]["input_ids"], expected_input_ids)
        assert torch.equal(dataset[0]["labels"], expected_labels)
    else:
        assert torch.equal(dataset[0]["input_ids"], expected_input_ids[:max_seq_length])
        assert torch.equal(dataset[0]["labels"], expected_labels[:max_seq_length])


def test_lazy_sft_dataset__returns_tokenized_inputs_from_data_source(mock_tokenizer):
    class Style(PromptStyle):
        def apply(self, prompt, **kwargs):
            return f"Q: {prompt} A:"

    data = [{"instruction": "Question", "output": "Answer"}]
    data_source = MockDataSource(data)

    dataset = LazySFTDataset(
        data_source=data_source,
        tokenizer=mock_tokenizer,
        prompt_style=Style(),
    )

    assert len(dataset) == len(data)
    sample = dataset[0]
    expected_input_ids = mock_tokenizer.encode("Q: Question A:Answer", eos=True)
    assert torch.equal(sample["input_ids"], expected_input_ids)
    # By default, mask_prompt=True and ignore_index=-100
    expected_labels = expected_input_ids.clone()
    expected_labels[: len(mock_tokenizer.encode("Q: Question A:"))] = -100
    assert torch.equal(sample["labels"], expected_labels)


@pytest.mark.parametrize("ignore_index", [-1, -100])
@pytest.mark.parametrize("pad_id", [0, 100])
def test_sft_collate_fn_padding(pad_id, ignore_index):
    collate = get_sft_collate_fn(pad_id=pad_id, ignore_index=ignore_index)
    samples = [
        {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([10, 20, 30])},
        {"input_ids": torch.tensor([4, 5, 6, 7, 8]), "labels": torch.tensor([40, 50, 60, 70, 80])},
    ]
    expected = {
        "input_ids": torch.tensor([[1, 2, 3, pad_id, pad_id], [4, 5, 6, 7, 8]]),
        "labels": torch.tensor([[10, 20, 30, ignore_index, ignore_index], [40, 50, 60, 70, 80]]),
    }
    batch = collate(samples)
    assert all(torch.equal(batch[k], expected[k]) for k in ("input_ids", "labels"))


def test_sft_collate_fn_truncation():
    collate = get_sft_collate_fn(max_seq_length=2)
    samples = [
        {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([10, 20, 30])},
        {"input_ids": torch.tensor([4, 5, 6, 7, 8]), "labels": torch.tensor([40, 50, 60, 70, 80])},
    ]
    expected = {"input_ids": torch.tensor([[1, 2], [4, 5]]), "labels": torch.tensor([[10, 20], [40, 50]])}
    batch = collate(samples)
    assert all(torch.equal(batch[k], expected[k]) for k in ("input_ids", "labels"))
