# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import json

import pytest

from litgpt.data import JSON
from litgpt.prompts import PromptStyle


@pytest.mark.parametrize("as_jsonl", [False, True])
@pytest.mark.parametrize("use_lazy_loading", [False, True])
def test_json__no_shuffle_contains_elements_in_order(as_jsonl, use_lazy_loading, tmp_path, mock_tokenizer):
    class Style(PromptStyle):
        def apply(self, prompt, **kwargs):
            return f"X: {prompt} {kwargs['input']} Y:"

    json_path = tmp_path / ("data.jsonl" if as_jsonl else "data.json")
    mock_data = [
        {"instruction": "Add", "input": "2+2", "output": "4"},
        {"instruction": "Subtract", "input": "5-3", "output": "2"},
        {"instruction": "Multiply", "input": "6*4", "output": "24"},
        {"instruction": "Divide", "input": "10/2", "output": "5"},
        {"instruction": "Exponentiate", "input": "2^3", "output": "8"},
        {"instruction": "Square root", "input": "√9", "output": "3"},
    ]

    with open(json_path, "w", encoding="utf-8") as fp:
        if as_jsonl:
            for line in mock_data:
                json.dump(line, fp)
                fp.write("\n")
        else:
            json.dump(mock_data, fp)

    data = JSON(
        json_path,
        val_split_fraction=0.5,
        prompt_style=Style(),
        num_workers=0,
        use_lazy_loading=use_lazy_loading,
        shuffle_training_data=False,  # Ensure no shuffling
        seed=42,
    )
    data.connect(tokenizer=mock_tokenizer, batch_size=2)
    data.prepare_data()
    data.setup()

    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()

    train_data = list(train_dataloader)
    val_data = list(val_dataloader)

    actual_train_prompts = [
        mock_tokenizer.decode(batch["input_ids"][i], exclude_special_tokens=True)
        for batch in train_data
        for i in range(batch["input_ids"].size(0))
    ]

    # Update expected training prompts
    expected_train_prompts = [
        "X: Add 2+2 Y:4",
        "X: Subtract 5-3 Y:2",
        "X: Multiply 6*4 Y:24",
    ]
    assert actual_train_prompts == expected_train_prompts

    actual_val_prompts = [
        mock_tokenizer.decode(batch["input_ids"][i], exclude_special_tokens=True)
        for batch in val_data
        for i in range(batch["input_ids"].size(0))
    ]

    # Update expected validation prompts
    expected_val_prompts = [
        "X: Divide 10/2 Y:5",
        "X: Exponentiate 2^3 Y:8",
        "X: Square root √9 Y:3",
    ]
    assert actual_val_prompts == expected_val_prompts

@pytest.mark.parametrize("as_jsonl", [False, True])
@pytest.mark.parametrize("use_lazy_loading", [False, True])
def test_json__with_shuffle_contains_expected_elements(as_jsonl, use_lazy_loading, tmp_path, mock_tokenizer):
    class Style(PromptStyle):
        def apply(self, prompt, **kwargs):
            return f"X: {prompt} {kwargs['input']} Y:"

    json_path = tmp_path / ("data.jsonl" if as_jsonl else "data.json")
    mock_data = [
        {"instruction": "Add", "input": "2+2", "output": "4"},
        {"instruction": "Subtract", "input": "5-3", "output": "2"},
        {"instruction": "Multiply", "input": "6*4", "output": "24"},
        {"instruction": "Divide", "input": "10/2", "output": "5"},
        {"instruction": "Exponentiate", "input": "2^3", "output": "8"},
        {"instruction": "Square root", "input": "√9", "output": "3"},
    ]

    with open(json_path, "w", encoding="utf-8") as fp:
        if as_jsonl:
            for line in mock_data:
                json.dump(line, fp)
                fp.write("\n")
        else:
            json.dump(mock_data, fp)

    data = JSON(
        json_path,
        val_split_fraction=0.5,
        prompt_style=Style(),
        num_workers=0,
        use_lazy_loading=use_lazy_loading,
        shuffle_training_data=True,  # Shuffle enabled (default)
        seed=42,
    )
    data.connect(tokenizer=mock_tokenizer, batch_size=2)
    data.prepare_data()  # does nothing
    data.setup()

    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()

    assert len(train_dataloader) == 2
    assert len(val_dataloader) == 2

    train_data = list(train_dataloader)
    val_data = list(val_dataloader)

    actual_train_prompts = set()
    for batch in train_data:
        for input_ids in batch["input_ids"]:
            decoded = mock_tokenizer.decode(input_ids, exclude_special_tokens=True)
            actual_train_prompts.add(decoded)

    expected_train_prompts = {
        "X: Add 2+2 Y:4",
        "X: Divide 10/2 Y:5",
        "X: Multiply 6*4 Y:24",
    }
    actual_train_prompts = {
        mock_tokenizer.decode(batch["input_ids"][i], exclude_special_tokens=True)
        for batch in train_data
        for i in range(batch["input_ids"].size(0))
    }
    assert actual_train_prompts == expected_train_prompts

    expected_val_prompts = {
        "X: Subtract 5-3 Y:2",
        "X: Exponentiate 2^3 Y:8",
        "X: Square root √9 Y:3",
    }
    actual_val_prompts = {
        mock_tokenizer.decode(batch["input_ids"][i], exclude_special_tokens=True)
        for batch in val_data
        for i in range(batch["input_ids"].size(0))
    }
    assert actual_val_prompts == expected_val_prompts


def test_json_input_validation(tmp_path):
    with pytest.raises(FileNotFoundError, match="The `json_path` must be a file or a directory"):
        JSON(tmp_path / "not exist")

    with pytest.raises(ValueError, match="`val_split_fraction` should not be set"):
        JSON(tmp_path, val_split_fraction=0.5)

    data = JSON(tmp_path)
    data.prepare_data()  # does nothing

    # Empty directory
    with pytest.raises(FileNotFoundError, match="must be a file or a directory containing"):
        data.setup()

    # Only train.json exists
    (tmp_path / "train.json").touch()
    with pytest.raises(FileNotFoundError, match="must be a file or a directory containing"):
        data.setup()

    with pytest.raises(ValueError, match="you must set `val_split_fraction` to a value between 0 and 1"):
        JSON(tmp_path / "train.json", val_split_fraction=None)


@pytest.mark.parametrize("as_jsonl", [False, True])
@pytest.mark.parametrize("use_lazy_loading", [False, True])
def test_json_with_splits(as_jsonl, use_lazy_loading, tmp_path, mock_tokenizer):
    mock_train_data = [
        {"instruction": "Add", "input": "2+2", "output": "4"},
        {"instruction": "Subtract", "input": "5-3", "output": "2"},
        {"instruction": "Exponentiate", "input": "2^3", "output": "8"},
    ]
    mock_test_data = [
        {"instruction": "Multiply", "input": "6*4", "output": "24"},
        {"instruction": "Divide", "input": "10/2", "output": "5"},
    ]

    train_file = tmp_path / ("train.jsonl" if as_jsonl else "train.json")
    val_file = tmp_path / ("val.jsonl" if as_jsonl else "val.json")

    with open(train_file, "w", encoding="utf-8") as fp:
        if as_jsonl:
            for line in mock_train_data:
                json.dump(line, fp)
                fp.write("\n")
        else:
            json.dump(mock_train_data, fp)
    with open(val_file, "w", encoding="utf-8") as fp:
        if as_jsonl:
            for line in mock_test_data:
                json.dump(line, fp)
                fp.write("\n")
        else:
            json.dump(mock_test_data, fp)

    data = JSON(tmp_path, num_workers=0, use_lazy_loading=use_lazy_loading)
    data.connect(tokenizer=mock_tokenizer, batch_size=2)
    data.prepare_data()  # does nothing
    data.setup()

    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()

    assert len(train_dataloader) == 2
    assert len(val_dataloader) == 1

    train_data = list(train_dataloader)
    val_data = list(val_dataloader)

    assert train_data[0]["input_ids"].size(0) == 2
    assert train_data[1]["input_ids"].size(0) == 1
    assert val_data[0]["input_ids"].size(0) == 2