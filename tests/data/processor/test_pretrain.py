# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import pytest
from transformers import AutoTokenizer

from llamafactory.data.processor.pretrain import PretrainDatasetProcessor
from llamafactory.data.template import get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.hparams import DataArguments


TINY_LLAMA3 = os.getenv("TINY_LLAMA3", "llamafactory/tiny-random-Llama-3")

DOCUMENTS = [
    "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron.",
    "Pi rho sigma tau upsilon phi chi psi omega alpha beta gamma delta epsilon zeta.",
    "Eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi.",
    "Chi psi omega and then some additional trailing words to fill out the block.",
    "One two three four five six seven eight nine ten eleven twelve thirteen.",
]

EXAMPLES = {"_prompt": [[{"role": "user", "content": document}] for document in DOCUMENTS]}


def _build_processor(template_name: str, cutoff_len: int, packing: bool) -> PretrainDatasetProcessor:
    tokenizer = AutoTokenizer.from_pretrained(TINY_LLAMA3)
    data_args = DataArguments(template=template_name, cutoff_len=cutoff_len, packing=packing)
    data_args.cutoff_len = cutoff_len  # `__post_init__` decrements it when packing, undo for clarity
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    return PretrainDatasetProcessor(template=template, tokenizer=tokenizer, processor=None, data_args=data_args)


def _prefix_ids(processor: PretrainDatasetProcessor) -> list[int]:
    return processor.template._convert_elements_to_ids(processor.tokenizer, processor.template.format_prefix.apply())


@pytest.mark.runs_on(["cpu", "mps"])
def test_pretrain_packed_blocks_are_exactly_cutoff_len():
    r"""With more data than one block, every block is exactly `cutoff_len` long."""
    processor = _build_processor("llama3", 24, packing=True)
    result = processor.preprocess_dataset(EXAMPLES)

    assert len(result["input_ids"]) > 1, "the fixture should fill more than one block"
    for input_ids, attention_mask in zip(result["input_ids"], result["attention_mask"]):
        assert len(input_ids) == 24
        assert len(attention_mask) == 24


@pytest.mark.runs_on(["cpu", "mps"])
def test_pretrain_packing_emits_one_short_block_when_data_is_smaller_than_a_block():
    r"""Too little data for a full block yields a single short block, which the collator pads."""
    processor = _build_processor("llama3", 512, packing=True)
    result = processor.preprocess_dataset(EXAMPLES)

    assert len(result["input_ids"]) == 1
    assert 0 < len(result["input_ids"][0]) < 512
    assert len(result["attention_mask"][0]) == len(result["input_ids"][0])


@pytest.mark.runs_on(["cpu", "mps"])
def test_pretrain_packed_blocks_start_with_template_prefix():
    processor = _build_processor("llama3", 24, packing=True)
    prefix_ids = _prefix_ids(processor)
    assert prefix_ids, "llama3 declares a sequence-start prefix"

    result = processor.preprocess_dataset(EXAMPLES)
    for input_ids in result["input_ids"]:
        assert input_ids[: len(prefix_ids)] == prefix_ids


@pytest.mark.runs_on(["cpu", "mps"])
def test_pretrain_packing_prepends_prefix_without_dropping_content():
    r"""The prefix must be inserted, not written over the first token of the block."""
    processor = _build_processor("llama3", 24, packing=True)
    prefix_ids = _prefix_ids(processor)
    result = processor.preprocess_dataset(EXAMPLES)

    # rebuild the document stream the processor concatenates, and check it survives verbatim.
    # llama3 separates documents with `<|end_of_text|>`, not the turn terminator that
    # `replace_eos` leaves in `tokenizer.eos_token`.
    eos_token = "<|end_of_text|>"
    stream = []
    for document in DOCUMENTS:
        stream += processor.tokenizer(document + eos_token, add_special_tokens=False)["input_ids"]

    content_size = 24 - len(prefix_ids)
    for block_index, input_ids in enumerate(result["input_ids"]):
        expected = stream[block_index * content_size : (block_index + 1) * content_size]
        assert input_ids[len(prefix_ids) :] == expected


@pytest.mark.runs_on(["cpu", "mps"])
def test_pretrain_unpacked_documents_carry_prefix_and_eos():
    processor = _build_processor("llama3", 64, packing=False)
    prefix_ids = _prefix_ids(processor)
    result = processor.preprocess_dataset(EXAMPLES)

    assert len(result["input_ids"]) == len(DOCUMENTS)
    for input_ids, document in zip(result["input_ids"], DOCUMENTS):
        assert input_ids[: len(prefix_ids)] == prefix_ids
        body = processor.tokenizer(document, add_special_tokens=False)["input_ids"]
        assert input_ids[len(prefix_ids) : len(prefix_ids) + len(body)] == body


@pytest.mark.runs_on(["cpu", "mps"])
@pytest.mark.parametrize("packing", [True, False])
def test_pretrain_never_doubles_the_prefix(packing: bool):
    r"""`add_special_tokens=False` keeps the tokenizer from adding a second bos of its own."""
    processor = _build_processor("llama3", 24, packing=packing)
    prefix_ids = _prefix_ids(processor)
    result = processor.preprocess_dataset(EXAMPLES)

    for input_ids in result["input_ids"]:
        assert input_ids[len(prefix_ids) : 2 * len(prefix_ids)] != prefix_ids


@pytest.mark.runs_on(["cpu", "mps"])
@pytest.mark.parametrize("packing", [True, False])
def test_pretrain_templates_without_prefix_get_no_prefix(packing: bool):
    processor = _build_processor("qwen", 24, packing=packing)
    assert _prefix_ids(processor) == []

    result = processor.preprocess_dataset(EXAMPLES)
    first_content_id = processor.tokenizer(DOCUMENTS[0], add_special_tokens=False)["input_ids"][0]
    assert result["input_ids"][0][0] == first_content_id


@pytest.mark.runs_on(["cpu", "mps"])
@pytest.mark.parametrize(
    ("stage", "packing", "neat_packing", "expected_cutoff_len"),
    [
        # PT slices at block_size == cutoff_len, so it must not be decremented.
        ("pt", None, False, 4096),
        ("pt", True, False, 4096),
        ("pt", False, False, 4096),
        # SFT packing emits cutoff_len + 1 tokens, so the decrement is what makes the packed
        # sequence come out at the length that was actually requested.
        ("sft", True, False, 4095),
        ("sft", None, True, 4095),
        ("sft", None, False, 4096),
        ("sft", False, False, 4096),
    ],
)
def test_packing_cutoff_len_is_stage_aware(
    stage: str, packing: bool | None, neat_packing: bool, expected_cutoff_len: int
):
    from llamafactory.hparams import get_train_args

    args = {
        "model_name_or_path": TINY_LLAMA3,
        "stage": stage,
        "do_train": True,
        "finetuning_type": "lora",
        "dataset": "alpaca_en_demo",
        "template": "llama3",
        "cutoff_len": 4096,
        "output_dir": "dummy_dir",
        "report_to": "none",
        "neat_packing": neat_packing,
    }
    if packing is not None:
        args["packing"] = packing

    _, data_args, _, _, _ = get_train_args(args)
    assert data_args.cutoff_len == expected_cutoff_len


def _neat_processor(cutoff_len: int) -> PretrainDatasetProcessor:
    tokenizer = AutoTokenizer.from_pretrained(TINY_LLAMA3)
    data_args = DataArguments(template="llama3", cutoff_len=cutoff_len, neat_packing=True)
    data_args.cutoff_len = cutoff_len
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    return PretrainDatasetProcessor(template=template, tokenizer=tokenizer, processor=None, data_args=data_args)


@pytest.mark.runs_on(["cpu", "mps"])
def test_neat_packing_emits_no_attention_mask():
    r"""Transformers only detects the packed format when the mask is absent."""
    processor = _neat_processor(64)
    result = processor.preprocess_dataset(EXAMPLES)

    assert set(result) == {"input_ids", "position_ids", "labels"}


@pytest.mark.runs_on(["cpu", "mps"])
def test_neat_packing_blocks_are_exactly_cutoff_len():
    processor = _neat_processor(64)
    result = processor.preprocess_dataset(EXAMPLES)

    for key in ("input_ids", "position_ids", "labels"):
        for row in result[key]:
            assert len(row) == 64


@pytest.mark.runs_on(["cpu", "mps"])
def test_neat_packing_restarts_position_ids_per_document():
    processor = _neat_processor(64)
    prefix_ids = _prefix_ids(processor)
    result = processor.preprocess_dataset(EXAMPLES)

    for input_ids, position_ids in zip(result["input_ids"], result["position_ids"]):
        starts = [i for i, position in enumerate(position_ids) if position == 0]
        assert starts and starts[0] == 0
        for lo, hi in zip(starts, starts[1:] + [len(position_ids)]):
            # each run is a contiguous 0..n-1, which is what transformers keys the block mask on
            assert position_ids[lo:hi] == list(range(hi - lo))
            if input_ids[lo] != processor.tokenizer.pad_token_id:
                assert input_ids[lo : lo + len(prefix_ids)] == prefix_ids


@pytest.mark.runs_on(["cpu", "mps"])
def test_neat_packing_masks_document_starts_and_padding():
    processor = _neat_processor(64)
    result = processor.preprocess_dataset(EXAMPLES)
    pad_token_id = processor.tokenizer.pad_token_id

    for input_ids, position_ids, labels in zip(result["input_ids"], result["position_ids"], result["labels"]):
        for i, (input_id, position, label) in enumerate(zip(input_ids, position_ids, labels)):
            if position == 0:
                # would otherwise be predicted from the previous document
                assert label == IGNORE_INDEX
            elif input_id == pad_token_id:
                assert label == IGNORE_INDEX
            else:
                assert label == input_id


@pytest.mark.runs_on(["cpu", "mps"])
def test_neat_packing_keeps_every_document():
    r"""Unlike plain packing it drops no remainder, and unlike SFT packing it drops nothing lengthy."""
    processor = _neat_processor(64)
    result = processor.preprocess_dataset(EXAMPLES)
    pad_token_id = processor.tokenizer.pad_token_id

    emitted = [i for row in result["input_ids"] for i in row if i != pad_token_id]
    expected = 0
    for document in DOCUMENTS:
        expected += len(processor.tokenizer(document + "<|end_of_text|>", add_special_tokens=False)["input_ids"])
    expected += len(_prefix_ids(processor)) * len(DOCUMENTS)
    assert len(emitted) == expected


@pytest.mark.runs_on(["cpu", "mps"])
def test_neat_packing_splits_documents_longer_than_a_block():
    r"""A long document is split across blocks instead of being dropped or truncated."""
    cutoff_len = 16
    processor = _neat_processor(cutoff_len)
    prefix_ids = _prefix_ids(processor)
    long_document = " ".join(f"word{i}" for i in range(200))
    result = processor.preprocess_dataset({"_prompt": [[{"role": "user", "content": long_document}]]})
    pad_token_id = processor.tokenizer.pad_token_id

    body = processor.tokenizer(long_document + "<|end_of_text|>", add_special_tokens=False)["input_ids"]
    content_size = cutoff_len - len(prefix_ids)
    expected_pieces = -(-len(body) // content_size)

    starts = [
        (row, i)
        for row in result["input_ids"]
        for i, token in enumerate(row)
        if token != pad_token_id and row[i : i + len(prefix_ids)] == prefix_ids
    ]
    assert len(starts) >= expected_pieces
    emitted = [i for row in result["input_ids"] for i in row if i != pad_token_id]
    assert len(emitted) == len(body) + len(prefix_ids) * expected_pieces
