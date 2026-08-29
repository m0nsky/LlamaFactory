# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/language-modeling/run_clm.py
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

from collections import defaultdict
from dataclasses import dataclass
from itertools import chain
from typing import Any

from ...extras.constants import IGNORE_INDEX
from .processor_utils import DatasetProcessor, greedy_knapsack


@dataclass
class PretrainDatasetProcessor(DatasetProcessor):
    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # build grouped texts with format `X1 X2 X3 ...` if packing is enabled

        # This token separates documents in the pretraining stream, so it must be the model's real
        # end-of-sequence token — not the chat turn terminator.
        #
        # Templates with `replace_eos=True` overwrite `tokenizer.eos_token` with their first stop
        # word, so by the time we get here it is a conversational control token: `<end_of_turn>`
        # for gemma/gemma3, `<turn|>` for gemma4. With packing enabled, loss is computed across
        # the joins, which trains the model that its own stop token is followed by more text — and
        # a model trained that way does not stop when generating.
        #
        # llama3 already carries this exception. Gemma needs the same: Google's tokenizer_config
        # declares `eos_token: <eos>` and `eot_token: <turn|>` as separate fields, and its
        # chat_template.jinja uses the turn tokens only — `<eos>` never appears there.
        #
        # Note what this does and does not achieve. Both tokens terminate generation (gemma4's
        # generation_config lists eos_token_id [1, 106, 50]), so this does not change what stops
        # the model. With packing on, loss is still computed across the joins, so the model still
        # learns that *something* is followed by more text — this moves that erosion off `<turn|>`,
        # which the chat format requires the model to emit to end a turn, and onto `<eos>`, which
        # it does not. Removing the erosion entirely needs `neat_packing` (see `_neat_pack`) or
        # `packing: false`.
        #
        # 41 of ~120 templates set replace_eos=True, so this affects far more than gemma —
        # llama4 among them. Only llama3 and gemma are handled here.
        if self.data_args.template == "llama3":
            eos_token = "<|end_of_text|>"
        elif self.data_args.template is not None and self.data_args.template.startswith("gemma"):
            eos_token = "<eos>"
        else:
            eos_token = self.tokenizer.eos_token
        text_examples = [messages[0]["content"] + eos_token for messages in examples["_prompt"]]

        # PT blocks deliberately do NOT open with the template's sequence-start token (`<bos>`).
        #
        # That is not an oversight, and it is not what the model's own pretraining convention would
        # suggest — it is a measured result. Prefixing every document with `<bos>` does exactly what
        # you would expect at the token level: on gemma-4-E4B-it it takes a cold document from
        # ppl 32,686 -> 9.6, and a mid-document fragment (what a packed block looks like) from
        # 10,154 -> 83, still 171x better at tokens 32..96 in. Those tokens otherwise carry enormous
        # loss and dominate the gradient.
        #
        # But this stage continued-pretrains an *instruct* model that is then served as a chat
        # model, and there `<bos>` already means "a chat turn is starting" — every inference prompt
        # opens with it. Prefixing ~1,800 raw-document blocks with `<bos>` overwrites that
        # association with "a long document is starting", and the served model answers chat prompts
        # with rambling, non-terminating prose.
        #
        # Ablation (gemma-4-E4B-it, 48 held-out questions, greedy decoding, checkpoints matched on
        # epochs; figure is the share of answers containing an 8-gram repeated >3x):
        #
        #                     plain packing        neat packing
        #     ~2.2 epochs   no-BOS  21%             10%
        #                   BOS     29%             44%
        #     ~2.7 epochs   no-BOS  31%             21%
        #                   BOS     60%             40%
        #
        #   pooled BOS effect over the four matched pairs: +22.4pp +/- 9.1 (95% CI) -- significant.
        #   The untrained base model loops on 0/48. Train loss is identical with and without `<bos>`
        #   (0.232 vs 0.234 at step 90), so this is a conditioning effect, not overfitting.
        #
        # If you are continued-pretraining a *base* model rather than an instruct one, the trade
        # runs the other way and the prefix is worth restoring. The fix that should get both is to
        # wrap PT documents in the chat structure the model is actually served with, so `<bos>` is
        # followed by `<|turn>` as at inference -- untested here.
        prefix_ids: list[int] = []

        if self.data_args.packing and self.data_args.neat_packing:
            return self._neat_pack(text_examples, prefix_ids)

        if not self.data_args.packing:
            # Leave room for the prefix so a document still ends up exactly `cutoff_len` long.
            result = self.tokenizer(
                text_examples,
                add_special_tokens=False,
                truncation=True,
                max_length=self.data_args.cutoff_len - len(prefix_ids),
            )
            if prefix_ids:
                result["input_ids"] = [prefix_ids + input_ids for input_ids in result["input_ids"]]
                result["attention_mask"] = [
                    [1] * len(prefix_ids) + attention_mask for attention_mask in result["attention_mask"]
                ]
        else:
            tokenized_examples = self.tokenizer(text_examples, add_special_tokens=False)
            concatenated_ids = list(chain(*tokenized_examples["input_ids"]))
            block_size = self.data_args.cutoff_len
            # The prefix is part of the block, not an addition to it: each block is one prefix plus
            # `content_size` tokens of the document stream, so blocks stay exactly `block_size` long.
            content_size = block_size - len(prefix_ids)
            total_length = len(concatenated_ids)
            total_length = max((total_length // content_size) * content_size, content_size)
            input_ids = [
                prefix_ids + concatenated_ids[i : i + content_size] for i in range(0, total_length, content_size)
            ]
            result = {"input_ids": input_ids, "attention_mask": [[1] * len(ids) for ids in input_ids]}

        return result

    def _neat_pack(self, text_examples: list[str], prefix_ids: list[int]) -> dict[str, list[Any]]:
        r"""Pack whole documents into blocks that do not attend to or predict across each other.

        Plain packing concatenates every document into one stream and re-slices it at fixed
        offsets, so a block is generally the tail of one document glued to the head of the next.
        The model then attends across that join and is trained to predict the next document from
        the previous one.

        Here each block instead holds whole documents, and `position_ids` restart at 0 for each.
        Transformers derives a block-diagonal mask from exactly that signal
        (`masking_utils.find_packed_sequence_indices`), which is why this emits no `attention_mask`
        — supplying one suppresses the detection.

        This is deliberately not LlamaFactory's `prepare_4d_attention_mask` route that SFT uses.
        A pre-built 4D mask is returned as-is by transformers for *every* layer type, so on a
        hybrid-attention model it also overwrites the sliding-window mask: measured on gemma4
        (`sliding_window` 512) at sequence length 1200, the 4D route let a sliding layer attend to
        700 tokens instead of 512, while this route correctly gave 512 and still respected the
        document boundary.
        """
        block_size = self.data_args.cutoff_len
        content_size = block_size - len(prefix_ids)
        tokenized_examples = self.tokenizer(text_examples, add_special_tokens=False)["input_ids"]

        # Split documents longer than a block rather than dropping them (what SFT packing does) or
        # truncating them (what unpacked PT does). Every piece opens with the prefix, so a
        # continuation still starts the way the model expects a sequence to start.
        pieces: list[list[int]] = []
        for input_ids in tokenized_examples:
            for start in range(0, max(len(input_ids), 1), content_size):
                pieces.append(prefix_ids + input_ids[start : start + content_size])

        lengths = [len(piece) for piece in pieces]
        length2indexes = defaultdict(list)
        for index, length in enumerate(lengths):
            length2indexes[length].append(index)

        model_inputs = defaultdict(list)
        for knapsack in greedy_knapsack(lengths, block_size):
            packed_input_ids, packed_position_ids, packed_labels = [], [], []
            for length in knapsack:
                piece = pieces[length2indexes[length].pop()]
                packed_input_ids += piece
                packed_position_ids += list(range(len(piece)))
                # The first token of a document would otherwise be predicted from the last token of
                # the previous one — a target the block-diagonal mask has already cut the context
                # for. (For the first document in a block the label is dropped by the loss shift
                # anyway, so masking it changes nothing.)
                packed_labels += [IGNORE_INDEX] + piece[1:]

            pad_length = block_size - len(packed_input_ids)
            if pad_length > 0:
                # Padding restarts the positions too, so it forms its own sequence and real tokens
                # cannot attend to it. Its labels are ignored, so it contributes no loss.
                packed_input_ids += [self.tokenizer.pad_token_id] * pad_length
                packed_position_ids += list(range(pad_length))
                packed_labels += [IGNORE_INDEX] * pad_length

            model_inputs["input_ids"].append(packed_input_ids)
            model_inputs["position_ids"].append(packed_position_ids)
            model_inputs["labels"].append(packed_labels)

        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
