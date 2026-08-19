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

from dataclasses import dataclass
from itertools import chain
from typing import Any

from .processor_utils import DatasetProcessor


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
        # it does not. Removing the erosion entirely needs `neat_packing` (block-diagonal attention,
        # currently SFT-only) or `packing: false`.
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

        if not self.data_args.packing:
            if getattr(self.tokenizer, "add_bos_token", False):
                text_examples = [self.tokenizer.bos_token + example for example in text_examples]

            result = self.tokenizer(
                text_examples, add_special_tokens=False, truncation=True, max_length=self.data_args.cutoff_len
            )
        else:
            tokenized_examples = self.tokenizer(text_examples, add_special_tokens=False)
            concatenated_examples = {k: list(chain(*tokenized_examples[k])) for k in tokenized_examples.keys()}
            total_length = len(concatenated_examples[list(concatenated_examples.keys())[0]])
            block_size = self.data_args.cutoff_len
            total_length = max((total_length // block_size) * block_size, block_size)
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            if getattr(self.tokenizer, "add_bos_token", False):
                for i in range(len(result["input_ids"])):
                    result["input_ids"][i][0] = self.tokenizer.bos_token_id

        return result

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
