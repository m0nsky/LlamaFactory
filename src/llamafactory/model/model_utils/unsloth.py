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

from typing import TYPE_CHECKING, Any, Optional

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from ...extras import logging
from ...extras.constants import AttentionFunction
from ...extras.misc import get_current_device


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

    from ...hparams import FinetuningArguments, ModelArguments


logger = logging.get_logger(__name__)

# The fields `tokenizers.AddedToken` actually accepts. A dict carrying anything else is not a
# serialized token, no matter what it holds under "content".
_ADDED_TOKEN_FIELDS = frozenset({"content", "single_word", "lstrip", "rstrip", "normalized", "special"})

# Captured at import time. This module is pulled in by llamafactory.model.loader, long before any
# `from unsloth import ...` executes, so this is the stock transformers implementation. Fetched
# defensively: if a future transformers drops the method there is nothing to repair, and failing
# here would break importing LlamaFactory at all.
_STOCK_CONVERT_ADDED_TOKENS = getattr(PreTrainedTokenizerBase, "convert_added_tokens", None)


def _fix_unsloth_convert_added_tokens() -> None:
    r"""Reinstall unsloth's `convert_added_tokens` patch with a strict predicate.

    unsloth_zoo (temporary_patches/misc.py) replaces `PreTrainedTokenizerBase.convert_added_tokens`
    with a version that turns *any* dict holding a string "content" key into an `AddedToken`:

        if isinstance(obj, dict) and "content" in obj and "__type" not in obj and isinstance(obj["content"], str):
            return AddedToken(**obj)

    That method is bidirectional. `save_pretrained` calls it with `save=True` precisely to turn
    `AddedToken` objects *into* plain dicts so `json.dumps` can encode them; the patch performs the
    opposite conversion on the way out and ignores `save` entirely. Any unrelated nested dict that
    happens to carry a string "content" therefore becomes an `AddedToken` during the checkpoint
    write, and the save dies with `TypeError: Object of type AddedToken is not JSON serializable`.

    google/gemma-4-E4B-it trips this as of revision ee0ef602 (2026-07-20), which added
    `response_template` to tokenizer_config.json: its `fields.content`, `fields.thinking` and
    `fields.tool_calls` entries each carry a string "content". LlamaFactory loads the tokenizer
    before unsloth is imported, so the patch is only live on the way out - training runs fine and
    then dies on the first checkpoint.

    We reinstall it matching only dicts whose keys are all genuine `AddedToken` fields. That keeps
    unsloth's intent (special-token dicts that lack `"__type": "AddedToken"`) while leaving foreign
    metadata alone in both directions. Note that merely adding `and not save` is *not* enough: if
    the tokenizer is ever loaded while the patch is already live, the round trip rewrites
    `response_template.fields.content` to `{"__type": "AddedToken", ...}` and silently drops the
    real contents.

    TODO: delete this once unsloth_zoo's own predicate respects `save` or narrows its match.
    Present and byte-identical in unsloth_zoo 2026.4.9 (installed), 2026.7.6 (latest release) and
    GitHub main/nightly as of 2026-07-27.
    """
    from transformers.tokenization_utils_base import AddedToken

    if _STOCK_CONVERT_ADDED_TOKENS is None:
        return

    if hasattr(_STOCK_CONVERT_ADDED_TOKENS, "_unsloth_patched"):
        logger.warning_rank0(
            "Unsloth was imported before LlamaFactory, cannot recover the original "
            "`convert_added_tokens`. Saving checkpoints may fail."
        )
        return

    def convert_added_tokens(cls, obj: Any, save: bool = False, add_type_field: bool = True) -> Any:
        if (
            isinstance(obj, dict)
            and "content" in obj
            and "__type" not in obj
            and isinstance(obj["content"], str)
            and _ADDED_TOKEN_FIELDS.issuperset(obj)
        ):
            return AddedToken(**obj)

        # Recurses through `cls.convert_added_tokens`, i.e. back into this function.
        return _STOCK_CONVERT_ADDED_TOKENS.__func__(cls, obj, save=save, add_type_field=add_type_field)

    # Stops unsloth reinstalling its own version if the patches are ever applied again.
    convert_added_tokens._unsloth_patched = True
    PreTrainedTokenizerBase.convert_added_tokens = classmethod(convert_added_tokens)


def _get_unsloth_kwargs(
    config: "PretrainedConfig",
    model_name_or_path: str,
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
) -> dict[str, Any]:
    kwargs = {
        "model_name": model_name_or_path,
        "max_seq_length": model_args.model_max_length or 4096,
        "dtype": model_args.compute_dtype,
        "load_in_4bit": model_args.quantization_bit == 4,
        "token": model_args.hf_hub_token,
        "full_finetuning": finetuning_args.finetuning_type == "full",
        "device_map": {"": get_current_device()},
        "rope_scaling": getattr(config, "rope_scaling", None),
        "fix_tokenizer": False,
        "trust_remote_code": model_args.trust_remote_code,
        "use_gradient_checkpointing": "unsloth",
    }

    # Pass attention implementation through to Unsloth via **kwargs. Unsloth
    # forwards this to the underlying HF from_pretrained call. Needed because
    # Unsloth re-reads config.json from disk and does not honor the config
    # object we patched in configure_attn_implementation().
    if model_args.flash_attn == AttentionFunction.SDPA:
        kwargs["attn_implementation"] = "sdpa"
    elif model_args.flash_attn == AttentionFunction.FA2:
        kwargs["attn_implementation"] = "flash_attention_2"
    elif model_args.flash_attn == AttentionFunction.DISABLED:
        kwargs["attn_implementation"] = "eager"
    # AUTO: don't set; let Unsloth decide based on its own default.

    return kwargs


def load_unsloth_pretrained_model(
    config: "PretrainedConfig", model_args: "ModelArguments", finetuning_args: "FinetuningArguments"
) -> Optional["PreTrainedModel"]:
    r"""Optionally load pretrained model with unsloth. Used in training."""
    from unsloth import FastLanguageModel  # type: ignore

    _fix_unsloth_convert_added_tokens()
    unsloth_kwargs = _get_unsloth_kwargs(config, model_args.model_name_or_path, model_args, finetuning_args)
    try:
        model, _ = FastLanguageModel.from_pretrained(**unsloth_kwargs)
    except NotImplementedError:
        logger.warning_rank0("Unsloth does not support model type {}.".format(getattr(config, "model_type", None)))
        model = None
        model_args.use_unsloth = False

    return model


def get_unsloth_peft_model(
    model: "PreTrainedModel", model_args: "ModelArguments", peft_kwargs: dict[str, Any]
) -> "PreTrainedModel":
    r"""Get the peft model for the pretrained model with unsloth. Used in training."""
    from unsloth import FastLanguageModel  # type: ignore

    _fix_unsloth_convert_added_tokens()
    unsloth_peft_kwargs = {
        "model": model,
        "max_seq_length": model_args.model_max_length,
        "use_gradient_checkpointing": "unsloth",
    }
    return FastLanguageModel.get_peft_model(**peft_kwargs, **unsloth_peft_kwargs)


def load_unsloth_peft_model(
    config: "PretrainedConfig",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool,
) -> "PreTrainedModel":
    r"""Load peft model with unsloth. Used in both training and inference."""
    from unsloth import FastLanguageModel  # type: ignore

    _fix_unsloth_convert_added_tokens()
    unsloth_kwargs = _get_unsloth_kwargs(config, model_args.adapter_name_or_path[0], model_args, finetuning_args)
    try:
        if not is_trainable:
            unsloth_kwargs["use_gradient_checkpointing"] = False

        model, _ = FastLanguageModel.from_pretrained(**unsloth_kwargs)
    except NotImplementedError:
        raise ValueError("Unsloth does not support model type {}.".format(getattr(config, "model_type", None)))

    if not is_trainable:
        FastLanguageModel.for_inference(model)

    return model
