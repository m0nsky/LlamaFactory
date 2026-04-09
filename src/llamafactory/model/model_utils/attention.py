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

from typing import TYPE_CHECKING

from ...extras import logging
from ...extras.constants import AttentionFunction
from ...extras.packages import is_torch_version_greater_than


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

    from ...hparams import ModelArguments


logger = logging.get_logger(__name__)


def configure_attn_implementation(config: "PretrainedConfig", model_args: "ModelArguments") -> None:
    from transformers.utils import is_flash_attn_2_available

    if getattr(config, "model_type", None) == "gpt_oss":
        from transformers.integrations.hub_kernels import load_and_register_kernel

        flash_attn3_kernel = "kernels-community/vllm-flash-attn3"
        load_and_register_kernel(flash_attn3_kernel)
        setattr(config, "_attn_implementation", flash_attn3_kernel)
        setattr(config, "_attn_implementation_internal", flash_attn3_kernel)
        model_args.flash_attn = AttentionFunction.FA3

        logger.info_rank0("Using FlashAttention-3 with attention sink for the gpt-oss model.")
        return

    if getattr(config, "model_type", None) == "gemma2":
        if model_args.flash_attn == AttentionFunction.AUTO or model_args.flash_attn == AttentionFunction.FA2:
            if is_flash_attn_2_available():
                if model_args.flash_attn != AttentionFunction.FA2:
                    logger.warning_rank0("Gemma 2 should use flash attention 2, change `flash_attn` to fa2.")
                    model_args.flash_attn = AttentionFunction.FA2
            else:
                logger.warning_rank0("FlashAttention-2 is not installed, use eager attention.")
                model_args.flash_attn = AttentionFunction.DISABLED
        elif model_args.flash_attn == AttentionFunction.SDPA:
            logger.warning_rank0(
                "Gemma-2 should use soft-capping attention, while the SDPA attention does not support it."
            )

    if getattr(config, "model_type", None) == "gemma4":
        # Gemma 4's full_attention layers use global_head_dim=512, which exceeds
        # FlashAttention-2's 256 head_dim limit. Force SDPA for auto/fa2 requests.
        if model_args.flash_attn in (AttentionFunction.AUTO, AttentionFunction.FA2):
            if model_args.flash_attn == AttentionFunction.FA2:
                logger.warning_rank0(
                    "Gemma 4 has heterogeneous attention head dimensions (global_head_dim=512) "
                    "that exceed FlashAttention-2's 256 limit. Forcing `flash_attn: sdpa`."
                )
            else:
                logger.info_rank0("Gemma 4 auto attention: using SDPA (FA2 unsupported for head_dim > 256).")
            model_args.flash_attn = AttentionFunction.SDPA

    if getattr(config, "model_type", None) in ["youtu", "youtu_vl"]:
        if model_args.flash_attn in (AttentionFunction.AUTO, AttentionFunction.SDPA):
            logger.warning_rank0("Youtu-VL does not support SDPA, forcing eager attention.")
            model_args.flash_attn = AttentionFunction.DISABLED

    if model_args.flash_attn == AttentionFunction.AUTO:
        return

    elif model_args.flash_attn == AttentionFunction.DISABLED:
        requested_attn_implementation = "eager"

    elif model_args.flash_attn == AttentionFunction.SDPA:
        if not is_torch_version_greater_than("2.1.1"):
            logger.warning_rank0("torch>=2.1.1 is required for SDPA attention.")
            return

        requested_attn_implementation = "sdpa"
    elif model_args.flash_attn == AttentionFunction.FA2:
        from transformers import is_torch_npu_available

        if not (is_flash_attn_2_available() or is_torch_npu_available()):
            logger.warning_rank0("FlashAttention-2 is not installed.")
            return

        requested_attn_implementation = "flash_attention_2"
    else:
        raise NotImplementedError(f"Unknown attention type: {model_args.flash_attn}")

    if getattr(config, "model_type", None) == "internlm2":  # special case for custom models
        setattr(config, "attn_implementation", requested_attn_implementation)
    elif getattr(config, "model_type", None) == "kimi_vl":
        setattr(config.vision_config, "_attn_implementation", requested_attn_implementation)
        setattr(config.text_config, "_attn_implementation", requested_attn_implementation)
    elif getattr(config, "model_type", None) == "youtu_vl":
        setattr(config, "attn_implementation", requested_attn_implementation)
        setattr(config, "_attn_implementation", requested_attn_implementation)
        if hasattr(config, "vision_config"):
            setattr(config.vision_config, "_attn_implementation", requested_attn_implementation)
        if hasattr(config, "text_config"):
            setattr(config.text_config, "_attn_implementation", requested_attn_implementation)
    elif getattr(config, "model_type", None) == "gemma4":
        # Gemma 4 is a composite model with nested text/vision/audio configs.
        # The text attention reads _attn_implementation from text_config (not the
        # outer config), so we propagate to all existing sub-configs.
        setattr(config, "_attn_implementation", requested_attn_implementation)
        if getattr(config, "text_config", None) is not None:
            setattr(config.text_config, "_attn_implementation", requested_attn_implementation)
        if getattr(config, "vision_config", None) is not None:
            setattr(config.vision_config, "_attn_implementation", requested_attn_implementation)
        if getattr(config, "audio_config", None) is not None:
            setattr(config.audio_config, "_attn_implementation", requested_attn_implementation)
    else:
        setattr(config, "_attn_implementation", requested_attn_implementation)


def print_attn_implementation(config: "PretrainedConfig") -> None:
    if getattr(config, "model_type", None) == "internlm2":  # special case for custom models
        attn_implementation = getattr(config, "attn_implementation", None)
    else:
        attn_implementation = getattr(config, "_attn_implementation", None)

    if attn_implementation == "flash_attention_2":
        logger.info_rank0("Using FlashAttention-2 for faster training and inference.")
    elif attn_implementation == "sdpa":
        logger.info_rank0("Using torch SDPA for faster training and inference.")
    else:
        logger.info_rank0("Using vanilla attention implementation.")


def apply_attn_implementation_post_load(model: "PreTrainedModel", model_args: "ModelArguments") -> None:
    r"""Post-load fix: force _attn_implementation on the loaded model's nested configs.

    Needed because some loader paths (notably Unsloth's FastLanguageModel.from_pretrained)
    re-read the model config from disk and ignore the Python config object we patched in
    configure_attn_implementation(). For composite models like Gemma 4 whose attention
    layers read self.config._attn_implementation at runtime (where self.config is the
    text_config, not the outer config), this propagation is required or the attention
    dispatch will fall back to the transformers default (flash_attention_2) and crash
    on large head dimensions.
    """
    if model_args.flash_attn == AttentionFunction.AUTO:
        return
    elif model_args.flash_attn == AttentionFunction.DISABLED:
        impl = "eager"
    elif model_args.flash_attn == AttentionFunction.SDPA:
        impl = "sdpa"
    elif model_args.flash_attn == AttentionFunction.FA2:
        impl = "flash_attention_2"
    else:
        return

    cfg = model.config
    setattr(cfg, "_attn_implementation", impl)
    for sub_name in ("text_config", "vision_config", "audio_config"):
        sub = getattr(cfg, sub_name, None)
        if sub is not None:
            setattr(sub, "_attn_implementation", impl)

    logger.info_rank0(f"Post-load: forced _attn_implementation='{impl}' on model config and nested configs.")
