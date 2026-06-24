# coding=utf-8
# Copyright 2024 The Dream team, HKUNLP Group and the HuggingFace Inc. team. All rights reserved.
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

import warnings
import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import random
import math
import json
import pickle

import torch
import torch.distributions as dists
from torch.nn import functional as F
from transformers import __version__
from transformers.generation.configuration_utils import (
    GenerationConfig
)
from transformers.utils import (
    ModelOutput,
    is_torchdynamo_compiling,
    logging,
)

logger = logging.get_logger(__name__)


def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits

def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, margin_confidence=False, neg_entropy=False, log_step=-1):

    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
            # print("temper>0.0")
            # print(confidence)
        except:
            confidence, x0 = probs.max(dim=-1)
    else:
        # print("temper=0.0")
        confidence, x0 = probs.max(dim=-1)
    
    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        # Extract top1 and top2 probabilities
        top1_probs = sorted_probs[:, 0] 
        top2_probs = sorted_probs[:, 1] 
        # Calculate confidence as top1 - top2
        confidence = top1_probs - top2_probs 
    
    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)
        # print(confidence)
    return confidence, x0


@dataclass
class DreamModelOutput(ModelOutput):
    sequences: torch.LongTensor = None
    history: Optional[Tuple[torch.FloatTensor]] = None


class DreamGenerationConfig(GenerationConfig):
    def __init__(self, **kwargs):
        self.temperature: float = kwargs.pop("temperature", 0.0)
        self.top_p: Optional[float] = kwargs.pop("top_p", None)
        self.top_k: Optional[int] = kwargs.pop("top_k", None)
        self.max_length = kwargs.pop("max_length", 20)
        self.max_new_tokens = kwargs.pop("max_new_tokens", None)
        # diffusion specific params
        self.eps: float = kwargs.pop("eps", 1e-3)
        self.steps: int = kwargs.pop("steps", 512)
        self.alg: str = kwargs.pop("alg", 'origin')
        self.alg_temp: Optional[float] = kwargs.pop("alg_temp", None)
        self.threshold: Optional[float] = kwargs.pop("threshold", None)

        # Parameters that define the output variables of `generate`
        self.num_return_sequences: int = kwargs.pop("num_return_sequences", 1)
        self.return_dict_in_generate: bool = kwargs.pop("return_dict_in_generate", False)
        self.output_history: bool = kwargs.pop("output_history", False)
        self.print_all_token_records: bool = kwargs.pop("print_all_token_records", False)

        # Special tokens that can be used at generation time
        self.mask_token_id = kwargs.pop("mask_token_id", None)
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)

        # Wild card
        self.generation_kwargs = kwargs.pop("generation_kwargs", {})

        # The remaining attributes do not parametrize `.generate()`, but are informative and/or used by the hub
        # interface.
        self._from_model_config = kwargs.pop("_from_model_config", False)
        self._commit_hash = kwargs.pop("_commit_hash", None)
        self.transformers_version = kwargs.pop("transformers_version", __version__)

        # Additional attributes without default values
        if not self._from_model_config:
            # we don't want to copy values from the model config if we're initializing a `GenerationConfig` from a
            # model's default configuration file
            for key, value in kwargs.items():
                try:
                    setattr(self, key, value)
                except AttributeError as err:
                    logger.error(f"Can't set {key} with value {value} for {self}")
                    raise err

        # Validate the values of the attributes
        self.validate(is_init=True)

    def validate(self, is_init=False):
        pass

class DreamGenerationMixin:
    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""
        # Do not call torch.repeat_interleave if expand_size is 1 because it clones
        # the input tensor and thus requires more memory although no change is applied
        if expand_size == 1:
            return input_ids, attention_mask
        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)
        if attention_mask is not None:
            attention_mask = attention_mask.repeat_interleave(expand_size, dim=0)
        return input_ids, attention_mask

    def _validate_generated_length(self, generation_config, input_ids_length, has_default_max_length):
        """Performs validation related to the resulting generated length"""

        # Can't throw warnings/exceptions during compilation
        if is_torchdynamo_compiling():
            return

        # 1. Max length warnings related to poor parameterization
        if has_default_max_length and generation_config.max_new_tokens is None and generation_config.max_length == 20:
            # 20 is the default max_length of the generation config
            warnings.warn(
                f"Using the model-agnostic default `max_length` (={generation_config.max_length}) to control the "
                "generation length. We recommend setting `max_new_tokens` to control the maximum length of the "
                "generation.",
                UserWarning,
            )
        if input_ids_length >= generation_config.max_length:
            input_ids_string = "input_ids"
            raise ValueError(
                f"Input length of {input_ids_string} is {input_ids_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_length` or, better yet, setting `max_new_tokens`."
            )

    def _prepare_generated_length(
        self,
        generation_config,
        has_default_max_length,
        input_ids_length,
    ):
        """Prepared max and min length in generation configs to avoid clashes between similar attributes"""

        if generation_config.max_new_tokens is not None:
            if not has_default_max_length and generation_config.max_length is not None:
                logger.warning(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
                )
            generation_config.max_length = generation_config.max_new_tokens + input_ids_length

        elif has_default_max_length:
            if generation_config.max_length == DreamGenerationConfig().max_length:
                generation_config.max_length = generation_config.max_length + input_ids_length
                max_position_embeddings = getattr(self.config, "max_position_embeddings", None)
                if max_position_embeddings is not None:
                    generation_config.max_length = min(generation_config.max_length, max_position_embeddings)

        return generation_config

    def _prepare_generation_config(
        self, generation_config: Optional[DreamGenerationConfig], **kwargs: Dict
    ) -> DreamGenerationConfig:
        """
        Prepares the base generation config, then applies any generation configuration options from kwargs. This
        function handles retrocompatibility with respect to configuration files.
        """
        # priority: `generation_config` argument > `model.generation_config` (the default generation config)
        using_model_generation_config = False
        if generation_config is None:
            generation_config = DreamGenerationConfig.from_model_config(self.config)
            using_model_generation_config = True

        # `torch.compile` can't compile `copy.deepcopy`, arguments in `kwargs` that are part of `generation_config`
        # will mutate the object with `.update`. As such, passing these arguments through `kwargs` is disabled -- an
        # exception will be raised in `_validate_model_kwargs`
        if not is_torchdynamo_compiling():
            generation_config = copy.deepcopy(generation_config)
            _kwargs = generation_config.update(**kwargs)
            # If `generation_config` is provided, let's fallback ALL special tokens to the default values for the model
            if not using_model_generation_config:
                if generation_config.bos_token_id is None:
                    generation_config.bos_token_id = self.generation_config.bos_token_id
                if generation_config.eos_token_id is None:
                    generation_config.eos_token_id = self.generation_config.eos_token_id
                if generation_config.pad_token_id is None:
                    generation_config.pad_token_id = self.generation_config.pad_token_id
                if generation_config.mask_token_id is None:
                    generation_config.mask_token_id = self.generation_config.mask_token_id

        return generation_config

    def _prepare_special_tokens(
        self,
        generation_config: DreamGenerationConfig,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """
        Prepares the special tokens for generation, overwriting the generation config with their processed versions
        converted to tensor.

        Note that `generation_config` is changed in place and stops being serializable after this method is called.
        That is no problem if called within `generate` (`generation_config` is a local copy that doesn't leave the
        function). However, if called outside `generate`, consider creating a copy of `generation_config` first.
        """

        # Convert special tokens to tensors
        def _tensor_or_none(token, device=None):
            if token is None:
                return token

            device = device if device is not None else self.device
            if isinstance(token, torch.Tensor):
                return token.to(device)
            return torch.tensor(token, device=device, dtype=torch.long)

        bos_token_tensor = _tensor_or_none(generation_config.bos_token_id, device=device)
        eos_token_tensor = _tensor_or_none(generation_config.eos_token_id, device=device)
        pad_token_tensor = _tensor_or_none(generation_config.pad_token_id, device=device)
        mask_token_tensor = _tensor_or_none(generation_config.mask_token_id, device=device)

        # We can have more than one eos token. Always treat it as a 1D tensor (when it exists).
        if eos_token_tensor is not None and eos_token_tensor.ndim == 0:
            eos_token_tensor = eos_token_tensor.unsqueeze(0)

        # Set pad token if unset (and there are conditions to do so)
        if pad_token_tensor is None and eos_token_tensor is not None:
            pad_token_tensor = eos_token_tensor[0]
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{pad_token_tensor} for open-end generation.")

        # Update generation config with the updated special tokens tensors
        # NOTE: this must be written into a different attribute name than the one holding the original special tokens
        # (in their non-tensor form), in order to enable end-to-end compilation. See
        # https://pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html#limitations
        generation_config._bos_token_tensor = bos_token_tensor
        generation_config._eos_token_tensor = eos_token_tensor
        generation_config._pad_token_tensor = pad_token_tensor
        generation_config._mask_token_tensor = mask_token_tensor

    @torch.no_grad()
    def diffusion_generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[DreamGenerationConfig] = None,
        **kwargs,
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        generation_config = self._prepare_generation_config(generation_config, **kwargs)
        generation_tokens_hook_func = kwargs.pop("generation_tokens_hook_func", lambda step, x, logits: x)
        generation_logits_hook_func = kwargs.pop("generation_logits_hook_func", lambda step, x, logits: logits)

        tokenizer = None
        if "tokenizer" in kwargs:
            tokenizer = kwargs.get("tokenizer")
        # 2. Define model inputs
        assert inputs is not None
        input_ids = inputs
        device = input_ids.device
        attention_mask = kwargs.pop("attention_mask", None)
        self._prepare_special_tokens(generation_config, device=device)

        # 3. Prepare `max_length`.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            input_ids_length=input_ids_length,
        )

        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)
        
        # 4. Check input_ids
        if not is_torchdynamo_compiling() and self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )
        if (
            hasattr(generation_config, "pad_token_id") and
            torch.any(input_ids == generation_config.pad_token_id) and 
            attention_mask is None
        ):
            warnings.warn(
                "Padding was detected but no attention mask is passed here. For correct "
                "generation results, please set `attention_mask` when batch-padding inputs.",
                UserWarning,
            )

        input_ids, attention_mask = self._expand_inputs_for_generation(
            expand_size=generation_config.num_return_sequences,
            input_ids=input_ids,
            attention_mask=attention_mask 
        )
        result = self._sample(
            input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            generation_tokens_hook_func=generation_tokens_hook_func,
            generation_logits_hook_func=generation_logits_hook_func,
            tokenizer=tokenizer,
            log=False,
        )
        return result

    def _sample(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        generation_config: DreamGenerationConfig,
        generation_tokens_hook_func,
        generation_logits_hook_func,
        tokenizer=None,
        log=False
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        import json
        import math
        import torch
        import torch.nn.functional as F

        # =========================
        # Init values
        # =========================
        output_history = generation_config.output_history
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        steps = generation_config.steps
        eps = generation_config.eps

        alg = generation_config.alg
        # alg = "maskgit_plus"

        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k

        # =========================
        # Fixed confidence APD args
        # =========================
        block_size = getattr(generation_config, "block_size", 32)

        # 兼容不同命名：threshold / confidence_threshold / fixed_threshold
        fixed_threshold = getattr(
            generation_config,
            "threshold",
            0.9
            )

        print_all_token_records = getattr(generation_config, "print_all_token_records", False)

        histories = [] if (return_dict_in_generate and output_history) else None

        selected_records = []
        all_token_records = []

        prompt_len = input_ids.shape[1]
        gen_length = max_length - prompt_len

        assert gen_length > 0, f"max_length={max_length} must be larger than prompt length={prompt_len}"

        num_blocks = math.ceil(gen_length / block_size)
        steps_per_block = max(1, steps // num_blocks)

        print(
            f"======== Dream fixed confidence adaptive parallel decoding: "
            f"alg={alg}, temperature={temperature}, threshold={fixed_threshold}, "
            f"steps={steps}, max_length={max_length}, block_size={block_size}, "
            f"num_blocks={num_blocks}, steps_per_block={steps_per_block} ==========",
            flush=True,
        )

        # =========================
        # Pad input_ids to max_length
        # =========================
        x = F.pad(
            input_ids,
            (0, max_length - input_ids.shape[1]),
            value=mask_token_id,
        )

        if attention_mask is not None and torch.any(attention_mask == 0.0):
            attention_mask = F.pad(
                attention_mask,
                (0, max_length - attention_mask.shape[1]),
                value=1.0,
            )
            tok_idx = attention_mask.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask == 0, 1)

            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
        else:
            tok_idx = None
            attention_mask = "full"

        # hook before generation
        x = generation_tokens_hook_func(None, x, None)

        forward_count = 0
        decoding_steps_used = 0
        global_step = 0

        # =========================
        # Block-by-block fixed confidence APD
        # =========================
        for block_idx in range(num_blocks):
            block_start = prompt_len + block_idx * block_size
            block_end = min(prompt_len + (block_idx + 1) * block_size, max_length)

            for local_step_idx in range(steps_per_block):
                global_step += 1
                local_step = local_step_idx + 1

                # Only current block mask positions are allowed to be decoded
                full_mask_index = (x == mask_token_id)

                current_block_mask_index = torch.zeros_like(full_mask_index, dtype=torch.bool)
                current_block_mask_index[:, block_start:block_end] = full_mask_index[:, block_start:block_end]

                # Current block finished
                if not current_block_mask_index.any():
                    break

                # Whole sequence finished
                if not full_mask_index.any():
                    break

                # =========================
                # Forward
                # =========================
                logits = self(x, attention_mask, tok_idx).logits
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
                forward_count += 1

                logits = generation_logits_hook_func(global_step - 1, x, logits)

                mask_logits = logits[current_block_mask_index]

                if alg == "maskgit_plus":
                    confidence, x0_flat = sample_tokens(
                        mask_logits,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        log_step=global_step - 1,
                    )
                elif alg == "topk_margin":
                    confidence, x0_flat = sample_tokens(
                        mask_logits,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        margin_confidence=True,
                    )
                elif alg == "entropy":
                    confidence, x0_flat = sample_tokens(
                        mask_logits,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        neg_entropy=True,
                    )
                else:
                    raise RuntimeError(f"Unknown alg: {alg}")

                # =========================
                # Full tensor for predicted tokens and confidence
                # =========================
                x0_full = torch.full_like(x, mask_token_id, device=x.device, dtype=torch.long)

                confidence_full = torch.full(
                    x.shape,
                    -torch.inf,
                    device=x.device,
                    dtype=logits.dtype,
                )

                x0_full[current_block_mask_index] = x0_flat
                confidence_full[current_block_mask_index] = confidence

                # =========================
                # Fixed confidence adaptive parallel selection
                # =========================
                selected_mask = torch.zeros_like(x, dtype=torch.bool)

                is_last_step_in_block = (local_step_idx == steps_per_block - 1)

                for batch_idx in range(x.size(0)):
                    candidate_positions = torch.where(current_block_mask_index[batch_idx])[0]

                    if candidate_positions.numel() == 0:
                        continue

                    if is_last_step_in_block:
                        # Last step: decode all remaining tokens in this block
                        selected_positions = candidate_positions
                    else:
                        candidate_conf = confidence_full[batch_idx, candidate_positions]

                        # Fixed confidence threshold
                        selected_positions = candidate_positions[candidate_conf >= fixed_threshold]

                        # Fallback: if no token exceeds threshold, decode the highest-confidence one
                        if selected_positions.numel() == 0:
                            best_idx = torch.argmax(candidate_conf)
                            selected_positions = candidate_positions[best_idx:best_idx + 1]

                    selected_mask[batch_idx, selected_positions] = True

                # =========================
                # Log all current-block candidates before commit
                # =========================
                for batch_idx in range(x.size(0)):
                    candidate_positions = torch.where(current_block_mask_index[batch_idx])[0]
                    selected_pos_set = set(torch.where(selected_mask[batch_idx])[0].detach().cpu().tolist())

                    for pos in candidate_positions:
                        pos_int = int(pos.item())
                        token_id = int(x0_full[batch_idx, pos_int].item())
                        conf = float(confidence_full[batch_idx, pos_int].item())

                        all_token_records.append({
                            "global_step": int(global_step),
                            "local_step": int(local_step),
                            "step": int(local_step),
                            "block": int(block_idx + 1),
                            "position": pos_int,
                            "block_relative_position": int(pos_int - block_start),
                            "generation_relative_position": int(pos_int - prompt_len),
                            "confidence": conf,
                            "token_id": token_id,
                            "selected": bool(pos_int in selected_pos_set),
                        })

                # =========================
                # Commit selected tokens + selected_records
                # =========================
                for batch_idx in range(x.size(0)):
                    selected_positions = torch.where(selected_mask[batch_idx])[0]

                    for pos in selected_positions:
                        pos_int = int(pos.item())
                        token_id = int(x0_full[batch_idx, pos_int].item())
                        conf = float(confidence_full[batch_idx, pos_int].item())

                        selected_records.append({
                            "global_step": int(global_step),
                            "local_step": int(local_step),
                            "step": int(local_step),
                            "block": int(block_idx + 1),
                            "position": pos_int,
                            "block_relative_position": int(pos_int - block_start),
                            "generation_relative_position": int(pos_int - prompt_len),
                            "confidence": conf,
                            "token_id": token_id,
                        })

                if selected_mask.any():
                    decoding_steps_used += 1

                x[selected_mask] = x0_full[selected_mask]

                x = generation_tokens_hook_func(global_step - 1, x, logits)

                if histories is not None:
                    histories.append(x.clone())

                if log:
                    remaining_in_block = int((x[:, block_start:block_end] == mask_token_id).sum().item())
                    selected_count = int(selected_mask.sum().item())

                    print(
                        f"[Block {block_idx + 1}/{num_blocks}] "
                        f"local_step={local_step}, global_step={global_step}, "
                        f"selected={selected_count}, "
                        f"remaining_in_block={remaining_in_block}",
                        flush=True,
                    )

        # =========================
        # Stats + JSON logging
        # =========================
        decoded_tokens = len(selected_records)
        tpf = decoded_tokens / forward_count if forward_count > 0 else 0.0
        avg_tokens_per_decoding_step = (
            decoded_tokens / decoding_steps_used if decoding_steps_used > 0 else 0.0
        )

        print("====== Decoding Statistics ======", flush=True)
        print(f"Decoded tokens: {decoded_tokens}", flush=True)
        print(f"Model forward calls: {forward_count}", flush=True)
        print(f"Actual decoding steps with unmask: {decoding_steps_used}", flush=True)
        print(f"TPF (tokens per forward): {tpf:.4f}", flush=True)
        print(f"Avg tokens per decoding step: {avg_tokens_per_decoding_step:.4f}", flush=True)
        print(f"All candidate token records: {len(all_token_records)}", flush=True)
        print(f"Fixed confidence threshold: {fixed_threshold}", flush=True)

        if print_all_token_records:
            print(json.dumps({
                "selected_records": selected_records,
                "all_token_records": all_token_records,
                "stats": {
                    "decoded_tokens": decoded_tokens,
                    "model_forward_calls": forward_count,
                    "steps": decoding_steps_used,
                    "tpf": tpf,
                    "tokens_per_decoding_step": avg_tokens_per_decoding_step,
                    "num_all_token_records": len(all_token_records),
                    "block_size": int(block_size),
                    "num_blocks": int(num_blocks),
                    "steps_per_block": int(steps_per_block),
                    "fixed_threshold": float(fixed_threshold),
                    "decoding_method": "fixed_confidence_adaptive_parallel",
                }
            }), flush=True)
        else:
            print(json.dumps(selected_records), flush=True)
            print(len(selected_records), flush=True)

        if return_dict_in_generate:
            return DreamModelOutput(
                sequences=x,
                history=histories,
            )
        else:
            return x