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

    # if log_step in (0, 8, 16, 64, 128):
    #     confidence_pickleable = confidence.cpu().float().numpy()
    #     print(confidence_pickleable)
    #     pickle.dump(confidence_pickleable, open(f"./entropy_sink_step{log_step}.pkl", 'wb'))
    
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

        # Parameters that define the output variables of `generate`
        self.num_return_sequences: int = kwargs.pop("num_return_sequences", 1)
        self.return_dict_in_generate: bool = kwargs.pop("return_dict_in_generate", False)
        self.output_history: bool = kwargs.pop("output_history", False)

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
# 曹
        result = self._sample_beam_search_auto_beam_step_average(
            input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            generation_tokens_hook_func=generation_tokens_hook_func,
            generation_logits_hook_func=generation_logits_hook_func,
            tokenizer=tokenizer,
            log=False,
        )
        return result

    def _sample_beam_search_auto_beam_step_average(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        generation_config: DreamGenerationConfig,
        generation_tokens_hook_func,
        generation_logits_hook_func,
        tokenizer=None,
        log=False
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        """改进的Beam Search采样方法 - 基于max_beam_size的动态调整"""
        # 初始化参数
        output_history = generation_config.output_history
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        eps = generation_config.eps
        alg = generation_config.alg
        alg = "maskgit_plus"
        alg_temp = generation_config.alg_temp
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k

        # 设置max_beam_size
        max_beam_size = 2  # 可以调整为更大的值
        entropy_threshhold = 0.90
        min_high_confidence_len = 1
        max_high_confidence_len = 5
        print(f"========动态Beam Search自动调整beam size和token/step,平均置信度排序: max_beam_size={max_beam_size},\
            entropy_threshhold={entropy_threshhold},alg={alg},\
            min_high_confidence_len={min_high_confidence_len}, max_high_confidence_len={max_high_confidence_len}, \
                max_length={max_length}==========", flush=True)
        histories = [] if (return_dict_in_generate and output_history) else None

        # 统计信息
        total_steps = 0
        beam_search_steps = 0

        # 填充输入
        x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)
        max_mask_length = max_length - input_ids.shape[1]
        print(f"===========max mask length{max_mask_length}============")
        if attention_mask is not None and torch.any(attention_mask == 0.0):
            attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
            tok_idx = attention_mask.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask == 0, 1)
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
        else:
            tok_idx = None
            attention_mask = "full"

        # 初始化beam - 现在存储(sequence, cumulative_avg_prob, num_unmasked_tokens)
        beam_size = 1  # 初始beam_size
        beam = [(x.clone(), 0.0, 0)]  # (sequence, cumulative_avg_prob, num_unmasked_tokens)
        beam = [(generation_tokens_hook_func(None, seq, None), avg_prob, num_tokens) 
                for seq, avg_prob, num_tokens in beam]
        
        if log:
            initial_mask_count = (x == mask_token_id).sum().item()
            print(f"初始mask数量: {initial_mask_count}")
            print(f"max_beam_size: {max_beam_size}")
            if tokenizer is not None:
                print(f"初始序列: {tokenizer.decode(x[0][-max_mask_length:])}")
            print()

        # 动态step循环
        while True:
            total_steps += 1
            
            if log:
                print(f"=== Step {total_steps} (beam_size={beam_size}) ===")
                print(f"当前beam大小: {len(beam)}")

            if len(beam) > 1:
                beam_search_steps += 1
            # 检查是否还有mask
            current_seq = beam[0][0]
            mask_index = (current_seq == mask_token_id)
            num_mask_token = mask_index.sum().item()
            
            if num_mask_token == 0:
                if log:
                    print("所有mask已填充，生成完成！")
                break

            new_beam_candidates = []
            has_multi_unmask_candidate = False  # 标记是否有一次性unmask多个token的候选

            # 对beam中的每个候选序列进行扩展
            for seq_idx, (seq, cumulative_avg_prob, num_unmasked_tokens) in enumerate(beam):
                if log:
                    print(f"--- 处理候选 {seq_idx+1} ---")
                    if tokenizer is not None:
                        decoded_text = tokenizer.decode(seq[0][-max_mask_length:])
                        print(f"当前序列: {decoded_text}")
                        print(f"当前平均概率: {cumulative_avg_prob:.4f}, 已unmask token数: {num_unmasked_tokens}")

                with torch.no_grad():
                    logits = self(seq, attention_mask, tok_idx).logits
                    logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)

                logits = generation_logits_hook_func(total_steps-1, seq, logits)

                # 找到所有mask位置
                mask_index = (seq == mask_token_id)
                
                if not mask_index.any():
                    new_beam_candidates.append((seq, cumulative_avg_prob, num_unmasked_tokens))
                    continue

                # 获取mask位置的logits
                mask_logits = logits[mask_index]
                mask_positions = mask_index.squeeze(0).nonzero().squeeze(1)
                
                # 为每个mask位置采样token和概率
                if alg == 'maskgit_plus':
                    token_probs, candidate_tokens = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, neg_entropy=False)
                elif alg == 'entropy':
                    token_probs, candidate_tokens = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, neg_entropy=True)
                else:
                    probs = F.softmax(mask_logits / temperature, dim=-1)
                    token_probs, candidate_tokens = torch.max(probs, dim=-1)
                
                if len(token_probs.size()) > 1:
                    token_probs = token_probs.squeeze(-1)
                    candidate_tokens = candidate_tokens.squeeze(-1)

                if log and tokenizer is not None:
                    print("当前mask位置的候选token:")
                    for pos_idx, (pos, token, prob) in enumerate(zip(mask_positions, candidate_tokens, token_probs)):
                        if pos_idx < 10:
                            token_text = tokenizer.decode([token])
                            print(f"  位置{pos.item()}: '{token_text}' (概率: {prob:.4f})")

                # 检查是否有>=?个confidence>entropy_threshhold的token（对应概率>0.9）
                high_confidence_mask = token_probs > entropy_threshhold
                high_confidence_indices = high_confidence_mask.nonzero().squeeze(1)
                
                # 策略选择
                if len(high_confidence_indices) >= min_high_confidence_len:
                    # 策略(1): 有>=2个高置信度token，一次性解码多个（不需要连续）
                    if log:
                        print(f"策略1: 发现{len(high_confidence_indices)}个高置信度token，一次性解码多个")
                
                    # 选择置信度最高的前几个token（最多5个）
                    num_to_unmask = min(len(high_confidence_indices), max_high_confidence_len)
                    top_probs, top_indices = torch.topk(token_probs[high_confidence_indices], num_to_unmask)
                    selected_indices = high_confidence_indices[top_indices]
                    
                    new_seq = seq.clone()
                    
                    # 计算新的平均概率
                    total_prob = cumulative_avg_prob * num_unmasked_tokens  # 还原总和
                    new_unmasked_count = num_unmasked_tokens + num_to_unmask
                    
                    if log and tokenizer is not None:
                        group_details = []
                    
                    for idx in range(num_to_unmask):
                        original_idx = selected_indices[idx].item()
                        pos = mask_positions[original_idx].item()
                        token = candidate_tokens[original_idx]
                        prob = top_probs[idx].item()
                        
                        new_seq[0, pos] = token
                        total_prob += prob  # 累加概率
                        
                        if log and tokenizer is not None:
                            token_text = tokenizer.decode([token])
                            group_details.append(f"位置{pos}->'{token_text}'(概率:{prob:.4f})")
                    
                    # 计算新的平均概率
                    new_avg_prob = total_prob / new_unmasked_count
                    
                    new_beam_candidates.append((new_seq, new_avg_prob, new_unmasked_count))
                    
                    if log and tokenizer is not None:
                        print(f"  一次性解码{num_to_unmask}个token: {', '.join(group_details)}")
                        print(f"  新平均概率: {new_avg_prob:.4f} (基于{new_unmasked_count}个token)")
                    
                    # 标记这是一个一次性unmask多个token的候选
                    has_multi_unmask_candidate = True
                    
                else:
                    # 策略(2): 没有足够的高置信度token，选择top max_beam_size进行unmask
                    if log:
                        print(f"策略2: 高置信度token不足({len(high_confidence_indices)}个)，选择top-{max_beam_size}进行unmask")
                    
                    # 选择top max_beam_size个位置
                    k = min(max_beam_size, len(token_probs))
                    top_probs, top_indices = torch.topk(token_probs, k)
                    top_positions = mask_positions[top_indices]
                    top_tokens = candidate_tokens[top_indices]
                    
                    for idx in range(k):
                        new_seq = seq.clone()
                        pos = top_positions[idx].item()
                        token = top_tokens[idx]
                        prob = top_probs[idx].item()
                        
                        new_seq[0, pos] = token
                        
                        # 计算新的平均概率
                        total_prob = cumulative_avg_prob * num_unmasked_tokens  # 还原总和
                        new_total_prob = total_prob + prob
                        new_unmasked_count = num_unmasked_tokens + 1
                        new_avg_prob = new_total_prob / new_unmasked_count
                        
                        new_beam_candidates.append((new_seq, new_avg_prob, new_unmasked_count))
                        
                        if log and tokenizer is not None and idx < 3:
                            token_text = tokenizer.decode([token])
                            print(f"  单token unmask: 位置{pos} -> '{token_text}', 概率{prob:.4f}, 新平均概率{new_avg_prob:.4f}")

            # 如果没有生成新的候选，提前结束
            if not new_beam_candidates:
                print("没有生成新的候选序列，提前结束")
                break

            # 全局排序，选择top beam_size个序列（按平均概率排序）
            new_beam_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # 去重
            uniq_new_beam_candidates = []
            seen = set()
            for tensor, avg_prob, num_tokens in new_beam_candidates:
                tensor_tuple = tuple(tensor.flatten().tolist())
                if tensor_tuple not in seen:
                    seen.add(tensor_tuple)
                    uniq_new_beam_candidates.append((tensor, avg_prob, num_tokens))
            
            # 重排序和调整beam_size
            if has_multi_unmask_candidate and uniq_new_beam_candidates:
                # 检查概率第一的序列是否是一次性unmask多个token的
                best_candidate = uniq_new_beam_candidates[0]
                best_seq, best_avg_prob, best_num_tokens = best_candidate
                
                # 简单判断：如果这个候选序列比原始序列少了多个mask，说明是一次性unmask多个
                original_mask_count = (beam[0][0] == mask_token_id).sum().item()
                current_mask_count = (best_seq == mask_token_id).sum().item()
                masks_unmasked = original_mask_count - current_mask_count
                
                if masks_unmasked >= 2:
                    # 策略(1): 概率第一的序列是一次性unmask多个token，缩小beam_size=1
                    beam_size = 1
                    beam = [best_candidate]  # 只保留最佳序列
                    if log:
                        print(f"重排序策略1: 最佳候选一次性unmask了{masks_unmasked}个token，beam_size缩小为1")
                        print(f"最佳候选平均概率: {best_avg_prob:.4f} (基于{best_num_tokens}个token)")
                else:
                    # 策略(2): 否则扩大beam_size=max_beam_size
                    beam_size = max_beam_size
                    beam = uniq_new_beam_candidates[:beam_size]
                    if log:
                        print(f"重排序策略2: 最佳候选只unmask了{masks_unmasked}个token，beam_size扩大为{max_beam_size}")
            else:
                # 策略(2): 没有一次性unmask多个token的候选，扩大beam_size=max_beam_size
                beam_size = max_beam_size
                beam = uniq_new_beam_candidates[:beam_size]
                if log:
                    print(f"重排序策略2: 无一次性unmask多个token候选，beam_size扩大为{max_beam_size}")
            
            # 应用token钩子
            beam = [(generation_tokens_hook_func(total_steps-1, seq, None), avg_prob, num_tokens) 
                    for seq, avg_prob, num_tokens in beam]
            
            # 记录历史
            if histories is not None and beam:
                best_seq = beam[0][0]
                histories.append(best_seq.clone())

            if log:
                best_seq, best_avg_prob, best_num_tokens = beam[0]
                print(f"当前最佳序列平均概率: {best_avg_prob:.4f} (基于{best_num_tokens}个token)")
                print(f"剩余mask数量: {(best_seq == mask_token_id).sum().item()}")
                if tokenizer is not None:
                    print("当前beam中的所有序列:")
                    for beam_idx, (seq, avg_prob, num_tokens) in enumerate(beam):
                        decoded_text = tokenizer.decode(seq[0][-max_mask_length:])
                        print(f"  Beam {beam_idx+1} (平均概率: {avg_prob:.4f}, token数: {num_tokens}): {decoded_text}")
                print()

        # 选择beam中最好的序列作为最终结果
        if beam:
            best_sequence = beam[0][0]
            final_avg_prob = beam[0][1]
            final_num_tokens = beam[0][2]
        else:
            best_sequence = x
            final_avg_prob = 0.0
            final_num_tokens = 0

        if log:
            print(f"=== 生成完成 ===")
            print(f"总步数: {total_steps}")
            print(f"Beam Search步数: {beam_search_steps}")
            print(f"最终序列平均概率: {final_avg_prob:.4f} (基于{final_num_tokens}个token)")
            print(f"最终mask数量: {(best_sequence == mask_token_id).sum().item()}")
            if tokenizer is not None:
                final_text = tokenizer.decode(best_sequence[0][-max_mask_length:])
                print(f"最终生成文本: {final_text}")

        if return_dict_in_generate:
            return DreamModelOutput(
                sequences=best_sequence.unsqueeze(0) if best_sequence.dim() == 1 else best_sequence,
                history=histories,
            )
        else:
            return best_sequence.unsqueeze(0) if best_sequence.dim() == 1 else best_sequence

    def _sample_beam_search_auto_beam_step(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        generation_config: DreamGenerationConfig,
        generation_tokens_hook_func,
        generation_logits_hook_func,
        tokenizer=None,
        log=False
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        """改进的Beam Search采样方法 - 基于max_beam_size的动态调整"""
        # 初始化参数
        output_history = generation_config.output_history
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        eps = generation_config.eps
        alg = generation_config.alg
        alg = "entropy"
        alg_temp = generation_config.alg_temp
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k

        max_beam_size = 2  # 可以调整为更大的值
        entropy_threshhold = -0.1
        min_high_confidence_len = 2
        max_high_confidence_len = 5
        print(f"========动态Beam Search自动调整beam size和token/step,置信度之和排序: max_beam_size={max_beam_size},\
            entropy_threshhold={entropy_threshhold},alg={alg},\
            min_high_confidence_len={min_high_confidence_len}, max_high_confidence_len={max_high_confidence_len}, \
                max_length={max_length}==========", flush=True)
        histories = [] if (return_dict_in_generate and output_history) else None

        # 统计信息
        total_steps = 0
        beam_search_steps = 0

        # 填充输入
        x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)
        max_mask_length = max_length - input_ids.shape[1]
        print(f"===========max mask length{max_mask_length}============")
        if attention_mask is not None and torch.any(attention_mask == 0.0):
            attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
            tok_idx = attention_mask.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask == 0, 1)
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
        else:
            tok_idx = None
            attention_mask = "full"

        # 初始化beam
        beam_size = 1  # 初始beam_size
        beam = [(x.clone(), 0.0)]  # (sequence, cumulative_log_prob)
        beam = [(generation_tokens_hook_func(None, seq, None), log_prob) 
                for seq, log_prob in beam]
        
        if log:
            initial_mask_count = (x == mask_token_id).sum().item()
            print(f"初始mask数量: {initial_mask_count}")
            print(f"max_beam_size: {max_beam_size}")
            if tokenizer is not None:
                print(f"初始序列: {tokenizer.decode(x[0][-max_mask_length:])}")
            print()

        # 动态step循环
        while True:
            total_steps += 1
            
            if log:
                print(f"=== Step {total_steps} (beam_size={beam_size}) ===")
                print(f"当前beam大小: {len(beam)}")

            if len(beam) > 1:
                beam_search_steps += 1
            # 检查是否还有mask
            current_seq = beam[0][0]
            mask_index = (current_seq == mask_token_id)
            num_mask_token = mask_index.sum().item()
            
            if num_mask_token == 0:
                if log:
                    print("所有mask已填充，生成完成！")
                break

            new_beam_candidates = []
            has_multi_unmask_candidate = False  # 标记是否有一次性unmask多个token的候选

            # 对beam中的每个候选序列进行扩展
            for seq_idx, (seq, cumulative_log_prob) in enumerate(beam):
                if log:
                    print(f"--- 处理候选 {seq_idx+1} ---")
                    if tokenizer is not None:
                        decoded_text = tokenizer.decode(seq[0][-max_mask_length:])
                        print(f"当前序列: {decoded_text}")
                        print(f"当前分数: {cumulative_log_prob:.4f}")

                with torch.no_grad():
                    logits = self(seq, attention_mask, tok_idx).logits
                    logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)

                logits = generation_logits_hook_func(total_steps-1, seq, logits)

                # 找到所有mask位置
                mask_index = (seq == mask_token_id)
                
                if not mask_index.any():
                    new_beam_candidates.append((seq, cumulative_log_prob))
                    continue

                # 获取mask位置的logits
                mask_logits = logits[mask_index]
                mask_positions = mask_index.squeeze(0).nonzero().squeeze(1)
                
                # 为每个mask位置采样token和概率
                if alg == 'maskgit_plus':
                    token_probs, candidate_tokens = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, neg_entropy=False)
                elif alg == 'entropy':
                    token_probs, candidate_tokens = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, neg_entropy=True)
                else:
                    probs = F.softmax(mask_logits / temperature, dim=-1)
                    token_probs, candidate_tokens = torch.max(probs, dim=-1)
                
                if len(token_probs.size()) > 1:
                    token_probs = token_probs.squeeze(-1)
                    candidate_tokens = candidate_tokens.squeeze(-1)

                if log and tokenizer is not None:
                    print("当前mask位置的候选token:")
                    for pos_idx, (pos, token, prob) in enumerate(zip(mask_positions, candidate_tokens, token_probs)):
                        if pos_idx < 10:
                            token_text = tokenizer.decode([token])
                            print(f"  位置{pos.item()}: '{token_text}' (概率: {prob:.4f})")

                # 检查是否有>=2个confidence>entropy_threshhold的token（对应概率>0.9）
                high_confidence_mask = token_probs > entropy_threshhold
                high_confidence_indices = high_confidence_mask.nonzero().squeeze(1)
                
                # 策略选择
                if len(high_confidence_indices) >= min_high_confidence_len:
                    # 策略(1): 有>=2个高置信度token，一次性解码多个（不需要连续）
                    if log:
                        print(f"策略1: 发现{len(high_confidence_indices)}个高置信度token，一次性解码多个")
                    
                    # 选择置信度最高的前几个token（最多5个）
                    num_to_unmask = min(len(high_confidence_indices), max_high_confidence_len)
                    top_probs, top_indices = torch.topk(token_probs[high_confidence_indices], num_to_unmask)
                    selected_indices = high_confidence_indices[top_indices]
                    
                    new_seq = seq.clone()
                    group_log_prob = cumulative_log_prob
                    
                    if log and tokenizer is not None:
                        group_details = []
                    
                    for idx in range(num_to_unmask):
                        original_idx = selected_indices[idx].item()
                        pos = mask_positions[original_idx].item()
                        token = candidate_tokens[original_idx]
                        prob = top_probs[idx].item()
                        
                        new_seq[0, pos] = token
                        group_log_prob += prob
                        
                        if log and tokenizer is not None:
                            token_text = tokenizer.decode([token])
                            group_details.append(f"位置{pos}->'{token_text}'(概率:{prob:.4f})")
                    
                    new_beam_candidates.append((new_seq, group_log_prob))
                    
                    if log and tokenizer is not None:
                        print(f"  一次性解码{num_to_unmask}个token: {', '.join(group_details)}")
                    
                    # 标记这是一个一次性unmask多个token的候选
                    has_multi_unmask_candidate = True
                    
                else:
                    # 策略(2): 没有足够的高置信度token，选择top max_beam_size进行unmask
                    if log:
                        print(f"策略2: 高置信度token不足({len(high_confidence_indices)}个)，选择top-{max_beam_size}进行unmask")
                    
                    # 选择top max_beam_size个位置
                    k = min(max_beam_size, len(token_probs))
                    top_probs, top_indices = torch.topk(token_probs, k)
                    top_positions = mask_positions[top_indices]
                    top_tokens = candidate_tokens[top_indices]
                    
                    for idx in range(k):
                        new_seq = seq.clone()
                        pos = top_positions[idx].item()
                        token = top_tokens[idx]
                        prob = top_probs[idx].item()
                        
                        new_seq[0, pos] = token
                        new_log_prob = cumulative_log_prob + prob
                        new_beam_candidates.append((new_seq, new_log_prob))
                        
                        if log and tokenizer is not None and idx < 3:
                            token_text = tokenizer.decode([token])
                            print(f"  单token unmask: 位置{pos} -> '{token_text}', 概率{prob:.4f}")

            # 如果没有生成新的候选，提前结束
            if not new_beam_candidates:
                print("没有生成新的候选序列，提前结束")
                break

            # 全局排序，选择top beam_size个序列
            new_beam_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # 去重
            uniq_new_beam_candidates = []
            seen = set()
            for tensor, float_val in new_beam_candidates:
                tensor_tuple = tuple(tensor.flatten().tolist())
                if tensor_tuple not in seen:
                    seen.add(tensor_tuple)
                    uniq_new_beam_candidates.append((tensor, float_val))
            
            # 重排序和调整beam_size
            if has_multi_unmask_candidate and uniq_new_beam_candidates:
                # 检查概率第一的序列是否是一次性unmask多个token的
                best_candidate = uniq_new_beam_candidates[0]
                best_seq, best_score = best_candidate
                
                # 简单判断：如果这个候选序列比原始序列少了多个mask，说明是一次性unmask多个
                original_mask_count = (beam[0][0] == mask_token_id).sum().item()
                current_mask_count = (best_seq == mask_token_id).sum().item()
                masks_unmasked = original_mask_count - current_mask_count
                
                if masks_unmasked >= 2:
                    # 策略(1): 概率第一的序列是一次性unmask多个token，缩小beam_size=1
                    beam_size = 1
                    beam = [best_candidate]  # 只保留最佳序列
                    if log:
                        print(f"重排序策略1: 最佳候选一次性unmask了{masks_unmasked}个token，beam_size缩小为1")
                else:
                    # 策略(2): 否则扩大beam_size=max_beam_size
                    beam_size = max_beam_size
                    beam = uniq_new_beam_candidates[:beam_size]
                    if log:
                        print(f"重排序策略2: 最佳候选只unmask了{masks_unmasked}个token，beam_size扩大为{max_beam_size}")
            else:
                # 策略(2): 没有一次性unmask多个token的候选，扩大beam_size=max_beam_size
                beam_size = max_beam_size
                beam = uniq_new_beam_candidates[:beam_size]
                if log:
                    print(f"重排序策略2: 无一次性unmask多个token候选，beam_size扩大为{max_beam_size}")
            
            # 应用token钩子
            beam = [(generation_tokens_hook_func(total_steps-1, seq, None), log_prob) 
                    for seq, log_prob in beam]
            
            # 记录历史
            if histories is not None and beam:
                best_seq = beam[0][0]
                histories.append(best_seq.clone())

            if log:
                best_seq, best_score = beam[0]
                print(f"当前最佳序列分数: {best_score:.4f}")
                print(f"剩余mask数量: {(best_seq == mask_token_id).sum().item()}")
                if tokenizer is not None:
                    print("当前beam中的所有序列:")
                    for beam_idx, (seq, score) in enumerate(beam):
                        decoded_text = tokenizer.decode(seq[0][-max_mask_length:])
                        print(f"  Beam {beam_idx+1} (分数: {score:.4f}): {decoded_text}")
                print()

        # 选择beam中最好的序列作为最终结果
        if beam:
            best_sequence = beam[0][0]
            final_score = beam[0][1]
        else:
            best_sequence = x
            final_score = 0.0

        if log:
            print(f"=== 生成完成 ===")
            print(f"总步数: {total_steps}")
            print(f"Beam Search步数: {beam_search_steps}")
            print(f"最终序列分数: {final_score:.4f}")
            print(f"最终mask数量: {(best_sequence == mask_token_id).sum().item()}")
            if tokenizer is not None:
                final_text = tokenizer.decode(best_sequence[0][-max_mask_length:])
                print(f"最终生成文本: {final_text}")

        if return_dict_in_generate:
            return DreamModelOutput(
                sequences=best_sequence.unsqueeze(0) if best_sequence.dim() == 1 else best_sequence,
                history=histories,
            )
        else:
            return best_sequence.unsqueeze(0) if best_sequence.dim() == 1 else best_sequence

    def _sample_beam_search_multi_token(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        generation_config: DreamGenerationConfig,
        generation_tokens_hook_func,
        generation_logits_hook_func,
        tokenizer=None,  # 新增tokenizer参数
        log=False
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        """改进的Beam Search采样方法 - 动态计算每步unmask数量"""
        # 初始化参数
        output_history = generation_config.output_history
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        steps = generation_config.steps
        eps = generation_config.eps
        alg = generation_config.alg
        alg_temp = generation_config.alg_temp
        # beam_size = generation_config.beam_size
        beam_size = 2
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k

        print(f"========Beam Search: parallel mask, beam_size={beam_size},max_length_with_prompt={max_length},steps={steps},alg={alg},temperature={temperature}==========", flush=True)
        histories = [] if (return_dict_in_generate and output_history) else None

        # 填充输入
        x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)
        max_mask_length = max_length - input_ids.shape[1]
        if attention_mask is not None and torch.any(attention_mask == 0.0):
            attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
            tok_idx = attention_mask.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask == 0, 1)
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
        else:
            tok_idx = None
            attention_mask = "full"

        timesteps = torch.linspace(1, eps, steps + 1, device=x.device)
        
        # 初始化beam
        beam = [(x.clone(), 0.0)]
        beam = [(generation_tokens_hook_func(None, seq, None), log_prob) 
                for seq, log_prob in beam]
        
        if log:
            initial_mask_count = (x == mask_token_id).sum().item()
            print(f"初始mask数量: {initial_mask_count}")
            print(f"总步数: {steps}, beam_size: {beam_size}")
            if tokenizer is not None:
                print(f"初始序列: {tokenizer.decode(x[0][-max_mask_length:])}")
            print()

        for i in range(steps):
            new_beam_candidates = []
            if log:
                print(f"=== Step {i+1}/{steps} ===")
                print(f"当前beam大小: {len(beam)}")
            
            # 动态计算当前步要unmask的数量
            # 使用参考代码中的公式：number_transfer_tokens = int(num_mask_token * (1 - s / t))
            t = timesteps[i]
            s = timesteps[i + 1]
            
            # 获取当前beam中第一个序列的mask数量（假设beam中序列的mask数量相近）
            current_seq = beam[0][0]
            mask_index = (current_seq == mask_token_id)
            num_mask_token = mask_index.sum().item()
            
            if i < steps - 1:
                # 正常步骤：根据时间步比例计算unmask数量
                unmask_per_step = max(1, int(num_mask_token * (1 - s / t)))
            else:
                # 最后一步：unmask所有剩余位置
                unmask_per_step = num_mask_token
            
            # 确保unmask数量不超过剩余mask数量
            unmask_per_step = min(unmask_per_step, num_mask_token)
            
            if log:
                print(f"时间步: t={t:.3f}, s={s:.3f}, 1-s/t={1-s/t:.3f}")
                print(f"剩余mask: {num_mask_token}, 本步unmask: {unmask_per_step}")

            # 计算需要的前k个位置数
            if num_mask_token < unmask_per_step:
                # 如果剩余mask不足，调整unmask数量
                actual_unmask = num_mask_token
                k = num_mask_token
            else:
                actual_unmask = unmask_per_step
                # 计算最小k：C(k, actual_unmask) >= beam_size
                k = actual_unmask
                combinations = 1
                while combinations < beam_size:
                    k += 1
                    combinations = math.comb(k, actual_unmask)
            
            if log:
                print(f"实际unmask: {actual_unmask}, 选取top-{k}位置")

            # 对beam中的每个候选序列进行扩展
            for seq_idx, (seq, cumulative_log_prob) in enumerate(beam):
                if log:  # 只对第一个候选序列详细日志
                    print(f"--- 处理候选 {seq_idx+1} ---")
                    if tokenizer is not None:
                        decoded_text = tokenizer.decode(seq[0][-max_mask_length:])
                        print(f"当前序列: {decoded_text}")
                        print(f"当前分数: {cumulative_log_prob:.4f}")

                with torch.no_grad():
                    logits = self(seq, attention_mask, tok_idx).logits
                    logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)

                logits = generation_logits_hook_func(i, seq, logits)

                # 找到所有mask位置
                mask_index = (seq == mask_token_id)
                
                if not mask_index.any():
                    if log and seq_idx == 0:
                        print("没有mask token，跳过扩展")
                    new_beam_candidates.append((seq, cumulative_log_prob))
                    continue

                # 获取mask位置的logits
                mask_logits = logits[mask_index]  # [num_mask, vocab_size]
                mask_positions = mask_index.squeeze(0).nonzero().squeeze(1)
                
                # 为每个mask位置采样最佳token
                if alg == 'maskgit_plus':
                    token_probs, candidate_tokens = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, neg_entropy=False)
                elif alg == 'entropy':
                    token_probs, candidate_tokens = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, neg_entropy=True)
                else:
                    probs = F.softmax(mask_logits / temperature, dim=-1)
                    token_probs, candidate_tokens = torch.max(probs, dim=-1)
                
                if len(token_probs.size()) > 1:
                    token_probs = token_probs.squeeze(-1)
                    candidate_tokens = candidate_tokens.squeeze(-1)

                if log and tokenizer is not None and seq_idx == 0:
                    print("当前mask位置的候选token:")
                    for pos_idx, (pos, token, prob) in enumerate(zip(mask_positions, candidate_tokens, token_probs)):
                        if pos_idx < 10:  # 只显示前10个
                            token_text = tokenizer.decode([token])
                            print(f"  位置{pos.item()}: '{token_text}' (概率: {prob:.4f})")

                # 选择top-k个位置
                if len(token_probs) <= k:
                    top_probs = token_probs
                    top_indices = torch.arange(len(token_probs))
                else:
                    top_probs, top_indices = torch.topk(token_probs, k)
                
                top_positions = mask_positions[top_indices]
                top_tokens = candidate_tokens[top_indices]

                if log and tokenizer is not None and seq_idx == 0:
                    print(f"选取的top-{k}位置和token:")
                    for idx, (pos, token, prob) in enumerate(zip(top_positions, top_tokens, top_probs)):
                        if idx < 10:  # 只显示前10个
                            token_text = tokenizer.decode([token])
                            print(f"  {idx+1}. 位置{pos.item()}: '{token_text}' (概率: {prob:.4f})")

                # 生成位置组合
                if actual_unmask == 0:
                    # 没有位置可unmask
                    new_beam_candidates.append((seq, cumulative_log_prob))
                    continue
                elif actual_unmask == 1:
                    # 单位置unmask，直接取top beam_size个
                    for idx in range(min(beam_size, len(top_positions))):
                        new_seq = seq.clone()
                        pos = top_positions[idx].item()
                        token = top_tokens[idx]
                        prob = top_probs[idx].item()
                        
                        new_seq[0, pos] = token
                        new_log_prob = cumulative_log_prob + prob
                        new_beam_candidates.append((new_seq, new_log_prob))
                        
                        if log and tokenizer is not None and seq_idx == 0 and idx < 2:  # 只显示前2个
                            token_text = tokenizer.decode([token])
                            print(f"  单位置unmask: 位置{pos} -> '{token_text}', 置信度{prob:.4f}")
                else:
                    # 多位置unmask：按顺序生成组合
                    from itertools import combinations
                    
                    # 生成所有可能的actual_unmask组合
                    position_combinations = list(combinations(range(len(top_positions)), actual_unmask))
                    
                    if log and seq_idx == 0:
                        print(f"生成{len(position_combinations)}个位置组合")
                    
                    # 按顺序选择前beam_size个组合
                    selected_combinations = position_combinations[:beam_size]
                    
                    for combo_idx, position_indices in enumerate(selected_combinations):
                        new_seq = seq.clone()
                        combo_log_prob = cumulative_log_prob
                        
                        if log and tokenizer is not None and seq_idx == 0 and combo_idx < 3:  # 只显示前3个组合
                            combo_details = []
                            
                        for pos_idx in position_indices:
                            actual_pos = top_positions[pos_idx].item()
                            token = top_tokens[pos_idx]
                            prob = top_probs[pos_idx].item()
                            
                            new_seq[0, actual_pos] = token
                            combo_log_prob += prob
                            
                            if log and tokenizer is not None and seq_idx == 0 and combo_idx < 3:
                                token_text = tokenizer.decode([token])
                                combo_details.append(f"位置{actual_pos}->'{token_text}'")
                        
                        new_beam_candidates.append((new_seq, combo_log_prob))
                        
                        if log and tokenizer is not None and seq_idx == 0 and combo_idx < 3:
                            print(f"  组合{combo_idx+1}: {', '.join(combo_details)}, 总置信度{combo_log_prob:.4f}")

            # 如果没有生成新的候选，提前结束
            if not new_beam_candidates:
                print("没有生成新的候选序列，提前结束")
                break

            # 全局排序，选择top beam_size个序列
            new_beam_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # 去重
            uniq_new_beam_candidates = []
            seen = set()
            for tensor, float_val in new_beam_candidates:
                tensor_tuple = tuple(tensor.flatten().tolist())
                if tensor_tuple not in seen:
                    seen.add(tensor_tuple)
                    uniq_new_beam_candidates.append((tensor, float_val))
            
            beam = uniq_new_beam_candidates[:beam_size]
            
            # 应用token钩子
            beam = [(generation_tokens_hook_func(i, seq, None), log_prob) 
                    for seq, log_prob in beam]
            
            # 记录历史
            if histories is not None and beam:
                best_seq = beam[0][0]
                histories.append(best_seq.clone())

            if log:
                best_seq, best_score = beam[0]
                print(f"当前最佳序列分数: {best_score:.4f}")
                print(f"剩余mask数量: {(best_seq == mask_token_id).sum().item()}")
                if tokenizer is not None:
                    print("当前beam中的所有序列:")
                    for beam_idx, (seq, score) in enumerate(beam):
                        decoded_text = tokenizer.decode(seq[0][-max_mask_length:])
                        print(f"  Beam {beam_idx+1} (分数: {score:.4f}): {decoded_text}")
                print()

        # 选择beam中最好的序列作为最终结果
        if beam:
            best_sequence = beam[0][0]
            final_score = beam[0][1]
        else:
            best_sequence = x
            final_score = 0.0

        if log:
            print(f"=== 生成完成 ===")
            print(f"最终序列分数: {final_score:.4f}")
            print(f"最终mask数量: {(best_sequence == mask_token_id).sum().item()}")
            if tokenizer is not None:
                final_text = tokenizer.decode(best_sequence[0][-max_mask_length:])
                print(f"最终生成文本: {final_text}")

        if return_dict_in_generate:
            return DreamModelOutput(
                sequences=best_sequence.unsqueeze(0) if best_sequence.dim() == 1 else best_sequence,
                history=histories,
            )
        else:
            return best_sequence.unsqueeze(0) if best_sequence.dim() == 1 else best_sequence

    def _sample_beam_search(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        generation_config: DreamGenerationConfig,
        generation_tokens_hook_func,
        generation_logits_hook_func,
        log=False,
        tokenizer=None
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        """正确的Beam Search采样方法 - 选择概率最高的位置进行unmask"""
        # 初始化参数
        output_history = generation_config.output_history
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        steps = generation_config.steps
        eps = generation_config.eps
        alg = generation_config.alg
        alg_temp = generation_config.alg_temp
        # beam_size = generation_config.beam_size
        beam_size = 2
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k

        print(f"========Beam Search, beam_size={beam_size},max_length={max_length},steps={steps},alg={alg},temperature={temperature}==========", flush=True)
        histories = [] if (return_dict_in_generate and output_history) else None

        # 填充输入
        x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)

        if attention_mask is not None and torch.any(attention_mask == 0.0):
            attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
            tok_idx = attention_mask.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask == 0, 1)
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
        else:
            tok_idx = None
            attention_mask = "full"

        timesteps = torch.linspace(1, eps, steps + 1, device=x.device)
        
        # 初始化beam: [(sequence, cumulative_log_prob)]
        beam = [(x.clone(), 0.0)]
        beam = [(generation_tokens_hook_func(None, seq, None), log_prob) 
                for seq, log_prob in beam]
        if log:
            print(f"=== Beam Search 初始序列 (最后128个token) ===")
            print(f"x[-128:]: {x[0, -128:].tolist()}")
            print(f"初始mask数量: {(x == mask_token_id).sum().item()}")
            print(f"beam_size: {beam_size}")
            print()
        
        for i in range(steps):
            new_beam_candidates = []
            if log:
                print(f"=== Beam Search Step {i+1}/{steps} ===")
                print(f"当前beam大小: {len(beam)}")
                print(f"当前beam分数: {[score for _, score in beam]}")
            
            # 对beam中的每个候选序列进行扩展
            for j, (seq, cumulative_log_prob) in enumerate(beam):

                if log:
                    print(f"--- 处理beam候选 {j+1} ---")
                    print(f"当前序列分数: {cumulative_log_prob}")
                    print(f"当前序列最后128token: {seq[0, -128:].tolist()}")
                
                with torch.no_grad():
                    logits = self(seq, attention_mask, tok_idx).logits
                    logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)

                logits = generation_logits_hook_func(i, seq, logits)

                # 找到所有mask位置
                mask_index = (seq == mask_token_id)
                
                if not mask_index.any():
                    if log:
                        print("没有mask token，跳过扩展")
                    new_beam_candidates.append((seq, cumulative_log_prob))
                    continue

                # 获取mask位置的logits
                mask_logits = logits[mask_index]  # [num_mask, vocab_size]
                
                # 为每个mask位置采样1个候选token
                
                if alg == 'maskgit_plus':
                    token_probs, candidate_tokens = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, neg_entropy=False)
                elif alg == 'entropy':
                    token_probs, candidate_tokens = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, neg_entropy=True)
                else:
                    # 对于其他算法，使用softmax计算概率
                    probs = F.softmax(mask_logits / temperature, dim=-1)
                    token_probs, candidate_tokens = torch.max(probs, dim=-1)
                
                if len(token_probs.size()) > 1:
                    token_probs = token_probs.squeeze(-1)  # [num_mask]
                    candidate_tokens = candidate_tokens.squeeze(-1)  # [num_mask]
                
                # (1) 打印所有mask token的prob
                mask_positions = mask_index.squeeze(0).nonzero().squeeze(1)
                if log:
                    print(f"(1) 所有mask token的置信度: {token_probs.tolist()}")
                    print(f"    mask位置索引: {mask_positions.tolist()}")
                
                # 选择概率最高的位置
                k = max(min(beam_size, len(token_probs)), 1)
                top_probs, top_indices = torch.topk(token_probs, k)
                if log:
                    print(f"(2) 选择的unmask位置 (top-{k}):")
                
                # 为每个选中的位置生成新序列
                for prob, idx_in_mask in zip(top_probs, top_indices):
                    # 找到实际的mask位置
                    idx_in_mask = idx_in_mask.item()
                    actual_mask_pos = mask_index.squeeze(0).nonzero()[idx_in_mask]
                    token = candidate_tokens[idx_in_mask]
                    
                    # 生成新序列
                    new_seq = seq.clone()
                    actual_mask_pos = actual_mask_pos.item()
                    new_seq[0, actual_mask_pos] = token
                    
                    # 计算新的累计概率
                    new_log_prob = cumulative_log_prob + prob.item()
                    if log:
                        print(f"    - 位置 {actual_mask_pos}: 置信度 {prob.item():.4f}, token {token}, 新分数 {new_log_prob:.4f}")
                    new_beam_candidates.append((new_seq, new_log_prob))

            # 如果没有生成新的候选，提前结束
            if not new_beam_candidates:
                print("没有生成新的候选序列，提前结束")
                break

            # 全局排序，选择top beam_size个序列
            if log:
                print(f"候选序列数量: {len(new_beam_candidates)}")
            
            new_beam_candidates.sort(key=lambda x: x[1], reverse=True)
            uniq_new_beam_candidates = []
            seen = set()
            for tensor, float_val in new_beam_candidates:
                # 将tensor转为tuple
                if tensor.dim() == 0:  # 标量张量
                    tensor_tuple = (tensor.item(),)
                else:  # 多维张量
                    tensor_tuple = tuple(tensor.flatten().tolist())
                # 检查是否已经出现过
                if tensor_tuple not in seen:
                    seen.add(tensor_tuple)
                    uniq_new_beam_candidates.append((tensor, float_val))
            if log:
                print(f"去重后候选序列数量: {len(uniq_new_beam_candidates)}")
            
            beam = uniq_new_beam_candidates[:beam_size]
            
            # 应用token钩子
            beam = [(generation_tokens_hook_func(i, seq, None), log_prob) 
                    for seq, log_prob in beam]
            
            # (3) 打印当前最佳序列
            if beam:
                best_seq, best_score = beam[0]
                if log:
                    print(f"(3) 当前最佳序列最后128token: {best_seq[0, -128:].tolist()}")
                    print(f"    当前最佳序列分数: {best_score:.4f}")
                    print(f"    剩余mask数量: {(best_seq == mask_token_id).sum().item()}")
            else:
                if log:
                    print("(3) beam为空")
            if log:
                print()
            
            # 记录历史
            if histories is not None and beam:
                best_seq = beam[0][0]
                histories.append(best_seq.clone())

        # 选择beam中最好的序列作为最终结果
        if beam:
            best_sequence = beam[0][0]
            final_score = beam[0][1]
        else:
            best_sequence = x
            final_score = 0.0

        if log:
            print(f"=== Beam Search 生成完成 ===")
            print(f"最终序列最后128个token: {best_sequence[0, -128:].tolist()}")
            print(f"最终序列分数: {final_score:.4f}")
            print(f"最终mask数量: {(best_sequence == mask_token_id).sum().item()}")

        if return_dict_in_generate:
            return DreamModelOutput(
                sequences=best_sequence.unsqueeze(0) if best_sequence.dim() == 1 else best_sequence,
                history=histories,
            )
        else:
            return best_sequence.unsqueeze(0) if best_sequence.dim() == 1 else best_sequence

    def _sample_beam_search_two_dimension(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        generation_config: DreamGenerationConfig,
        generation_tokens_hook_func,
        generation_logits_hook_func,
        log=False
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        """改进的Beam Search采样方法 - 同时对位置和token进行搜索"""
        # 初始化参数
        output_history = generation_config.output_history
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        steps = generation_config.steps
        eps = generation_config.eps
        beam_size = 2  # 现在使用配置的beam_size
        token_beam_size = beam_size
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k

        histories = [] if (return_dict_in_generate and output_history) else None

        # 填充输入
        x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)

        if attention_mask is not None and torch.any(attention_mask == 0.0):
            attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
            tok_idx = attention_mask.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask == 0, 1)
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
        else:
            tok_idx = None
            attention_mask = "full"

        timesteps = torch.linspace(1, eps, steps + 1, device=x.device)
        
        # 初始化beam: [(sequence, cumulative_log_prob)]
        beam = [(x.clone(), 0.0)]
        beam = [(generation_tokens_hook_func(None, seq, None), log_prob) 
                for seq, log_prob in beam]
        
        if log:
            print(f"=== Beam Search 初始序列 (最后128个token) ===")
            print(f"x[-128:]: {x[0, -128:].tolist()}")
            print(f"初始mask数量: {(x == mask_token_id).sum().item()}")
            print(f"beam_size: {beam_size}, token_beam_size: {token_beam_size}")
            print()
        
        for i in range(steps):
            new_beam_candidates = []
            if log:
                print(f"=== Beam Search Step {i+1}/{steps} ===")
                print(f"当前beam大小: {len(beam)}")
                print(f"当前beam分数: {[score for _, score in beam]}")
            
            # 对beam中的每个候选序列进行扩展
            for j, (seq, cumulative_log_prob) in enumerate(beam):
                if log:
                    print(f"--- 处理beam候选 {j+1} ---")
                    print(f"当前序列分数: {cumulative_log_prob}")
                    # print(f"当前序列最后128token: {seq[0, -128:].tolist()}")
                
                with torch.no_grad():
                    logits = self(seq, attention_mask, tok_idx).logits
                    logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)

                logits = generation_logits_hook_func(i, seq, logits)

                # 找到所有mask位置
                mask_index = (seq == mask_token_id)
                
                if not mask_index.any():
                    if log:
                        print("没有mask token，跳过扩展")
                    new_beam_candidates.append((seq, cumulative_log_prob))
                    continue

                # 获取mask位置的logits
                mask_logits = logits[mask_index]  # [num_mask, vocab_size]
                
                # 为每个mask位置采样多个候选token
                token_confidences, candidate_tokens = sample_tokens_with_beam(
                    mask_logits, 
                    temperature=temperature, 
                    top_p=top_p, 
                    top_k=top_k,
                    beam_size=token_beam_size  # 每个位置采样多个token
                )
                
                # token_confidences: [num_mask, token_beam_size]
                # candidate_tokens: [num_mask, token_beam_size]
                
                mask_positions = mask_index.squeeze(0).nonzero().squeeze(1)
                if log:
                    print(f"(1) 每个mask位置有 {token_beam_size} 个候选token")
                    print(f"    mask位置索引: {mask_positions.tolist()}")
                # 替换原来的双重循环
                num_masks = mask_logits.shape[0]

                # 将置信度展平，然后取top-k
                all_confidences = token_confidences.view(-1)  # [num_masks * token_beam_size]
                all_tokens = candidate_tokens.view(-1)        # [num_masks * token_beam_size]

                # 计算每个候选对应的mask位置索引
                mask_indices = torch.arange(num_masks, device=token_confidences.device)
                mask_indices = mask_indices.unsqueeze(1).expand(-1, token_beam_size).reshape(-1)  # [num_masks * token_beam_size]

                # 选择top-k个候选
                k = min(beam_size * token_beam_size, len(all_confidences))
                top_confidences, top_indices = torch.topk(all_confidences, k)
                def stable_sort_topk(confidences, indices, k):
                    """
                    对topk结果进行稳定排序：先按置信度降序，置信度相同时按索引升序
                    """
                    # 将结果转为list方便排序
                    paired = [(conf.item(), idx.item()) for conf, idx in zip(confidences, indices)]
                    
                    # 稳定排序：先按置信度降序，再按索引升序
                    paired_sorted = sorted(paired, key=lambda x: (-x[0], x[1]))
                    
                    # 取前k个
                    paired_sorted = paired_sorted[:k]
                    
                    # 重新转为tensor
                    sorted_confidences = torch.tensor([x[0] for x in paired_sorted], device=confidences.device)
                    sorted_indices = torch.tensor([x[1] for x in paired_sorted], device=indices.device)
                    
                    return sorted_confidences, sorted_indices
                # 稳定排序
                top_confidences, top_indices = stable_sort_topk(top_confidences, top_indices, k)
                for idx in range(k):
                    flat_idx = top_indices[idx].item()
                    confidence = top_confidences[idx].item()
                    token = all_tokens[flat_idx].item()
                    mask_idx = mask_indices[flat_idx].item()
                    actual_mask_pos = mask_positions[mask_idx].item()
                    
                    # 生成新序列
                    new_seq = seq.clone()
                    new_seq[0, actual_mask_pos] = token
                    
                    # 计算新的累计概率
                    new_log_prob = cumulative_log_prob + confidence
                    
                    if log and idx < 3:
                        print(f"    - 位置 {actual_mask_pos}, token {token}, "
                            f"置信度 {confidence:.4f}, 新分数 {new_log_prob:.4f}")
                    
                    new_beam_candidates.append((new_seq, new_log_prob))

            if log:
                print(f"count of new_beam_candidates: {len(new_beam_candidates)}")
            # 如果没有生成新的候选，提前结束
            if not new_beam_candidates:
                print("没有生成新的候选序列，提前结束")
                break

            # 全局排序，选择top beam_size个序列
            if log:
                print(f"候选序列总数: {len(new_beam_candidates)}")
            
            new_beam_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # 去重
            uniq_new_beam_candidates = []
            seen = set()
            for tensor, float_val in new_beam_candidates:
                if tensor.dim() == 0:
                    tensor_tuple = (tensor.item(),)
                else:
                    tensor_tuple = tuple(tensor.flatten().tolist())
                if tensor_tuple not in seen:
                    seen.add(tensor_tuple)
                    uniq_new_beam_candidates.append((tensor, float_val))
            
            if log:
                print(f"去重后候选序列数量: {len(uniq_new_beam_candidates)}")
            
            beam = uniq_new_beam_candidates[:beam_size]
            
            # 应用token钩子
            beam = [(generation_tokens_hook_func(i, seq, None), log_prob) 
                    for seq, log_prob in beam]
            
            # 打印当前最佳序列
            if beam:
                best_seq, best_score = beam[0]
                if log:
                    print(f"(3) 当前最佳序列最后128token: {best_seq[0, -128:].tolist()}")
                    print(f"    当前最佳序列分数: {best_score:.4f}")
                    print(f"    剩余mask数量: {(best_seq == mask_token_id).sum().item()}")
            else:
                if log:
                    print("(3) beam为空")
            if log:
                print()
            
            # 记录历史
            if histories is not None and beam:
                best_seq = beam[0][0]
                histories.append(best_seq.clone())

        # 选择beam中最好的序列作为最终结果
        if beam:
            best_sequence = beam[0][0]
            final_score = beam[0][1]
        else:
            best_sequence = x
            final_score = 0.0

        if log:
            print(f"=== Beam Search 生成完成 ===")
            print(f"最终序列最后128个token: {best_sequence[0, -128:].tolist()}")
            print(f"最终序列分数: {final_score:.4f}")
            print(f"最终mask数量: {(best_sequence == mask_token_id).sum().item()}")

        if return_dict_in_generate:
            return DreamModelOutput(
                sequences=best_sequence.unsqueeze(0) if best_sequence.dim() == 1 else best_sequence,
                history=histories,
            )
        else:
            return best_sequence.unsqueeze(0) if best_sequence.dim() == 1 else best_sequence

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
        # init values
        output_history = generation_config.output_history
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        steps = generation_config.steps
        eps = generation_config.eps
        alg = generation_config.alg
        alg = "maskgit_plus"
        alg_temp = generation_config.alg_temp
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k

        histories = [] if (return_dict_in_generate and output_history) else None
        
        # 新增：用于记录Unmask步骤的列表
        unmask_records = []

        print(f"========原始解码方式: alg:{alg}, temperature:{temperature}, steps:{steps}, max_length:{max_length}==========", flush=True)
        # pad input_ids to max_length
        x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)

        if attention_mask is not None and torch.any(attention_mask == 0.0):
            # we do not mask the [MASK] tokens so value = 1.0
            attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
            tok_idx = attention_mask.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask == 0, 1)
            # attention_mask is of shape [B, N]
            # broadcast to [B, 1, N, N]
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
        else:
            tok_idx = None
            attention_mask = "full"

        timesteps = torch.linspace(1, eps, steps + 1, device=x.device)

        # this allows user-defined token control of the intermediate steps
        x = generation_tokens_hook_func(None, x, None)
        for i in range(steps):
            mask_index = (x == mask_token_id)
            logits = self(x, attention_mask, tok_idx).logits
            logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)

            # this allows user-defined logits control of the intermediate steps
            logits = generation_logits_hook_func(i, x, logits)

            mask_logits = logits[mask_index]
            t = timesteps[i]
            s = timesteps[i + 1]
        
            if alg == 'origin':
                p_transfer = 1 - s / t if i < steps - 1 else 1
                x0 = torch.zeros_like(x[mask_index], device=self.device, dtype=torch.long) + mask_token_id
                transfer_index_t_s = torch.rand(*x0.shape, device=self.device) < p_transfer
                
                # 记录origin算法的Unmask信息
                    # 获取要转移的位置
                transfer_positions = torch.where(transfer_index_t_s)[0]
                for pos_idx in transfer_positions:
                    # 找到全局位置（需要从mask_index中映射）
                    global_pos = torch.where(mask_index)[0][pos_idx].item() if mask_index.sum() > 0 else -1
                    token_id = x0[pos_idx].item()
                    # 计算该位置的置信度（softmax最大值）
                    token_probs = F.softmax(mask_logits[pos_idx], dim=-1)
                    confidence = token_probs.max().item()
                    token_str = tokenizer.decode([token_id]) if tokenizer else str(token_id)
                    
                    unmask_records.append({
                        "step": i + 1,  # 添加step，从1开始
                        "position": global_pos,
                        "confidence": confidence,
                        "token_id": token_id,
                        "token_str": token_str
                    })
                
                _, x0[transfer_index_t_s] = sample_tokens(mask_logits[transfer_index_t_s], temperature=temperature, top_p=top_p, top_k=top_k)
                x[mask_index] = x0.clone()
            else:
                if alg == 'maskgit_plus':
                    confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, log_step=i)
                elif alg == 'topk_margin':
                    confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, margin_confidence=True)
                elif alg == 'entropy':
                    confidence, x0 = sample_tokens(mask_logits, temperature, top_p=top_p, top_k=top_k, neg_entropy=True)
                else:
                    raise RuntimeError(f"Unknown alg: {alg}")
                

                num_mask_token = mask_index.sum() / mask_index.shape[0]
                number_transfer_tokens = int(num_mask_token * (1 - s / t)) if i < steps - 1 else int(num_mask_token)
                # full confidence包含prompt, confidence不包含prompt
                full_confidence = torch.full_like(x, -torch.inf, device=self.device, dtype=logits.dtype)
                full_confidence[mask_index] = confidence
                
                # 记录置信度算法的Unmask信息
                if number_transfer_tokens > 0:
                    if alg_temp is None or alg_temp == 0:
                        _, transfer_index = torch.topk(full_confidence, number_transfer_tokens)
                    else:
                        full_confidence = full_confidence / alg_temp
                        full_confidence = F.softmax(full_confidence, dim=-1)
                        transfer_index = torch.multinomial(full_confidence, num_samples=number_transfer_tokens)
                    
                    x_ = torch.zeros_like(x, device=self.device, dtype=torch.long) + mask_token_id
                    x_[mask_index] = x0.clone()
                    row_indices = torch.arange(x.size(0), device=self.device).unsqueeze(1).expand_as(transfer_index)
                    
                    # 记录每个被unmask的位置信息
                    for batch_idx in range(x.size(0)):
                        for col in range(number_transfer_tokens):
                            pos = transfer_index[batch_idx, col].item()
                            generate_pos = pos - input_ids.size()[1]
                            token_id = x_[batch_idx, pos].item()
                            # print(full_confidence.size())
                            conf = full_confidence[batch_idx][pos].item()
                            token_str = tokenizer.decode([token_id]) if tokenizer else str(token_id)
                            
                            unmask_records.append({
                                "step": i + 1,  # 添加step，从1开始
                                "position": pos,
                                "confidence": conf,
                                "token_id": token_id,
                                "token_str": token_str
                            })
                    
                    x[row_indices, transfer_index] = x_[row_indices, transfer_index]

            # this allows user-defined token control of the intermediate steps
            x = generation_tokens_hook_func(i, x, logits)

            if histories is not None:
                histories.append(x.clone())
        import json
        print(json.dumps(unmask_records), flush=True)
        if return_dict_in_generate:
            return DreamModelOutput(
                sequences=x,
                history=histories,
                # unmask_records=unmask_records  # 新增：返回记录信息
            )
        else:
            # 如果不返回dict，我们可以将records附加到其他返回值中
            # 但根据你的需要，这里可能需要调整返回值结构
            return x
    
    def _sample_beam_search_cache_batch(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        generation_config: DreamGenerationConfig,
        generation_tokens_hook_func,
        generation_logits_hook_func,
        log=False
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        """正确的Beam Search采样方法 - 选择概率最高的位置进行unmask"""
        # 初始化参数
        output_history = generation_config.output_history
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        steps = generation_config.steps
        eps = generation_config.eps
        alg = generation_config.alg
        alg_temp = generation_config.alg_temp
        # beam_size = generation_config.beam_size
        beam_size = 2
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k

        print(f"========current method: Beam Search PRUNE+CACHE, algo: {alg}, beam size: {beam_size}==========", flush=True)
        histories = [] if (return_dict_in_generate and output_history) else None

        # 填充输入
        x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)

        if attention_mask is not None and torch.any(attention_mask == 0.0):
            attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
            tok_idx = attention_mask.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask == 0, 1)
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
        else:
            tok_idx = None
            attention_mask = "full"

        timesteps = torch.linspace(1, eps, steps + 1, device=x.device)
        
        # 初始化beam: [(sequence, cumulative_log_prob)]
        beam = [(x.clone(), 0.0)]
        beam = [(generation_tokens_hook_func(None, seq, None), log_prob) 
                for seq, log_prob in beam]
        if log:
            print(f"=== Beam Search 初始序列 (最后128个token) ===")
            print(f"x[-128:]: {x[0, -128:].tolist()}")
            print(f"初始mask数量: {(x == mask_token_id).sum().item()}")
            print(f"beam_size: {beam_size}")
            print()
        
        for i in range(steps):
            new_beam_candidates = []
            if log:
                print(f"=== Beam Search Step {i+1}/{steps} ===")
                print(f"当前beam大小: {len(beam)}")
                print(f"当前beam分数: {[score for _, score in beam]}")
            

            beam_sequences = []
            beam_indices = []  # 记录每个beam在batch中的位置

            for j, (seq, cumulative_log_prob) in enumerate(beam):
                beam_sequences.append(seq)
                beam_indices.append(j)

            # 将所有序列拼接成一个batch
            batch_sequences = torch.cat(beam_sequences, dim=0)  # (num_beams, seq_len)
            
            # 扩展attention_mask以匹配batch_size
            if attention_mask == "full":
                batch_attention_mask = "full"
            elif attention_mask is not None:
                batch_attention_mask = attention_mask.expand(len(beam_sequences), -1)  # (num_beams, seq_len)
            else:
                batch_attention_mask = None
            
            # 一次性计算所有beam的logits
            with torch.no_grad():
                # batch_logits = self(batch_sequences, batch_attention_mask, tok_idx).logits  # (num_beams, seq_len, vocab_size)
                batch_logits = self(batch_sequences, batch_attention_mask, tok_idx, shared_prefix_length=input_ids.shape[1]).logits
                batch_logits = torch.cat([batch_logits[:,:1], batch_logits[:, :-1]], dim=1)
                
                # 现在在beam循环中直接使用预计算的logits
            for j, (seq, cumulative_log_prob) in enumerate(beam):
                if log:
                    print(f"--- 处理beam候选 {j+1} ---")
                    print(f"当前序列分数: {cumulative_log_prob}")
                    print(f"当前序列最后128token: {seq[0, -128:].tolist()}")
                
                # 直接从batch_logits中取对应的logits
                logits = batch_logits[j:j+1]  # (1, seq_len, vocab_size)

                logits = generation_logits_hook_func(i, seq, logits)

                # 找到所有mask位置
                mask_index = (seq == mask_token_id)
                
                if not mask_index.any():
                    if log:
                        print("没有mask token，跳过扩展")
                    new_beam_candidates.append((seq, cumulative_log_prob))
                    continue

                # 获取mask位置的logits
                mask_logits = logits[mask_index]  # [num_mask, vocab_size]
                
                # 为每个mask位置采样1个候选token
                
                if alg == 'maskgit_plus':
                    token_probs, candidate_tokens = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, neg_entropy=False)
                elif alg == 'entropy':
                    token_probs, candidate_tokens = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, neg_entropy=True)
                else:
                    # 对于其他算法，使用softmax计算概率
                    probs = F.softmax(mask_logits / temperature, dim=-1)
                    token_probs, candidate_tokens = torch.max(probs, dim=-1)
                
                if len(token_probs.size()) > 1:
                    token_probs = token_probs.squeeze(-1)  # [num_mask]
                    candidate_tokens = candidate_tokens.squeeze(-1)  # [num_mask]
                
                # (1) 打印所有mask token的prob
                mask_positions = mask_index.squeeze(0).nonzero().squeeze(1)
                if log:
                    print(f"(1) 所有mask token的置信度: {token_probs.tolist()}")
                    print(f"    mask位置索引: {mask_positions.tolist()}")
                
                # 选择概率最高的位置
                k = max(min(beam_size, len(token_probs)), 1)
                top_probs, top_indices = torch.topk(token_probs, k)
                if log:
                    print(f"(2) 选择的unmask位置 (top-{k}):")
                
                # 为每个选中的位置生成新序列
                for prob, idx_in_mask in zip(top_probs, top_indices):
                    # 找到实际的mask位置
                    idx_in_mask = idx_in_mask.item()
                    actual_mask_pos = mask_index.squeeze(0).nonzero()[idx_in_mask]
                    token = candidate_tokens[idx_in_mask]
                    
                    # 生成新序列
                    new_seq = seq.clone()
                    actual_mask_pos = actual_mask_pos.item()
                    new_seq[0, actual_mask_pos] = token
                    
                    # 计算新的累计概率
                    new_log_prob = cumulative_log_prob + prob.item()
                    if log:
                        print(f"    - 位置 {actual_mask_pos}: 置信度 {prob.item():.4f}, token {token}, 新分数 {new_log_prob:.4f}")
                    new_beam_candidates.append((new_seq, new_log_prob))

            # 如果没有生成新的候选，提前结束
            if not new_beam_candidates:
                print("没有生成新的候选序列，提前结束")
                break

            # 全局排序，选择top beam_size个序列
            if log:
                print(f"候选序列数量: {len(new_beam_candidates)}")
            
            new_beam_candidates.sort(key=lambda x: x[1], reverse=True)
            uniq_new_beam_candidates = []
            seen = set()
            for tensor, float_val in new_beam_candidates:
                # 将tensor转为tuple
                if tensor.dim() == 0:  # 标量张量
                    tensor_tuple = (tensor.item(),)
                else:  # 多维张量
                    tensor_tuple = tuple(tensor.flatten().tolist())
                # 检查是否已经出现过
                if tensor_tuple not in seen:
                    seen.add(tensor_tuple)
                    uniq_new_beam_candidates.append((tensor, float_val))
            if log:
                print(f"去重后候选序列数量: {len(uniq_new_beam_candidates)}")
            
            uniq_new_beam_candidates_uniq_score = uniq_new_beam_candidates[:beam_size]
            beam = [uniq_new_beam_candidates_uniq_score[0]]
            if len(uniq_new_beam_candidates_uniq_score) > 1:
                for candidate in uniq_new_beam_candidates_uniq_score[1:]:
                    if candidate[1] != beam[-1][1]:
                        beam.append(candidate)
            
            # beam = uniq_new_beam_candidates[:beam_size]
            
            # 应用token钩子
            beam = [(generation_tokens_hook_func(i, seq, None), log_prob) 
                    for seq, log_prob in beam]
            
            # (3) 打印当前最佳序列
            if beam:
                best_seq, best_score = beam[0]
                if log:
                    print(f"(3) 当前最佳序列最后128token: {best_seq[0, -128:].tolist()}")
                    print(f"    当前最佳序列分数: {best_score:.4f}")
                    print(f"    剩余mask数量: {(best_seq == mask_token_id).sum().item()}")
            else:
                if log:
                    print("(3) beam为空")
            if log:
                print()
            
            # 记录历史
            if histories is not None and beam:
                best_seq = beam[0][0]
                histories.append(best_seq.clone())

        # 选择beam中最好的序列作为最终结果
        if beam:
            best_sequence = beam[0][0]
            final_score = beam[0][1]
        else:
            best_sequence = x
            final_score = 0.0

        if log:
            print(f"=== Beam Search 生成完成 ===")
            print(f"最终序列最后128个token: {best_sequence[0, -128:].tolist()}")
            print(f"最终序列分数: {final_score:.4f}")
            print(f"最终mask数量: {(best_sequence == mask_token_id).sum().item()}")

        if return_dict_in_generate:
            return DreamModelOutput(
                sequences=best_sequence.unsqueeze(0) if best_sequence.dim() == 1 else best_sequence,
                history=histories,
            )
        else:
            return best_sequence.unsqueeze(0) if best_sequence.dim() == 1 else best_sequence


    def _sample_l2r_casual_mask(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        generation_config: DreamGenerationConfig,
        generation_tokens_hook_func,
        generation_logits_hook_func,
        tokenizer=None,
        log=False
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        # init values
        output_history = generation_config.output_history
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        steps = generation_config.steps
        eps = generation_config.eps
        alg = generation_config.alg
        alg_temp = generation_config.alg_temp
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k

        histories = [] if (return_dict_in_generate and output_history) else None

        # pad input_ids to max_length
        x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)

        # 创建因果注意力掩码（只关注前面token）
        batch_size, seq_len = x.shape
        causal_attention_mask = torch.tril(torch.ones((batch_size, 1, seq_len, seq_len), device=x.device))
        
        if attention_mask is not None and torch.any(attention_mask == 0.0):
            # we do not mask the [MASK] tokens so value = 1.0
            attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
            tok_idx = attention_mask.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask == 0, 1)
            # 结合因果掩码和输入掩码
            input_attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
            # 应用因果掩码
            attention_mask = torch.logical_and(input_attention_mask, causal_attention_mask)
        else:
            tok_idx = None
            attention_mask = causal_attention_mask  # 使用纯因果掩码

        timesteps = torch.linspace(1, eps, steps + 1, device=x.device)

        # this allows user-defined token control of the intermediate steps
        x = generation_tokens_hook_func(None, x, None)
        
        for i in range(steps):
            # 只考虑最左边的[MASK] token
            mask_positions = (x == mask_token_id)
            
            # 创建从左到右的mask：只暴露第一个[MASK]位置
            left_to_right_mask = torch.zeros_like(x, dtype=torch.bool)
            
            # 对于每个序列，找到第一个[MASK]的位置
            for batch_idx in range(x.shape[0]):
                mask_indices = torch.where(mask_positions[batch_idx])[0]
                if len(mask_indices) > 0:
                    first_mask_idx = mask_indices[0]  # 最左边的[MASK]
                    left_to_right_mask[batch_idx, first_mask_idx] = True
            
            # 使用因果注意力掩码进行前向传播
            logits = self(x, attention_mask, tok_idx).logits
            logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)

            # this allows user-defined logits control of the intermediate steps
            logits = generation_logits_hook_func(i, x, logits)

            # 只处理最左边的[MASK]位置的logits
            mask_logits = logits[left_to_right_mask]
            
            t = timesteps[i]
            s = timesteps[i + 1]
        
            if alg == 'origin':
                # 对于origin算法，强制填充最左边的[MASK]
                p_transfer = 1 - s / t if i < steps - 1 else 1
                if mask_logits.numel() > 0:
                    x0 = torch.zeros(mask_logits.shape[0], device=self.device, dtype=torch.long) + mask_token_id
                    transfer_index_t_s = torch.rand(x0.shape, device=self.device) < p_transfer
                    
                    # 采样token
                    _, sampled_tokens = sample_tokens(mask_logits[transfer_index_t_s], 
                                                    temperature=temperature, top_p=top_p, top_k=top_k)
                    
                    # 更新最左边的[MASK]位置
                    batch_mask_indices = torch.where(left_to_right_mask)
                    if len(batch_mask_indices[0]) > 0:
                        for idx in range(len(batch_mask_indices[0])):
                            batch_idx = batch_mask_indices[0][idx]
                            pos_idx = batch_mask_indices[1][idx]
                            if idx < len(transfer_index_t_s) and transfer_index_t_s[idx]:
                                if idx < len(sampled_tokens):
                                    x[batch_idx, pos_idx] = sampled_tokens[idx]
                            
            else:
                if mask_logits.numel() > 0:
                    if alg == 'maskgit_plus':
                        confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k)
                    elif alg == 'topk_margin':
                        confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, margin_confidence=True)
                    elif alg == 'entropy':
                        confidence, x0 = sample_tokens(mask_logits, temperature, top_p=top_p, top_k=top_k, neg_entropy=True)
                    else:
                        raise RuntimeError(f"Unknown alg: {alg}")
                    
                    # 对于非origin算法，总是填充最左边的[MASK]
                    batch_mask_indices = torch.where(left_to_right_mask)
                    if len(batch_mask_indices[0]) > 0:
                        for idx in range(len(batch_mask_indices[0])):
                            batch_idx = batch_mask_indices[0][idx]
                            pos_idx = batch_mask_indices[1][idx]
                            if idx < len(x0):
                                x[batch_idx, pos_idx] = x0[idx]

            # this allows user-defined token control of the intermediate steps
            x = generation_tokens_hook_func(i, x, logits)

            if histories is not None:
                histories.append(x.clone())
        
        if return_dict_in_generate:
            return DreamModelOutput(
                sequences=x,
                history=histories,
            )
        else:
            return x

    def _sample_l2r(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        generation_config: DreamGenerationConfig,
        generation_tokens_hook_func,
        generation_logits_hook_func,
        tokenizer=None,
        log=False
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        # init values
        output_history = generation_config.output_history
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        steps = generation_config.steps
        eps = generation_config.eps
        alg = generation_config.alg
        alg_temp = generation_config.alg_temp
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k

        histories = [] if (return_dict_in_generate and output_history) else None

        # pad input_ids to max_length
        x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)

        if attention_mask is not None and torch.any(attention_mask == 0.0):
            # we do not mask the [MASK] tokens so value = 1.0
            attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
            tok_idx = attention_mask.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask == 0, 1)
            # attention_mask is of shape [B, N]
            # broadcast to [B, 1, N, N]
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
        else:
            tok_idx = None
            attention_mask = "full"

        timesteps = torch.linspace(1, eps, steps + 1, device=x.device)

        # this allows user-defined token control of the intermediate steps
        x = generation_tokens_hook_func(None, x, None)
        
        for i in range(steps):
            # 只考虑最左边的[MASK] token
            mask_positions = (x == mask_token_id)
            
            # 创建从左到右的mask：只暴露第一个[MASK]位置
            left_to_right_mask = torch.zeros_like(x, dtype=torch.bool)
            
            # 对于每个序列，找到第一个[MASK]的位置
            for batch_idx in range(x.shape[0]):
                mask_indices = torch.where(mask_positions[batch_idx])[0]
                if len(mask_indices) > 0:
                    first_mask_idx = mask_indices[0]  # 最左边的[MASK]
                    left_to_right_mask[batch_idx, first_mask_idx] = True
            
            # 只对第一个[MASK]位置计算logits
            logits = self(x, attention_mask, tok_idx).logits
            logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)

            # this allows user-defined logits control of the intermediate steps
            logits = generation_logits_hook_func(i, x, logits)

            # 只处理最左边的[MASK]位置的logits
            mask_logits = logits[left_to_right_mask]
            
            t = timesteps[i]
            s = timesteps[i + 1]
        
            if alg == 'origin':
                # 对于origin算法，强制填充最左边的[MASK]
                p_transfer = 1 - s / t if i < steps - 1 else 1
                x0 = torch.zeros_like(mask_logits.shape[0], device=self.device, dtype=torch.long) + mask_token_id
                transfer_index_t_s = torch.rand(x0.shape, device=self.device) < p_transfer
                
                # 采样token
                _, sampled_tokens = sample_tokens(mask_logits[transfer_index_t_s], 
                                                temperature=temperature, top_p=top_p, top_k=top_k)
                
                # 更新最左边的[MASK]位置
                for batch_idx in range(x.shape[0]):
                    if left_to_right_mask[batch_idx].any():
                        first_mask_idx = torch.where(left_to_right_mask[batch_idx])[0][0]
                        if transfer_index_t_s[batch_idx] if transfer_index_t_s.dim() > 0 else transfer_index_t_s:
                            x[batch_idx, first_mask_idx] = sampled_tokens[batch_idx] if sampled_tokens.dim() > 0 else sampled_tokens
                            
            else:
                if alg == 'maskgit_plus':
                    confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k)
                elif alg == 'topk_margin':
                    confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, margin_confidence=True)
                elif alg == 'entropy':
                    confidence, x0 = sample_tokens(mask_logits, temperature, top_p=top_p, top_k=top_k, neg_entropy=True)
                else:
                    raise RuntimeError(f"Unknown alg: {alg}")
                
                # 对于非origin算法，总是填充最左边的[MASK]
                num_left_mask = left_to_right_mask.sum().item()
                if num_left_mask > 0:
                    # 直接使用采样的token填充最左边的[MASK]
                    for batch_idx in range(x.shape[0]):
                        if left_to_right_mask[batch_idx].any():
                            first_mask_idx = torch.where(left_to_right_mask[batch_idx])[0][0]
                            # 获取对应的采样token
                            batch_mask_count = left_to_right_mask[:batch_idx+1].sum() - 1
                            token_idx = batch_mask_count if batch_mask_count < len(x0) else -1
                            x[batch_idx, first_mask_idx] = x0[token_idx]

            # this allows user-defined token control of the intermediate steps
            x = generation_tokens_hook_func(i, x, logits)

            if histories is not None:
                histories.append(x.clone())
        
        if return_dict_in_generate:
            return DreamModelOutput(
                sequences=x,
                history=histories,
            )
        else:
            return x