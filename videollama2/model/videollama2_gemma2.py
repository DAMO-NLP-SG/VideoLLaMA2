# Adopted from: https://github.com/haotian-liu/LLaVA. Below is the original copyright:
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         Gemma2Config, Gemma2Model, Gemma2ForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from .videollama2_arch import Videollama2MetaModel, Videollama2MetaForCausalLM


class Videollama2Gemma2Config(Gemma2Config):
    model_type = "videollama2_gemma2"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = "videollama2_gemma2"


class Videollama2Gemma2Model(Videollama2MetaModel, Gemma2Model):
    config_class = Videollama2Gemma2Config

    def __init__(self, config: Gemma2Config):
        super(Videollama2Gemma2Model, self).__init__(config)


class Videollama2Gemma2ForCausalLM(Gemma2ForCausalLM, Videollama2MetaForCausalLM):
    config_class = Videollama2Gemma2Config

    def __init__(self, config, **kwargs):
        super(Gemma2ForCausalLM, self).__init__(config)
        self.model = Videollama2Gemma2Model(config)
        # self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[int] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                attention_mask,
                past_key_values,
                labels,
                images
            )

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        outputs.labels = labels

        return outputs

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                input_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids=inputs,
                attention_mask=attention_mask,
                past_key_values=None,
                labels=None,
                images=images
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def _prepare_generated_length(self, model_input_name, inputs_tensor, **kwargs):
        if model_input_name == "inputs_embeds":
            self.inputs_embeds_length = inputs_tensor.size(1)
        else:
            self.inputs_embeds_length = 0
        return super()._prepare_generated_length(
            model_input_name=model_input_name, 
            inputs_tensor=inputs_tensor, 
            **kwargs)

    def _get_cache(self, cache_implementation: str, max_batch_size: int, max_cache_len: int, **kwargs):
        return super()._get_cache(
            cache_implementation=cache_implementation,
            max_batch_size=max_batch_size,
            max_cache_len=max_cache_len + self.inputs_embeds_length,
            **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        return _inputs


AutoConfig.register("videollama2_gemma2", Videollama2Gemma2Config)
AutoModelForCausalLM.register(Videollama2Gemma2Config, Videollama2Gemma2ForCausalLM)
