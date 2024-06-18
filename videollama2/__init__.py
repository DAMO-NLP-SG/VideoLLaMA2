import copy
from functools import partial

import torch

from .model import Videollama2LlamaForCausalLM, Videollama2MistralForCausalLM
from .model.builder import load_pretrained_model
from .conversation import conv_templates, SeparatorStyle
from .mm_utils import process_video, tokenizer_MMODAL_token, get_model_name_from_path, KeywordsStoppingCriteria
from .constants import NUM_FRAMES, DEFAULT_MMODAL_TOKEN, DEFAULT_MMODAL_START_TOKEN, DEFAULT_MMODAL_END_TOKEN, MMODAL_TOKEN_INDEX


def model_init(model_path=None):
    model_path = "DAMO-NLP-SG/VideoLLaMA2-7B" if model_path is None else model_path
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, None, model_name)

    num_frames = model.config.num_frames if hasattr(model.config, "num_frames") else NUM_FRAMES

    return model, partial(process_video, aspect_ratio=None, processor=processor, num_frames=num_frames), tokenizer


def infer(model, video, instruct, tokenizer, do_sample=False):
    """inference api of VideoLLaMA2 for video understanding.

    Args:
        model: VideoLLaMA2 model.
        video (torch.Tensor): video tensor (T, C, H, W).
        instruct (str): text instruction for understanding video.
        tokenizer: tokenizer.
        do_sample (bool): whether to sample.
    Returns:
        str: response of the model.
    """

    # 1. vision preprocess (load & transform image or video).
    tensor = [video.half().cuda()]
    modals = ["video"]

    # 2. text preprocess (tag process & generate prompt).
    modal_token = DEFAULT_MMODAL_TOKEN['VIDEO']
    modal_index = MMODAL_TOKEN_INDEX["VIDEO"]
    instruct = modal_token + '\n' + instruct

    conv = conv_templates["llama_2"].copy()
    conv.append_message(conv.roles[0], instruct)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_MMODAL_token(prompt, tokenizer, modal_index, return_tensors='pt').unsqueeze(0).cuda()
    attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

    # 3. generate response according to visual signals and prompts. 
    stop_str = conv.sep if conv.sep_style in [SeparatorStyle.SINGLE] else conv.sep2
    # keywords = ["<s>", "</s>"]
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_masks,
            images_or_videos=tensor,
            modal_list=modals,
            do_sample=do_sample,
            temperature=0.2 if do_sample else 0.0,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            pad_token_id=tokenizer.eos_token_id,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    return outputs


def x_infer(video, question, model, tokenizer, mode='vanilla', do_sample=False):
    if mode == 'mcqa':
        instruction = f'{question}\nAnswer with the option\'s letter from the given choices directly and only give the best option.'
        return infer(model=model, tokenizer=tokenizer, video=video, instruct=instruction, do_sample=do_sample)
    elif mode == 'openend':
        instruction = f'{question}\nAnswer the question using a single word or a short phrase with multiple words.'
        return infer(model=model, tokenizer=tokenizer, video=video, instruct=instruction, do_sample=do_sample)
    elif mode == 'vanilla':
        instruction = question
        return infer(model=model, tokenizer=tokenizer, video=video, instruct=instruction, do_sample=do_sample)
