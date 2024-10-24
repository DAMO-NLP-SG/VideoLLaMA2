import os
import copy
import warnings
import shutil
from functools import partial

import torch

from .model import load_pretrained_model
from .mm_utils import process_image, process_video, tokenizer_multimodal_token, get_model_name_from_path, KeywordsStoppingCriteria, process_audio_file
from .constants import NUM_FRAMES, DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN, MODAL_INDEX_MAP, DEFAULT_AUDIO_TOKEN


def model_init(model_path=None, **kwargs):
    model_path = "DAMO-NLP-SG/VideoLLaMA2-7B" if model_path is None else model_path
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, None, model_name, **kwargs)

    if tokenizer.pad_token is None and tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token

    num_frames = model.config.num_frames if hasattr(model.config, "num_frames") else NUM_FRAMES
    processor = {
        'image': partial(process_image, processor=processor, aspect_ratio=None),
        'video': partial(process_video, processor=processor, aspect_ratio=None, num_frames=num_frames),
        'audio': process_audio_file,
    }

    return model, processor, tokenizer


def mm_infer(image_or_video, instruct, model, tokenizer, modal='video', **kwargs):
    """inference api of VideoLLaMA2 for video understanding.

    Args:
        model: VideoLLaMA2 model.
        image_or_video (torch.Tensor): image tensor (1, C, H, W) / video tensor (T, C, H, W).
        instruct (str): text instruction for understanding video.
        tokenizer: tokenizer.
        do_sample (bool): whether to sample.
        modal (str): inference modality.
    Returns:
        str: response of the model.
    """

    # 1. text preprocess (tag process & generate prompt).
    if modal == 'image':
        modal_token = DEFAULT_IMAGE_TOKEN
    elif modal == 'video':
        modal_token = DEFAULT_VIDEO_TOKEN
    elif modal == 'text':
        modal_token = ''
    elif modal == 'audio':
        modal_token = DEFAULT_AUDIO_TOKEN
    else:
        raise ValueError(f"Unsupported modal: {modal}")

    # 1. vision preprocess (load & transform image or video).
    if modal == 'text':
        tensor = None
    else:
        if isinstance(image_or_video, dict):
            tensor = {k: v.half().cuda() for k, v in image_or_video.items()}
        else:
            tensor = image_or_video.half().cuda() 
        tensor = [(tensor, modal)]

    # 2. text preprocess (tag process & generate prompt).
    if isinstance(instruct, str):
        message = [{'role': 'user', 'content': modal_token + '\n' + instruct}]
    elif isinstance(instruct, list):
        message = copy.deepcopy(instruct)
        message[0]['content'] = modal_token + '\n' + message[0]['content']
    else:
        raise ValueError(f"Unsupported type of instruct: {type(instruct)}")

    if model.config.model_type in ['videollama2', 'videollama2_mistral', 'videollama2_mixtral']:
        system_message = [
            {'role': 'system', 'content': (
            """<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."""
            """\n"""
            """If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>""")
            }
        ]
    else:
        system_message = []

    message = system_message + message
    prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

    input_ids = tokenizer_multimodal_token(prompt, tokenizer, modal_token, return_tensors='pt').unsqueeze(0).long().cuda()
    attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

    # 3. generate response according to visual signals and prompts. 
    keywords = [tokenizer.eos_token]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    do_sample = kwargs.get('do_sample', False)
    temperature = kwargs.get('temperature', 0.2 if do_sample else 0.0)
    top_p = kwargs.get('top_p', 0.9)
    max_new_tokens = kwargs.get('max_new_tokens', 2048)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_masks,
            images=tensor,
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            pad_token_id=tokenizer.eos_token_id,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    return outputs
