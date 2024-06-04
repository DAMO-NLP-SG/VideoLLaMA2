import math
import os
import argparse
import json
import warnings
from tqdm import tqdm

import torch
import numpy as np
import transformers
import decord
from decord import VideoReader, cpu

import sys
sys.path.append('./')
from videollama2.conversation import conv_templates, SeparatorStyle
from videollama2.constants import NUM_FRAMES, DEFAULT_MMODAL_TOKEN, DEFAULT_MMODAL_START_TOKEN, DEFAULT_MMODAL_END_TOKEN, MMODAL_TOKEN_INDEX
from videollama2.mm_utils import get_model_name_from_path, tokenizer_MMODAL_token, KeywordsStoppingCriteria, process_video
from videollama2.model.builder import load_pretrained_model


# NOTE: Ignore TypedStorage warning, which refers to this link~(https://github.com/pytorch/pytorch/issues/97207#issuecomment-1494781560)
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

default_mm_token = DEFAULT_MMODAL_TOKEN["VIDEO"]
default_mm_start_token =  DEFAULT_MMODAL_START_TOKEN["VIDEO"]
default_mm_end_token = DEFAULT_MMODAL_END_TOKEN["VIDEO"]
modal_token_index = MMODAL_TOKEN_INDEX["VIDEO"]


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def get_model_output(model, tokenizer, video_tensor, questions, conv_mode="v1", device='cuda'):

    input_ids = []
    modal_list = []
    for qs in questions:
        if model.config.mm_use_im_start_end:
            qs = default_mm_start_token + default_mm_token + default_mm_end_token + "\n" + qs
        else:
            qs = default_mm_token + "\n" + qs

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_id = tokenizer_MMODAL_token(prompt, tokenizer, modal_token_index, return_tensors='pt')
        input_ids.append(input_id)
        modal_list.append("video")

    # left pad sequence
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [x.flip(dims=[0]) for x in input_ids],
        batch_first=True,
        padding_value=tokenizer.pad_token_id).flip(dims=[1]).to(device)

    attention_mask=input_ids.ne(tokenizer.pad_token_id).to(device)

    video_tensor = video_tensor.half().to(args.device)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            images_or_videos=video_tensor,
            modal_list=modal_list,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id)

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return outputs


def run_inference(args):
    # Initialize the model
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)

    num_frames = model.config.num_frames if hasattr(model.config, "num_frames") else NUM_FRAMES

    gt_questions = json.load(open(args.question_file, "r"))
    gt_questions = get_chunk(gt_questions, args.num_chunks, args.chunk_idx)
    
    answer_file = os.path.join(args.output_file)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    ans_file = open(answer_file, "w")

    video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    # Iterate over each sample in the ground truth file
    for idx, sample in enumerate(tqdm(gt_questions)):
        video_name = sample['video_path']
        question = sample['question']
        answer = sample['captions']

        video_path = os.path.join(args.video_folder, video_name)

        video_tensor = process_video(video_path, processor, aspect_ratio=None, sample_scheme='uniform')
        output = get_model_output(model, tokenizer, video_tensor[None], [question], args.conv_mode, args.device)[0]

        sample_set = {'video_name': video_name, 'question': question, 'answer': answer, 'pred': output}
        ans_file.write(json.dumps(sample_set) + "\n")

    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--model-path', help='', required=True)
    parser.add_argument('--model_base', help='', default=None, type=str, required=False)
    parser.add_argument('--video-folder', help='Directory containing video files.', required=True)
    parser.add_argument('--question-file', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--output-file', help='Directory to save the model results JSON.', required=True)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument("--model_max_length", type=int, required=False, default=2048)

    args = parser.parse_args()
    run_inference(args)
