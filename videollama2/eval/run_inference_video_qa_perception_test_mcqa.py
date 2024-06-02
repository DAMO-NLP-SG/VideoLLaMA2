import os
import re
import math
import json
import argparse
import warnings
from tqdm import tqdm

import torch
import decord
import numpy as np
import transformers
from decord import VideoReader, cpu
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append('./')
from videollama2.conversation import conv_templates, SeparatorStyle
from videollama2.constants import NUM_FRAMES, DEFAULT_MMODAL_TOKEN, DEFAULT_MMODAL_START_TOKEN, DEFAULT_MMODAL_END_TOKEN, MMODAL_TOKEN_INDEX
from videollama2.mm_utils import get_model_name_from_path, tokenizer_MMODAL_token, KeywordsStoppingCriteria, process_videos
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


class PerceptionTestMCQADataset(Dataset):

    video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    def __init__(self, data_list, processor, num_segments=8):
        self.data_list = data_list
        self.processor = processor
        self.num_segments = num_segments

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        line = self.data_list[idx]
        video_name = line['metadata']['video_id']
        mc_questions = line['mc_question']

        for fmt in self.video_formats:  # Added this line
            temp_path = os.path.join(args.video_folder, f"{video_name}{fmt}")
            if os.path.exists(temp_path):
                video_path = temp_path
                break

        decord_vr = VideoReader(uri=video_path, ctx=cpu(0))
        frames = decord_vr.get_batch(np.linspace(0, len(decord_vr) - 1, self.num_segments, dtype=int)).asnumpy()
        video_tensor = self.processor.preprocess(frames, return_tensors='pt')['pixel_values']  # do not pad for video frames

        qs = []
        qids = []
        ops = []
        for q in mc_questions:
            question = q['question']
            qid = q['id']
            options = q['options']
            option_question = f'Question: {question}\nOptions:\n(A) {options[0]}\n(B) {options[1]}\n(C) {options[2]}\nAnswer with the option\'s letter from the given choices directly and only give the best option.'

            qs.append(option_question)
            qids.append(qid)
            ops.append(options)

        return {
            'video': video_tensor,
            'video_id': video_name,
            'questions': qs,
            'question_ids': qids,
            'options': ops,
        }


def collate_fn(batch):
    vid = [x['video'] for x in batch]
    v_id = [x['video_id'] for x in batch]
    qs = [x['questions'] for x in batch]
    q_ids = [x['question_ids'] for x in batch]
    ops = [x['options'] for x in batch]
    vid = torch.stack(vid, dim=0)
    return vid, v_id, qs, q_ids, ops


def get_model_output(model, tokenizer, qs, video_tensor, args):
    if model.config.mm_use_im_start_end:
        qs = default_mm_start_token + default_mm_token + default_mm_end_token + "\n" + qs
    else:
        qs = default_mm_token + "\n" + qs

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(args.device)
    input_ids = tokenizer_MMODAL_token(prompt, tokenizer, modal_token_index, return_tensors='pt').to(args.device)

    attention_mask=input_ids.ne(tokenizer.pad_token_id).to(args.device)

    modal_list = ["video"]
    video_tensor = video_tensor.to(dtype=torch.float16, device=args.device, non_blocking=True)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
            images_or_videos=[video_tensor],
            modal_list=modal_list,
            do_sample=False,
            max_new_tokens=1024,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id)

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs


def run_inference(args):
    # Initialize the model
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)

    questions = json.load(open(args.question_file, "r"))
    questions = list(questions.values())
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    num_frames = model.config.num_frames if hasattr(model.config, "num_frames") else NUM_FRAMES

    assert args.batch_size == 1, "Batch size must be 1 for inference"
    dataset = PerceptionTestMCQADataset(questions, processor, num_frames)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn)

    answer_file = os.path.expanduser(args.answer_file)
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    ans_file = open(answer_file, "w")

    output_list = []  # List to store the output results

    # Iterate over each sample in the ground truth file
    for i, (video_tensor, video_id, questions, question_ids, options) in enumerate(tqdm(dataloader)):

        # reduce batch dimension
        video_tensor = video_tensor[0]
        video_id = video_id[0]
        questions = questions[0]
        question_ids = question_ids[0]
        options = options[0]

        qas = []
        for idx, question in enumerate(questions):
            letters = ['(A)', '(B)', '(C)']
            question_id = question_ids[idx]
            _options = options[idx]

            output = get_model_output(model, tokenizer, question, video_tensor, args)
            pred_answer = re.findall('\(*[A-C]\)*', output)
            if len(pred_answer) == 0:
                tmp_options = [x.lower() for x in _options]
                if output.lower() in tmp_options:
                    tmp_options = [x.lower() for x in _options]
                    pred_idx = tmp_options.index(output.lower())
                else:
                    pred_idx = 2
            else:
                pred_answer = pred_answer[0].strip()
                if not pred_answer.startswith('('):
                    pred_answer = f'({pred_answer})'
                pred_idx = letters.index(pred_answer)

            qas.append({'id': question_id, 'answer_id': pred_idx, 'answer': _options[pred_idx]})

        ans_file.write('\"{}\": {},\n'.format(video_id, json.dumps(qas)))

    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--model-path', help='', required=True)
    parser.add_argument('--model_base', help='', default=None, type=str, required=False)
    parser.add_argument('--video-folder', help='Directory containing video files.', required=True)
    parser.add_argument('--question-file', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--answer-file', help='Path to the ground truth file containing answers.', required=True)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument("--model_max_length", type=int, required=False, default=2048)
    parser.add_argument("--batch-size", type=int, required=False, default=1)
    parser.add_argument("--num-workers", type=int, required=False, default=8)
    args = parser.parse_args()

    run_inference(args)
