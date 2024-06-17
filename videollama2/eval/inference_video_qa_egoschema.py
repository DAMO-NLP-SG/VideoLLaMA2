import os
import re
import math
import json
import argparse
import warnings

import torch
import decord
import numpy as np
import transformers
from PIL import Image
from tqdm import tqdm
from decord import VideoReader, cpu
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.transforms import functional as F

import sys
sys.path.append('./')
from videollama2.conversation import conv_templates, SeparatorStyle
from videollama2.constants import NUM_FRAMES, DEFAULT_MMODAL_TOKEN, DEFAULT_MMODAL_START_TOKEN, DEFAULT_MMODAL_END_TOKEN, MMODAL_TOKEN_INDEX
from videollama2.mm_utils import get_model_name_from_path, tokenizer_MMODAL_token, KeywordsStoppingCriteria, process_videos, expand2square
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


class EgoschemaDataset(Dataset):

    video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    def __init__(self, data_list, processor, num_segments=8):
        self.data_list = data_list
        self.processor = processor
        self.num_segments = num_segments

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        line = self.data_list[idx]
        q_uid = line['q_uid']

        for fmt in self.video_formats:  # Added this line
            temp_path = os.path.join(args.video_folder, f"{q_uid}{fmt}")
            if os.path.exists(temp_path):
                video_path = temp_path
                break

        decord_vr = VideoReader(uri=video_path, ctx=cpu(0))
        frames = decord_vr.get_batch(np.linspace(0, len(decord_vr) - 1, self.num_segments, dtype=int)).asnumpy()
        video_tensor = self.processor.preprocess(frames, return_tensors='pt')['pixel_values']  # do not pad for video frames

        question = line['question']
        a0 = line['option 0']
        a1 = line['option 1']
        a2 = line['option 2']
        a3 = line['option 3']
        a4 = line['option 4']
        axs = [a0, a1, a2, a3, a4]
        ops = ['(A)', '(B)', '(C)', '(D)', '(E)']

        option_question = f'Question: {question}\nOptions:\n(A) {a0}\n(B) {a1}\n(C) {a2}\n(D) {a3}\n(E) {a4}\nAnswer with the option\'s letter from the given choices directly and only give the best option.' 

        return {
            'q_uid': q_uid,
            'video': video_tensor, 
            'question': option_question,
        }


def build_egoschema_eval(args, processor, num_frames):
    questions = json.load(open(args.question_file, "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    dataset = EgoschemaDataset(questions, processor, num_segments=num_frames)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    return dataloader


def egoschema_dump(ans_file, line, outputs):
    for idx, output in enumerate(outputs):
        q_uid = line['q_uid'][idx]
        letters = ['A', 'B', 'C', 'D', 'E']

        pred_answer = re.findall('[\(\ ]*[A-E][\)\ ]*', output)
        if len(pred_answer) == 0:
            pred_idx = 2
        else:
            pred_answer = pred_answer[0].strip()
            # if pred_answer.startswith('('):
            pred_answer = pred_answer.strip('()')
            pred_idx = letters.index(pred_answer)
        ans_file.write(f'{q_uid}, {pred_idx}\n')


def get_model_output(model, video_tensor, tokenizer, questions, conv_mode="v1", device='cuda'):
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

    print(input_ids)

    attention_mask=input_ids.ne(tokenizer.pad_token_id).to(device)

    video_tensor = video_tensor.half().to(args.device)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            images_or_videos=video_tensor,
            modal_list=modal_list,
            do_sample=False,
            max_new_tokens=1024,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id)

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return outputs


def run_inference(args):
    """
    Run inference on ActivityNet QA DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    # Initialize the model
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)

    num_frames = model.config.num_frames if hasattr(model.config, "num_frames") else NUM_FRAMES

    answer_file = os.path.expanduser(args.answer_file)
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    ans_file = open(answer_file, "w")

    output_list = []  # List to store the output results

    if args.dataset == 'egoschema':
        val_loader = build_egoschema_eval(args, processor, num_frames)
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented.")

    # Iterate over each sample in the ground truth file
    for i, line in enumerate(tqdm(val_loader)):
        video_tensor = line['video']
        questions = line['question']

        outputs = get_model_output(model, video_tensor, tokenizer, questions, args.conv_mode, args.device)

        if args.dataset == 'egoschema':
            egoschema_dump(ans_file, line, outputs)
        else:
            raise NotImplementedError(f"Dataset {args.dataset} not implemented.")

    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multiple-Choice Video QA Evaluation Script.')

    parser.add_argument('--dataset', help='Dataset to evaluate on.', required=True)
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
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()
    run_inference(args)
