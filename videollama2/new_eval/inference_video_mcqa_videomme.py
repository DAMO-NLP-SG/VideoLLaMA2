import os
import re
import math
import json
import copy
import argparse
import warnings
import traceback

import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append('./')
from videollama2 import model_init, x_infer

# NOTE: Ignore TypedStorage warning, which refers to this link~(https://github.com/pytorch/pytorch/issues/97207#issuecomment-1494781560)
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


class VideoMMEDataset(Dataset):

    video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    def __init__(self, data_folder, data_list, processor):
        self.data_folder = data_folder
        self.data_list = data_list
        self.processor = processor

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        line = self.data_list[idx]

        video_ytid = line['url'][-11:]

        for fmt in self.video_formats:  # Added this line
            temp_path = os.path.join(self.data_folder, f'v_{video_ytid}{fmt}')
            if os.path.exists(temp_path):
                video_path = temp_path
                break

        try:
            video_tensor = self.processor(video_path)
        except:
            traceback.print_exc()
            print(f'It occurs error when reading {video_ytid}')
            video_tensor = None

        return {
            'video': video_tensor, 
            'record': line,
        }


def collate_fn(batch):
    vid = [x['video'] for x in batch]
    rcs = [x['record'] for x in batch]
    return vid, rcs


def build_videomme_eval(args, processor):
    questions = json.load(open(args.question_file, "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    dataset = VideoMMEDataset(args.video_folder, questions, processor)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    return dataloader


def videomme_dump(output):
    letters = ['A', 'B', 'C', 'D']

    pred_answer = re.findall('[\(\ ]*([A-D])[\)\.\ ]*', output)
    if len(pred_answer) == 0:
        assert False
        pred_idx = 2
    else:
        pred_answer = pred_answer[0].strip()
        pred_answer = pred_answer.strip('()')
        pred_idx = letters.index(pred_answer)
    
    return letters[pred_idx]


def run_inference(args):
    # Initialize the model
    model, processor, tokenizer = model_init(args.model_path)

    answer_file = os.path.expanduser(args.answer_file)
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    ans_file = open(answer_file, "w")

    val_loader = build_videomme_eval(args, processor)

    # Iterate over each sample in the ground truth file
    for i, (videos, records) in enumerate(tqdm(val_loader)):
        video_tensor  = videos[0]
        record = records[0]

        new_record = copy.deepcopy(record)

        if video_tensor is None:
            new_record['missing'] = True
            ans_file.write(json.dumps(new_record) + ",\n")
            continue
        else:
            new_record['missing'] = False

        questions = record['questions']
        for idx, question in enumerate(questions):
            q = question['question']
            ops = question['choices']

            instruct = "Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.\n"
            instruct += f"{q}\n"
            for op_idx, op in enumerate(ops):
                instruct += f"{chr(65 + op_idx)}. {op}\n"
            instruct += "The best answer is: "
            
            output = x_infer(
                video_tensor,
                instruct,
                mode='vanilla',
                model=model,
                tokenizer=tokenizer,
                do_sample=True
            )

            new_record['questions'][idx]['response'] = videomme_dump(output)

        ans_file.write(json.dumps(new_record) + ",\n")

    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', help='', required=True)
    parser.add_argument('--video-folder', help='Directory containing video files.', required=True)
    parser.add_argument('--question-file', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--answer-file', help='Path to the ground truth file containing answers.', required=True)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    run_inference(args)
