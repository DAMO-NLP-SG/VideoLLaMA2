import os
import re
import math
import json
import argparse
import warnings
from tqdm import tqdm

import torch
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


class VCGPTDataset(Dataset):

    video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    def __init__(self, data_list, processor):
        self.data_list = data_list
        self.processor = processor

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        line = self.data_list[idx]
        question1 = line['Q1']
        question2 = line['Q2']
        answer = line['A']
        video_name = line['video_name']

        for fmt in self.video_formats:  # Added this line
            temp_path = os.path.join(args.video_folder, f"{video_name}{fmt}")
            if os.path.exists(temp_path):
                video_path = temp_path
                break

        video_tensor = self.processor(video_path)

        return {
            'video': video_tensor,
            'video_name': video_name,
            'question1': question1,
            'question2': question2,
            'answer': answer,
        }


def collate_fn(batch):
    vid = [x['video'] for x in batch]
    v_id = [x['video_name'] for x in batch]
    qus1 = [x['question1'] for x in batch]
    qus2 = [x['question2'] for x in batch]
    ans = [x['answer'] for x in batch]
    vid = torch.stack(vid, dim=0)
    return vid, v_id, qus1, qus2, ans


def run_inference(args):
    # Initialize the model
    model, processor, tokenizer = model_init(args.model_path)

    questions = json.load(open(args.question_file, "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    assert args.batch_size == 1, "Batch size must be 1 for inference"
    dataset = VCGPTDataset(questions, processor)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn)

    answer_file = os.path.expanduser(args.answer_file)
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    ans_file = open(answer_file, "w")

    output_list = []  # List to store the output results

    # Iterate over each sample in the ground truth file
    for i, (video_tensors, video_names, questions1, questions2, answers) in enumerate(tqdm(dataloader)):

        # reduce batch dimension
        video_tensor = video_tensors[0]
        video_name = video_names[0]
        question1 = questions1[0]
        question2 = questions2[0]
        answer = answers[0]

        output1 = x_infer(
            video_tensor,
            question1, 
            mode='vanilla',
            model=model,
            tokenizer=tokenizer,
            do_sample=False,
        )

        output2 = x_infer(
            video_tensor,
            question2, 
            mode='vanilla',
            model=model,
            tokenizer=tokenizer,
            do_sample=False,
        )

        qa = {'video_name': video_name, 'Q1': question1, 'Q2': question2, 'A': answer, 'P1': output1, 'P2': output2}

        ans_file.write(json.dumps(qa) + "\n")

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
