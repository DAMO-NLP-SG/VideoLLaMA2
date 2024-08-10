import os
import re
import math
import json
import argparse
import warnings
import traceback
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append('./')
from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


class PerceptionTestMCQADataset(Dataset):

    video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    def __init__(self, data_list, processor):
        self.data_list = data_list
        self.processor = processor

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
        
        video_tensor = self.processor(video_path)

        instructs = []
        qids = []
        ops = []
        for q in mc_questions:
            question = q['question']
            qid = q['id']
            options = q['options']
            instruct = f'Question: {question}\nOptions:\n(A) {options[0]}\n(B) {options[1]}\n(C) {options[2]}\nAnswer with the option\'s letter from the given choices directly and only give the best option.'

            instructs.append(instruct)
            qids.append(qid)
            ops.append(options)

        return {
            'video': video_tensor,
            'video_id': video_name,
            'instructs': instructs,
            'question_ids': qids,
            'options': ops,
        }


def collate_fn(batch):
    vid = [x['video'] for x in batch]
    v_id = [x['video_id'] for x in batch]
    ins = [x['instructs'] for x in batch]
    q_ids = [x['question_ids'] for x in batch]
    ops = [x['options'] for x in batch]
    vid = torch.stack(vid, dim=0)
    return vid, v_id, ins, q_ids, ops


def run_inference(args):
    disable_torch_init()

    model, processor, tokenizer = model_init(args.model_path)

    questions = json.load(open(args.question_file, "r"))
    questions = list(questions.values())
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    assert args.batch_size == 1, "Batch size must be 1 for inference"
    dataset = PerceptionTestMCQADataset(questions, processor['video'])
    dataloader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn)

    answer_file = os.path.expanduser(args.answer_file)
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    ans_file = open(answer_file, "w")

    # Iterate over each sample in the ground truth file
    for i, (video_tensor, video_id, instructs, question_ids, options) in enumerate(tqdm(dataloader)):

        # reduce batch dimension
        video_tensor = video_tensor[0]
        video_id = video_id[0]
        instructs = instructs[0]
        question_ids = question_ids[0]
        options = options[0]

        qas = []
        for idx, instruct in enumerate(instructs):
            letters = ['(A)', '(B)', '(C)']
            question_id = question_ids[idx]
            _options = options[idx]

            output = mm_infer(
                video_tensor,
                instruct,
                model=model,
                tokenizer=tokenizer,
                modal='video',
                do_sample=False,
            )

            output = output.replace('answer', '')
            output = output.replace('Answer', '')
            pred_answer = re.findall('\(*[A-C]\)*', output)
            try:
                assert len(pred_answer) >= 1, 'The video \"{}\" instruct: \n\"{}\"\n output: \n\"{}\"\n is not in the expected format'.format(video_id, instruct, output)
                pred_answer = pred_answer[0].strip()
                # if not pred_answer.startswith('('):
                pred_answer = pred_answer.strip('()')
                pred_answer = f'({pred_answer})'
                pred_idx = letters.index(pred_answer)
            except:
                traceback.print_exc()
                tmp_options = [x.lower() for x in _options]
                if output.lower() in tmp_options:
                    tmp_options = [x.lower() for x in _options]
                    pred_idx = tmp_options.index(output.lower())
                else:
                    pred_idx = 2

            qas.append({'id': question_id, 'answer_id': pred_idx, 'answer': _options[pred_idx]})

        ans_file.write('\"{}\": {},\n'.format(video_id, json.dumps(qas)))

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
    parser.add_argument("--model_max_length", type=int, required=False, default=2048)
    parser.add_argument("--batch-size", type=int, required=False, default=1)
    parser.add_argument("--num-workers", type=int, required=False, default=8)
    args = parser.parse_args()

    run_inference(args)
