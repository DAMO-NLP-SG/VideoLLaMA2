import os
import re
import math
import json
import argparse
import warnings
import traceback

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append('./')
from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init

# NOTE: Ignore TypedStorage warning, which refers to this link~(https://github.com/pytorch/pytorch/issues/97207#issuecomment-1494781560)
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


class EgoschemaDataset(Dataset):

    video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    def __init__(self, data_folder, data_list, processor):
        self.data_folder = data_folder
        self.data_list = data_list
        self.processor = processor

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        line = self.data_list[idx]
        q_uid = line['q_uid']

        for fmt in self.video_formats:  # Added this line
            temp_path = os.path.join(self.data_folder, f"{q_uid}{fmt}")
            if os.path.exists(temp_path):
                video_path = temp_path
                break

        video_tensor = self.processor(video_path)

        question = line['question']
        a0 = line['option 0']
        a1 = line['option 1']
        a2 = line['option 2']
        a3 = line['option 3']
        a4 = line['option 4']
        axs = [a0, a1, a2, a3, a4]
        ops = ['(A)', '(B)', '(C)', '(D)', '(E)']

        instruct = f'Select the best answer to the following multiple-choice question based on the video.\n{question}\nOptions:\n(A) {a0}\n(B) {a1}\n(C) {a2}\n(D) {a3}\n(E) {a4}\nAnswer with the option\'s letter from the given choices directly and only give the best option. The best answer is: ' 

        return {
            'q_uid': q_uid,
            'video': video_tensor, 
            'instruct': instruct,
        }


def build_egoschema_eval(args, processor):
    questions = json.load(open(args.question_file, "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    dataset = EgoschemaDataset(args.video_folder, questions, processor)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return dataloader


def egoschema_dump(ans_file, line, outputs):
    for idx, output in enumerate(outputs):
        q_uid = line['q_uid'][idx]
        instruct = line['instruct'][idx]
        letters = ['A', 'B', 'C', 'D', 'E']

        output = output.replace('answer', '')
        output = output.replace('Answer', '')
        pred_answer = re.findall('[\(\ ]*[A-E][\)\ ]*', output)
        try:
            
            assert len(pred_answer) >= 1, 'The video \"{}\" instruct: \n\"{}\"\n output: \n\"{}\"\n is not in the expected format'.format(line['q_uid'], instruct, output)
            pred_answer = pred_answer[0].strip()
            pred_answer = pred_answer.strip('()')
            pred_idx = letters.index(pred_answer)
        except:
            traceback.print_exc()
            pred_idx = 2

        ans_file.write(f'{q_uid}, {pred_idx}\n')


def run_inference(args):
    disable_torch_init()

    model, processor, tokenizer = model_init(args.model_path)

    answer_file = os.path.expanduser(args.answer_file)
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    ans_file = open(answer_file, "w")

    val_loader = build_egoschema_eval(args, processor['video'])

    # Iterate over each sample in the ground truth file
    for i, line in enumerate(tqdm(val_loader)):
        video_tensor = line['video'][0]
        instruct = line['instruct'][0]

        try:
            pred = mm_infer(
                video_tensor,
                instruct,
                model=model,
                tokenizer=tokenizer,
                modal='video',
                do_sample=False,
            )
        except:
            traceback.print_exc()
            pred = 'C'

        egoschema_dump(ans_file, line, [pred])

    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multiple-Choice Video QA Evaluation Script.')

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
