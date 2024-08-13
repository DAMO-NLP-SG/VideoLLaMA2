import math
import os
import argparse
import json
import warnings
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


class MSVCDataset(Dataset):

    video_formats = ['.mp4', '.webm', '.avi', '.mov', '.mkv']

    def __init__(self, folder, questions, processor):
        self.folder = folder
        self.questions = questions
        self.processor = processor

    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        sample = self.questions[idx]

        video_name = sample['video_path']
        question   = sample['question']
        answer     = sample['captions']

        video_path = os.path.join(self.folder, video_name)
        video_tensor = self.processor(video_path)

        return {
            'video':       video_tensor,
            'video_name':  video_name,
            'question':    question,
            'answer':      answer,
        }


def collate_fn(batch):
    vid  = [x['video'] for x in batch]
    v_id = [x['video_name'] for x in batch]
    qus  = [x['question'] for x in batch]
    ans  = [x['answer'] for x in batch]
    return vid, v_id, qus, ans


def run_inference(args):
    disable_torch_init()

    model, processor, tokenizer = model_init(args.model_path)

    gt_questions = json.load(open(args.question_file, "r"))
    gt_questions = get_chunk(gt_questions, args.num_chunks, args.chunk_idx)

    answer_file = os.path.join(args.output_file)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    ans_file = open(answer_file, "w")

    assert args.batch_size == 1, "Batch size must be 1 for inference"
    dataset = MSVCDataset(args.video_folder, gt_questions, processor['video'])
    dataloader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn)

    # Iterate over each sample in the ground truth file
    for idx, (video_tensors, video_names, questions, answers) in enumerate(tqdm(dataloader)):
        video_tensor = video_tensors[0]
        video_name   = video_names[0]
        question     = questions[0]
        answer       = answers[0]

        output = mm_infer(
            video_tensor,
            question, 
            model=model,
            tokenizer=tokenizer,
            modal='video',
            do_sample=False,
        )

        sample_set = {'video_name': video_name, 'question': question, 'answer': answer, 'pred': output}
        ans_file.write(json.dumps(sample_set) + "\n")

    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', help='', required=True)
    parser.add_argument('--video-folder', help='Directory containing video files.', required=True)
    parser.add_argument('--question-file', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--output-file', help='Directory to save the model results JSON.', required=True)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument("--batch-size", type=int, required=False, default=1)
    parser.add_argument("--num-workers", type=int, required=False, default=8)
    args = parser.parse_args()

    run_inference(args)
