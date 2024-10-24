import os
import json
import math
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


class AVQADataset(Dataset):

    def __init__(self, questions, processor):
        self.questions = questions
        self.processor = processor

    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        sample = self.questions[idx]

        video_path  = sample['video']
        question    = sample['conversations'][0]["value"].replace("<video>", "").strip()
        question_id = video_path.split("/")[-1]
        answer      = sample['conversations'][1]["value"]

        try:
            audio_video_dict = self.processor(video_path, va=True)
        except:
            print("video read error")
            audio_video_dict = None

        return {
            'audio_video':  audio_video_dict,
            'video_name':  video_path.split("/")[-1],
            'question':    question,
            'question_id': question_id,
            'answer':      answer,
        }

class AVSDDataset(Dataset):

    def __init__(self, questions, processor):
        self.questions = questions
        self.processor = processor

    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        sample = self.questions[idx]

        video_path  = sample['video']
        question    = sample['conversations'][0]["value"].replace("<video>", "").strip()
        question_id = video_path.split("/")[-1]
        answer      = sample['conversations'][1]["value"]

        try:
            audio_video_dict = self.processor(video_path, va=True)
        except:
            print("video read error")
            audio_video_dict = None

        return {
            'audio_video':  audio_video_dict,
            'video_name':  video_path.split("/")[-1],
            'question':    question,
            'question_id': question_id,
            'answer':      answer,
        }


class AVSSDDataset(Dataset):

    def __init__(self, questions, processor):
        self.questions = questions
        self.processor = processor

    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        sample = self.questions[idx]

        video_path  = sample['video']
        question    = "Identify the event in the video."
        question_id = video_path.split("/")[-1]
        answer      = sample['conversations'][1]["value"]

        try:
            audio_video_dict = self.processor(video_path, va=True)
        except:
            print("video read error")
            audio_video_dict = None

        return {
            'audio_video':  audio_video_dict,
            'video_name':  video_path.split("/")[-1],
            'question':    question,
            'question_id': question_id,
            'answer':      answer,
        }


def collate_fn(batch):
    aud_vid  = [x['audio_video'] for x in batch]
    v_id = [x['video_name'] for x in batch]
    qus  = [x['question'] for x in batch]
    qid  = [x['question_id'] for x in batch]
    ans  = [x['answer'] for x in batch]
    return aud_vid, v_id, qus, qid, ans


def run_inference(args):
    disable_torch_init()

    # Initialize the model
    model, processor, tokenizer = model_init(args.model_path)

    gt_questions = json.load(open(args.question_file, "r"))
    gt_questions = get_chunk(gt_questions, args.num_chunks, args.chunk_idx)

    assert args.batch_size == 1, "Batch size must be 1 for inference"
    if args.dataset == "AVQA":
        dataset = AVQADataset(gt_questions, processor['video'])
    elif args.dataset == "AVSD":
        dataset = AVSDDataset(gt_questions, processor['video'])
    elif args.dataset == "AVSSD":
        dataset = AVSSDDataset(gt_questions, processor['video'])
    else:
        raise NotImplementedError
    dataloader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn)

    answer_file = os.path.join(args.output_file)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    ans_file = open(answer_file, "w")

    # Iterate over each sample in the ground truth file
    for i, (aud_vid_tensors, video_names, questions, question_ids, answers) in enumerate(tqdm(dataloader)):
        audio_video_tensor = aud_vid_tensors[0]
        video_name   = video_names[0]
        question     = questions[0]
        question_id  = question_ids[0]
        answer       = answers[0]

        try:
            output = mm_infer(
                audio_video_tensor,
                question,
                model=model,
                tokenizer=tokenizer,
                modal='video',
                do_sample=False,
            )
        except:
            traceback.print_exc()
            output = "error"

        sample_set = {'id': question_id, 'question': question, 'answer': answer, 'pred': output}
        ans_file.write(json.dumps(sample_set) + "\n")

    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', help='', required=True)
    parser.add_argument('--video-folder', help='Directory containing video files.', required=True)
    parser.add_argument('--question-file', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--answer-file', help='Path to the ground truth file containing answers.', required=False)
    parser.add_argument('--output-file', help='Directory to save the model results JSON.', required=True)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument("--batch-size", type=int, required=False, default=1)
    parser.add_argument("--num-workers", type=int, required=False, default=8)
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    run_inference(args)
