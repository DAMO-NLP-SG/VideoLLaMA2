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


class ClothoAQADataset(Dataset):

    audoi_formats = ['.wav', '.flac']

    def __init__(self, questions, processor):
        self.questions = questions
        self.processor = processor

    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        sample = self.questions[idx]

        audio_path  = sample['audio']
        question    = sample['conversations'][0]["value"]
        wrapped_question = f"Question: {question}\nAnswer the question using a single word."
        question_id = sample['id']
        answer      = sample['conversations'][1]["value"]

        audio_tensor = self.processor(audio_path)

        return {
            'audio':       audio_tensor,
            'audio_name':  audio_path.split("/")[-1],
            'question':    wrapped_question,
            'question_id': question_id,
            'answer':      answer,
        }

class TUT2017Dataset(Dataset):

    audoi_formats = ['.wav', '.flac']

    def __init__(self, questions, processor):
        self.questions = questions
        self.processor = processor

    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        sample = self.questions[idx]

        audio_path  = sample['audio']
        wrapped_question = f"Question: Identify the sound event in the audio.\nOptions:\n(A) beach\n(B) bus\n(C) cafe or restaurant\n(D) car\n(E) city center\n(F) forest path\n(G) grocery store\n(H) home\n(I) library\n(J) metro station\n(K) office\n(L) park\n(M) residential area\n(N) train\n(O) tram\n.Answer with the option's letter from the given choices directly and only give the best option."
        question_id = audio_path.split("/")[-1]
        answer      = sample['gt']

        audio_tensor = self.processor(audio_path)

        return {
            'audio':       audio_tensor,
            'audio_name':  audio_path.split("/")[-1],
            'question':    wrapped_question,
            'question_id': question_id,
            'answer':      answer,
        }

class VocalSoundDataset(Dataset):

    audoi_formats = ['.wav', '.flac']

    def __init__(self, questions, processor):
        self.questions = questions
        self.processor = processor

    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        sample = self.questions[idx]

        audio_path  = sample['audio']
        wrapped_question = f"Identify the human sound in the audio.\nOptions:\n(A) Laughter\n(B) Sigh\n(C) Cough\n(D) Throat clearing\n(E) Sneeze\n(F) Sniff\n.Answer with the option's letter from the given choices directly and only give the best option."
        question_id = audio_path.split("/")[-1]
        answer      = sample['gt']

        audio_tensor = self.processor(audio_path)

        return {
            'audio':       audio_tensor,
            'audio_name':  audio_path.split("/")[-1],
            'question':    wrapped_question,
            'question_id': question_id,
            'answer':      answer,
        }


def collate_fn(batch):
    vid  = [x['audio'] for x in batch]
    v_id = [x['audio_name'] for x in batch]
    qus  = [x['question'] for x in batch]
    qid  = [x['question_id'] for x in batch]
    ans  = [x['answer'] for x in batch]
    return vid, v_id, qus, qid, ans


def run_inference(args):
    disable_torch_init()

    # Initialize the model
    model, processor, tokenizer = model_init(args.model_path)
    model.model.vision_tower = None

    assert args.batch_size == 1, "Batch size must be 1 for inference"
    if args.dataset == "clothoAQA":
        gt_questions = json.load(open(args.question_file, "r"))
        gt_questions = get_chunk(gt_questions, args.num_chunks, args.chunk_idx)
        dataset = ClothoAQADataset(gt_questions, processor['audio'])
    elif args.dataset == "TUT2017":
        gt_questions = []
        with open(args.question_file, "r") as fp:
            for x in fp.readlines():
                gt_questions.append(json.loads(x))
                gt_questions[-1]["audio"] = os.path.join(args.video_folder, gt_questions[-1]["audio"])
        gt_questions = get_chunk(gt_questions, args.num_chunks, args.chunk_idx)
        dataset = TUT2017Dataset(gt_questions, processor['audio'])
    elif args.dataset == "vocalsound":
        gt_questions = []
        with open(args.question_file, "r") as fp:
            for x in fp.readlines():
                gt_questions.append(json.loads(x))
                gt_questions[-1]["audio"] = os.path.join(args.video_folder, gt_questions[-1]["audio"].split("/")[-1])
        gt_questions = get_chunk(gt_questions, args.num_chunks, args.chunk_idx)
        dataset = VocalSoundDataset(gt_questions, processor['audio'])
    else:
        raise NotImplementedError

    dataloader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn)

    answer_file = os.path.join(args.output_file)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    ans_file = open(answer_file, "w")

    # Iterate over each sample in the ground truth file
    for i, (audio_tensors, audio_names, questions, question_ids, answers) in enumerate(tqdm(dataloader)):
        audio_tensor = audio_tensors[0]
        audio_name   = audio_names[0]
        question     = questions[0]
        question_id  = question_ids[0]
        answer       = answers[0]

        # question = question + '\n' + 'Answer the question using a single word or a short phrase with multiple words.'

        try:
            output = mm_infer(
                audio_tensor,
                question,
                model=model,
                tokenizer=tokenizer,
                modal='audio',
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
