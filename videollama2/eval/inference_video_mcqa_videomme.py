import os
import re
import math
import json
import copy
import argparse
import warnings
import traceback

import cv2
import torch
import pysubs2
import numpy as np
import pyarrow.parquet as pq
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


def get_seq_frames(total_num_frames, desired_num_frames):
    """
    Calculate the indices of frames to extract from a video.

    Parameters:
    total_num_frames (int): Total number of frames in the video.
    desired_num_frames (int): Desired number of frames to extract.

    Returns:
    list: List of indices of frames to extract.
    """

    # Calculate the size of each segment from which a frame will be extracted
    seg_size = float(total_num_frames - 1) / desired_num_frames

    seq = []
    for i in range(desired_num_frames):
        # Calculate the start and end indices of each segment
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))

        # Append the middle index of the segment to the list
        seq.append((start + end) // 2)

    return seq


class VideoMMEDataset(Dataset):

    video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    def __init__(self, video_folder, subtitle_folder, data_list, processor):
        self.video_folder = video_folder
        self.subtitle_folder = subtitle_folder
        self.data_list = data_list
        self.processor = processor

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        line = self.data_list[idx]

        video_ytid = line['url'].split('watch?v=')[-1]

        for fmt in self.video_formats:  # Added this line
            temp_path = os.path.join(self.video_folder, f'{video_ytid}{fmt}')
            if os.path.exists(temp_path):
                video_path = temp_path
                break

        subtitle_path = os.path.join(self.subtitle_folder, f'{video_ytid}.srt')

        try:
            video_tensor = self.processor(video_path)
            num_frames = video_tensor.shape[0]
        except:
            traceback.print_exc()
            print(f'It occurs error when reading {video_ytid}')
            video_tensor = None
            num_frames = 0

        if video_tensor is not None and os.path.exists(subtitle_path):
            cv2_vr = cv2.VideoCapture(video_path)
            duration = int(cv2_vr.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cv2_vr.get(cv2.CAP_PROP_FPS)
            selected_frame_ids = get_seq_frames(duration, num_frames)

            subs = pysubs2.load(subtitle_path, encoding="utf-8")
            subtitles = []
            for seleced_frame_id in selected_frame_ids:
                sub_text = ""
                cur_time = pysubs2.make_time(fps=fps, frames=seleced_frame_id)
                for sub in subs:
                    if sub.start < cur_time and sub.end > cur_time:
                        sub_text = sub.text.replace("\\N", " ")
                        break
                if sub_text.strip():
                    subtitles.append(sub_text)
            subtitles = "\n".join(subtitles)
        else:
            subtitles = ""

        return {
            'video': video_tensor,
            'subtitle': subtitles,
            'record': line,
        }


def collate_fn(batch):
    vid = [x['video'] for x in batch]
    sub = [x['subtitle'] for x in batch]
    rcs = [x['record'] for x in batch]
    return vid, sub, rcs


def load_parquet(parquet_file):
    table = pq.read_table(parquet_file)

    # Convert PyArrow Table to pandas DataFrame
    df = table.to_pandas()

    jsons = []
    for record in df.itertuples():

        if len(jsons) < int(record.video_id):
            jsons.append({
                "video_id": record.video_id,
                "youtube_id": record.videoID,
                "url": record.url,
                "duration": record.duration,
                "domain": record.domain,
                "sub_category": record.sub_category,
                "questions": [
                    {
                        "question_id": record.question_id,
                        "task_type": record.task_type,
                        "question": record.question,
                        "choices": list(record.options),
                        "answer": record.answer,
                    }
                ]
            })
        else:
            jsons[-1]['questions'].append({
                "question_id": record.question_id,
                "task_type": record.task_type,
                "question": record.question,
                "choices": list(record.options),
                "answer": record.answer,
            })

    return jsons


def build_videomme_eval(args, processor):
    # convert parquet to json
    questions = load_parquet(args.question_file)
    # questions = json.load(open(args.question_file, "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    dataset = VideoMMEDataset(args.video_folder, args.subtitle_folder, questions, processor)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    return dataloader


def videomme_dump(record, instruct, options, output):
    letters = ['A', 'B', 'C', 'D']

    digit2word = {
        '1': 'one',
        '2': 'two',
        '3': 'three',
        '4': 'four',
        '5': 'five',
        '6': 'six',
        '7': 'seven',
        '8': 'eight',
        '9': 'nine',
        '0': 'zero',
    }

    output = output.replace('answer', '')
    output = output.replace('Answer', '')
    pred_answer = re.findall('[\(\ \[]*([A-D])[\)\.\ \]]*', output)
    try:
        find_flag = False
        if len(pred_answer) == 0:
            for idx, opt in enumerate(options):
                # Arabic numerals -> English words
                opt2 = opt
                if opt in digit2word:
                    opt2 = digit2word[opt]
                if opt.lower() in output.lower() or opt2.lower() in output.lower():
                    pred_idx = idx
                    find_flag = True
                    break
        else:
            pred_answer = pred_answer[0].strip()
            pred_answer = pred_answer.strip('()')
            pred_idx = letters.index(pred_answer)
            find_flag = True

        assert find_flag, 'The video \"{}\" instruct: \n\"{}\"\n output: \n\"{}\"\n is not in the expected format'.format(record['youtube_id'], instruct, output)
    except:
        traceback.print_exc()
        pred_idx = 2

    return letters[pred_idx]


def run_inference(args):
    disable_torch_init()

    # Initialize the model
    model, processor, tokenizer = model_init(args.model_path)
    model.model.audio_tower = None
    answer_file = os.path.expanduser(args.answer_file)
    answer_sub_file = answer_file.replace('.json', '_sub.json')
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    ans_file = open(answer_file, "w")
    ans_sub_file = open(answer_sub_file, "w")

    val_loader = build_videomme_eval(args, processor['video'])

    # Iterate over each sample in the ground truth file
    for i, (videos, subtitles, records) in enumerate(tqdm(val_loader)):
        video_tensor  = videos[0]
        subtitle = subtitles[0]
        record = records[0]

        new_record = copy.deepcopy(record)
        new_record_sub = copy.deepcopy(record)

        if video_tensor is None:
            new_record['missing'] = True
            ans_file.write(json.dumps(new_record) + ",\n")
            new_record_sub['missing'] = True
            ans_sub_file.write(json.dumps(new_record_sub) + ",\n")
            continue
        else:
            new_record['missing'] = False
            new_record_sub['missing'] = False

        questions = record['questions']
        for idx, question in enumerate(questions):
            q = question['question']
            choices = question['choices']
            options = [re.findall('[A-D]\. (.*).', c)[0] for c in choices]

            instruct = "Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.\n"
            instruct += f"{q}\n"
            for cho_idx, cho in enumerate(choices):
                instruct += f"{cho}\n"
            # instruct += "The best option is: "
            instruct += "Answer with the option\'s letter from the given choices directly and only give the best option. The best answer is: "
            output = mm_infer(video_tensor, instruct, model=model, tokenizer=tokenizer, modal='video', do_sample=False)
            new_record['questions'][idx]['response'] = videomme_dump(record, instruct, options, output)

            instruct = f"This video's subtitles are listed below:\n{subtitle}\n" + instruct
            output = mm_infer(video_tensor, instruct, model=model, tokenizer=tokenizer, modal='video', do_sample=False)
            new_record_sub['questions'][idx]['response'] = videomme_dump(record, instruct, options, output)

        ans_file.write(json.dumps(new_record) + ",\n")
        ans_sub_file.write(json.dumps(new_record_sub) + ",\n")

    ans_file.close()
    ans_sub_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', help='', required=True)
    parser.add_argument('--video-folder', help='Directory containing video files.', required=True)
    parser.add_argument('--subtitle-folder', help='Directory containing subtitle files.', required=True)
    parser.add_argument('--question-file', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--answer-file', help='Path to the ground truth file containing answers.', required=True)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    run_inference(args)
