import os
import re
import math
import json
import argparse
import warnings
import traceback

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from decord import VideoReader, cpu
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


class MVBenchDataset(Dataset):

    def __init__(self, data_list, processor):
        self.data_list = data_list
        self.processor = processor

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        bound = (None, None)
        if self.data_list[idx]['bound']:
            bound = (self.data_list[idx]['data']['start'], self.data_list[idx]['data']['end'])
        video_path = os.path.join(self.data_list[idx]['prefix'], self.data_list[idx]['data']['video'])
        torch_imgs = self.processor(video_path, s=bound[0], e=bound[1])
        question = self.data_list[idx]['data']['question']
        options = self.data_list[idx]['data']['candidates']
        answer = self.data_list[idx]['data']['answer']
        task_type = self.data_list[idx]['task_type']

        answer_idx = -1
        letters = []
        options_string = ''
        for option_idx, c in enumerate(options):
            letters.append(f"{chr(ord('A') + option_idx)}")
            options_string += f"({chr(ord('A') + option_idx)}) {c}\n"
            if c == answer:
                answer_idx = option_idx

        instruct = f'Question: {question}\nOptions:\n{options_string}Answer with the option\'s letter from the given choices directly and only give the best option.' 

        return {
            'video': torch_imgs, 
            'video_path': video_path,
            'instruct': instruct,
            'letters': letters,
            'options': options,
            'answer_idx': answer_idx,
            'task_type': task_type
        }


tasks = {
    "Action Sequence": ("action_sequence.json", "star/Charades_v1_480/", "video", True), # has start & end
    "Action Prediction": ("action_prediction.json", "star/Charades_v1_480/", "video", True), # has start & end
    "Action Antonym": ("action_antonym.json", "ssv2_video/", "video", False),
    "Fine-grained Action": ("fine_grained_action.json", "Moments_in_Time_Raw/videos/", "video", False),
    "Unexpected Action": ("unexpected_action.json", "FunQA_test/test/", "video", False),
    "Object Existence": ("object_existence.json", "clevrer/video_validation/", "video", False),
    "Object Interaction": ("object_interaction.json", "star/Charades_v1_480/", "video", True), # has start & end
    "Object Shuffle": ("object_shuffle.json", "perception/videos/", "video", False),
    "Moving Direction": ("moving_direction.json", "clevrer/video_validation/", "video", False),
    "Action Localization": ("action_localization.json", "sta/sta_video/", "video", True),  # has start & end
    "Scene Transition": ("scene_transition.json", "scene_qa/video/", "video", False),
    "Action Count": ("action_count.json", "perception/videos/", "video", False),
    "Moving Count": ("moving_count.json", "clevrer/video_validation/", "video", False),
    "Moving Attribute": ("moving_attribute.json", "clevrer/video_validation/", "video", False),
    "State Change": ("state_change.json", "perception/videos/", "video", False),
    "Fine-grained Pose": ("fine_grained_pose.json", "nturgbd/", "video", False),
    "Character Order": ("character_order.json", "perception/videos/", "video", False),
    "Egocentric Navigation": ("egocentric_navigation.json", "vlnqa/", "video", False),
    "Episodic Reasoning": ("episodic_reasoning.json", "tvqa/frames_fps3_hq/", "frame", True),  # has start & end, read frame
    "Counterfactual Inference": ("counterfactual_inference.json", "clevrer/video_validation/", "video", False),
}


def build_mvbench_eval(args, processor):
    data_list = []
    for task_name, task in tasks.items():
        json_file = os.path.join(args.question_file, task[0])
        vis_folder = os.path.join(args.video_folder, task[1])
        with open(json_file, 'r') as f:
            json_data = json.load(f)
        for data in json_data:
            data_list.append({
                'task_type': task_name,
                'prefix': vis_folder,
                'data_type': task[2],
                'bound': task[3],
                'data': data
            })
    data_list = get_chunk(data_list, args.num_chunks, args.chunk_idx)
    dataset = MVBenchDataset(data_list, processor)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return dataloader


def mvbench_dump(vid, instruct, letters, options, output):
    
    output = output.replace('answer', '')
    output = output.replace('Answer', '')
    pred_answer = re.findall(f'[\(,\ ]*[{letters[0]}-{letters[-1]}][\),\ ]*', output)
    try:
        find_flag = False
        if len(pred_answer) == 0:
            for idx, opt in enumerate(options):
                # Arabic numerals -> English words
                if opt.lower() in output.lower():
                    pred_idx = idx
                    find_flag = True
                    break
        else:
            pred_answer = pred_answer[0].strip()
            pred_answer = pred_answer.strip('()')
            pred_idx = letters.index(pred_answer)
            find_flag = True

        assert find_flag, 'The video \"{}\" instruct: \n\"{}\"\n output: \n\"{}\"\n is not in the expected format'.format(vid, instruct, output)
    except:
        traceback.print_exc()
        pred_idx = 2
    
    return pred_idx


def run_inference(args):
    disable_torch_init()

    model, processor, tokenizer = model_init(args.model_path)

    answer_file = os.path.expanduser(args.answer_file)
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    ans_file = open(answer_file, "w")

    val_loader = build_mvbench_eval(args, processor['video'])

    # NOTE: only support batch size 1 for now
    for i, line in enumerate(tqdm(val_loader)):
        vid = line['video_path'][0]
        video_tensor = line['video'][0]
        task_type = line['task_type'][0]
        instruct  = line['instruct'][0]
        letters   = list(zip(*line['letters']))[0]
        options   = list(zip(*line['options']))[0]
        answer_idx = line['answer_idx'][0].item()

        output = mm_infer(
            video_tensor,
            instruct,
            model=model,
            tokenizer=tokenizer,
            modal='video',
            do_sample=False,
        )

        pred_idx = mvbench_dump(vid, instruct, letters, options, output)

        ans_file.write(json.dumps({"vid": vid, "task_type": task_type, "pred": pred_idx, "gt": answer_idx}) + '\n')

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
