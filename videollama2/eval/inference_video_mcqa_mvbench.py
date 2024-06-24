import os
import re
import math
import json
import traceback
import argparse
import warnings

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from decord import VideoReader, cpu
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append('./')
from videollama2 import model_init, x_infer
from videollama2.constants import NUM_FRAMES

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

    def __init__(self, data_list, processor, num_segments=8):
        self.data_list = data_list

        self.decord_method = {
            'video': self.read_video,
            'gif': self.read_gif,
            'frame': self.read_frame,
        }

        self.processor = processor
        self.num_segments = num_segments

    def __str__(self):
        len_list = {}
        option_list = {}
        for data in self.data_list:
            if data['task_type'] not in len_list:
                len_list[data['task_type']] = 0
            len_list[data['task_type']] += 1
            if data['task_type'] not in option_list:
                option_list[data['task_type']] = 0
            option_list[data['task_type']] += len(data['data']['candidates'])

        correct = 0
        total = 0
        res = f"There are {len(self.data_list)} videos as follow:\n"
        for k, v in len_list.items():
            correct += len_list[k]
            total += option_list[k]
            res += f"{v} for {k} ({option_list[k]} options => {len_list[k]/option_list[k]*100:.2f}%)\n"
            correct = correct + 1 / option_list[k]
        res += f"Total random accuracy: {correct/total*100:.2f}%"
        return res.rstrip()

    def __len__(self):
        return len(self.data_list)

    def get_index(self, bound, fps, max_frame, first_idx=0):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / self.num_segments
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(self.num_segments)
        ])
        return frame_indices

    def read_video(self, video_path, bound=None):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())

        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0) 
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy())
            images_group.append(img)
        # images_group = [expand2square(img, tuple(int(x*255) for x in self.processor.image_mean)) for img in images_group]
        torch_imgs = self.processor(images_group)
        return torch_imgs

    def read_gif(self, video_path, bound=None, fps=25):
        gif = imageio.get_reader(video_path)
        max_frame = len(gif) - 1
        
        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0) 
        for index, frame in enumerate(gif):
            if index in frame_indices:
                img = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                img = Image.fromarray(img)
                images_group.append(img)
        # images_group = [expand2square(img, tuple(int(x*255) for x in self.processor.image_mean)) for img in images_group]
        torch_imgs = self.processor(images_group)
        return torch_imgs

    def read_frame(self, video_path, bound=None, fps=3):
        max_frame = len(os.listdir(video_path))
        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=1) # frame_idx starts from 1
        for frame_index in frame_indices:
            img = Image.open(os.path.join(video_path, f"{frame_index:05d}.jpg"))
            images_group.append(img)
        # images_group = [expand2square(img, tuple(int(x*255) for x in self.processor.image_mean)) for img in images_group]
        torch_imgs = self.processor(images_group)
        return torch_imgs

    def qa_template(self, data):
        question = f"Question: {data['question']}\n"
        question += "Options:\n"
        answer = data['answer']
        answer_idx = -1
        for idx, c in enumerate(data['candidates']):
            question += f"({chr(ord('A') + idx)}) {c}\n"
            if c == answer:
                answer_idx = idx
        question = question.rstrip()
        answer = f"({chr(ord('A') + answer_idx)}) {answer}"
        return question, answer

    def __getitem__(self, idx):
        decord_method = self.decord_method[self.data_list[idx]['data_type']]
        bound = None
        if self.data_list[idx]['bound']:
            bound = (
                self.data_list[idx]['data']['start'],
                self.data_list[idx]['data']['end'],
            )
        video_path = os.path.join(self.data_list[idx]['prefix'], self.data_list[idx]['data']['video'])
        torch_imgs = decord_method(video_path, bound)
        question = self.data_list[idx]['data']['question']
        options = self.data_list[idx]['data']['candidates']
        answer = self.data_list[idx]['data']['answer']
        task_type = self.data_list[idx]['task_type']

        # question, answer = self.qa_template(self.data_list[idx]['data'])

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
            'letters': ','.join(letters),
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


def build_mvbench_eval(args, processor, num_frames):
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
    dataset = MVBenchDataset(data_list, processor, num_segments=num_frames)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    return dataloader


def mvbench_dump(ans_file, line, outputs):
    for idx, output in enumerate(outputs):
        vid = line['video_path'][idx]
        task_type = line['task_type'][idx]
        letters = line['letters'][idx].split(',')
        answer_idx = line['answer_idx'][idx].item()

        pred_answer = re.findall(f'[\(,\ ]*[{letters[0]}-{letters[-1]}][\),\ ]*', output)
        try:
            assert len(pred_answer) >= 1, 'The video \"{}\" output \"{}\" is not in the expected format'.format(line['video_path'], instruct + '\n' + output)
            pred_answer = pred_answer[0].strip()
            if pred_answer.startswith('('):
                pred_answer = pred_answer.strip('()')
            pred_idx = letters.index(pred_answer)
        except:
            traceback.print_exc()
            pred_idx = 2

        ans_file.write(json.dumps({"vid": vid, "task_type": task_type, "pred": pred_idx, "gt": answer_idx}) + '\n')


def run_inference(args):
    model, processor, tokenizer = model_init(args.model_path)

    num_frames = model.config.num_frames if hasattr(model.config, "num_frames") else NUM_FRAMES

    answer_file = os.path.expanduser(args.answer_file)
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    ans_file = open(answer_file, "w")

    val_loader = build_mvbench_eval(args, processor, num_frames)

    # NOTE: only support batch size 1 for now
    for i, line in enumerate(tqdm(val_loader)):
        video_tensor = line['video'][0]
        instruct = line['instruct'][0]

        pred = x_infer(
            video_tensor,
            instruct,
            mode='vanilla',
            model=model,
            tokenizer=tokenizer,
            do_sample=False
        )

        mvbench_dump(ans_file, line, [pred])

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
