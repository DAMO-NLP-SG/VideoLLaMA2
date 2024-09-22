import os
import math
import json
import random
import argparse

from tqdm import tqdm

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


def inference(args):
    disable_torch_init()
    model_path = "/mnt/zhangh/xyf/va2/output/videollama2_audio_visual_stage3_a_v_va_256_16f"
    model, processor, tokenizer = model_init(model_path)

    # if args.modal_type == "a":
    #     model.model.vision_tower = None
    # elif args.modal_type == "v":
    #     model.model.audio_tower = None
    # elif args.modal_type == "av":
    #     pass
    # else:
    #     raise NotImplementedError

    all_data = json.load(open('/mnt/zhangh/sicong/video_hallu/benchmark_final_data/all_data_final.json', 'r'))

    folder = "/mnt/zhangh/sicong/video_hallu/benchmark_final_data"

    results = open(f'results_{args.num_chunks}_{args.chunk_idx}.json', 'w')

    # preds = json.load(open('results.json', 'r'))

    # for idx, data_sample in enumerate(all_data):
    #     data_sample['pred'] = preds[idx]['answer']

    # with open(f'results_real.json', 'w') as f:
    #     json.dump(all_data, f, indent=4)

    part_data = get_chunk(all_data, args.num_chunks, args.chunk_idx)

    # exit(0)
    for data_sample in tqdm(part_data):
        video_path = None
        audio_path = None
        if "video_path" in data_sample and data_sample['video_path'] is not None:
            video_path = os.path.join(folder, data_sample['video_path'])
        if "audio_path" in data_sample and data_sample['audio_path'] is not None:
            audio_path = os.path.join(folder, data_sample['audio_path'])

        if video_path is None and audio_path is not None:
            modal_type = "a"
            audio_video_path = audio_path
        elif video_path is not None and audio_path is None:
            modal_type = "v"
            audio_video_path = video_path
        else:
            modal_type = "av"
            audio_video_path = video_path.replace('.mp4', '_with_audio.mp4')

        # audio_video_path = "/mnt/data/xyf/00000368.mp4"
        #/mnt/data/xyf/WBS4I.mp4
        #"/mnt/data/xyf/xyf/Y--ZHUMfueO0.flac"
        #"/mnt/data/xyf/AVQA/AVQA/scratch/shared/beegfs/hchen/train_data/VGGSound_final/video/"
        preprocess = processor['audio' if modal_type == "a" else "video"]
        if modal_type == "a":
            audio_video_tensor = preprocess(audio_video_path)
        else:
            audio_video_tensor = preprocess(audio_video_path, va=True if modal_type == "av" else False)

        question = data_sample['question'] + " Answer with yes or no."
        #Please describe the video with visual and audio information.
        #Please describe the video.
        #Please describe the audio.
        output = mm_infer(
            audio_video_tensor,
            question,
            model=model,
            tokenizer=tokenizer,
            modal='audio' if modal_type == "a" else "video",
            do_sample=False,
        )

        output = output.lower()

        data_sample['output'] = output

        if 'yes' in output:
            data_sample['pred'] = 'yes'
        elif 'no' in output or 'not' in output or 'did not' in output or "didn't" in output:
            data_sample['pred'] = 'no'
        else:
            data_sample['pred'] = random.choices(['yes', 'no'])[0]

        # print(question, output, data_sample['pred'])

        results.write(json.dumps(data_sample) + '\n')

    # with open(f'results_{args.chunk_idx}.json', 'w') as f:
    #     json.dump(all_data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    args = parser.parse_args()

    inference(args)
