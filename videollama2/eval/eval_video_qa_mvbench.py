import json
import argparse
from tabulate import tabulate


tasks = {
    "Action Sequence": ("action_sequence.json", "star/Charades_v1_480/", "video", True), # has start & end
    "Action Prediction": ("action_prediction.json", "star/Charades_v1_480/", "video", True), # has start & end
    "Action Antonym": ("action_antonym.json", "ssv2_video/", "video", False),
    "Fine-grained Action": ("fine_grained_action.json", "pMoments_in_Time_Raw/videos/", "video", False),
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


def main():
    args = parse_args()
    res = [eval(x.strip()) for x in open(args.pred_path, 'r').readlines()]
    task_types = tasks.keys()
    task_acc = {x: [] for x in task_types}
    acc = []
    for i, x in enumerate(res):
        value = 1
        if x['pred'] != x['gt']:
            value = 0
        acc.append(value)
        task_acc[x['task_type']].append(value)
    acc = sum(acc) * 100 / len(acc)
    task_acc = {x: sum(task_acc[x]) * 100 / len(task_acc[x]) for x in task_acc}
    print(f"{args.pred_path}:", acc)
    task_names = list(tasks.keys())
    
    table_data = []
    for i in range(len(task_names) // 4):
        row_task_names = task_names[i * 4: (i + 1) * 4]
        row_task_acc = [task_acc[x] for x in row_task_names]
        table_data.append(row_task_names)
        table_data.append(row_task_acc)
    print(tabulate(table_data, floatfmt=".1f"), '\n')


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate video captioning.")
    parser.add_argument("--pred_path", default=r'', help="The path to file containing prediction.")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
