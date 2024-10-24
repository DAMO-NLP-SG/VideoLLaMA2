import os
import ast
import json
import time
import argparse
import traceback
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import accuracy_score


def main(args):

    file = open(args.pred_path)
    new_pred_contents = [eval(i.strip()) for i in file.readlines()]

    # Calculate average score and accuracy
    binary_refs = []
    binary_hyps = []
    all_refs = []
    all_hyps = []
   
    for sample in tqdm(new_pred_contents):
        # Computing accuracy
        all_refs.append(sample['answer'].upper())
        all_hyps.append(sample['pred'].rstrip(' .').upper())

        if sample['answer'].lower() in ["yes", "no"]:
            binary_refs.append(sample['answer'].lower())
            binary_hyps.append(sample['pred'].lower())

    # Calculate accuracy for all responses using sklearn's accuracy_score
    overall_accuracy = accuracy_score(all_refs, all_hyps)
    print(f'Overall Accuracy: {overall_accuracy:.2%}', f'Total Responses: {len(all_refs)}')  # Formatted as percentage
    # Calculate binary accuracy
    bi_score = accuracy_score(binary_refs, binary_hyps)
    print(f'Binary Accuracy: {bi_score:.2%}', f'Total Binary Responses: {len(binary_refs)}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="question-answer")
    parser.add_argument("--pred-path", required=True, help="The path to file containing prediction.")
    args = parser.parse_args()

    main(args)
