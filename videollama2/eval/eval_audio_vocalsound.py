import os
import ast
import json
import time
import argparse
import traceback
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import accuracy_score

label_mapping = {
    '(A)': 'Laughter',
    '(B)': 'Sigh',
    '(C)': 'Cough',
    '(D)': 'Throat clearing',
    '(E)': 'Sneeze',
    '(F)': 'Sniff',
    'A': 'Laughter',
    'B': 'Sigh',
    'C': 'Cough',
    'D': 'Throat clearing',
    'E': 'Sneeze',
    'F': 'Sniff',
}


def main(args):

    file = open(args.pred_path)
    new_pred_contents = [eval(i.strip()) for i in file.readlines()]

    # Calculate average score and accuracy
    correct_count = 0

    for sample in tqdm(new_pred_contents):

        prediction = sample['pred'].strip().upper()
        answer = sample['answer'].upper()

        if answer in prediction:
            correct_count += 1
        elif prediction in label_mapping and answer in label_mapping[prediction].upper():
            correct_count += 1

    # Calculate accuracy for all responses using sklearn's accuracy_score
    print(f'Overall Accuracy: {correct_count / len(new_pred_contents):.2%}', f'Total Responses: {len(new_pred_contents)}')  # Formatted as percentage


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="question-answer")
    parser.add_argument("--pred-path", required=True, help="The path to file containing prediction.")
    args = parser.parse_args()

    main(args)
