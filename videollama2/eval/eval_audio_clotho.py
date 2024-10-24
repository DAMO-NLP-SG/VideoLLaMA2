import os
import argparse
import json
import ast
import traceback
from tqdm import tqdm
from multiprocessing.pool import Pool

from openai import AzureOpenAI


def init():
    client = AzureOpenAI(
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
        api_key=os.getenv("AZURE_OPENAI_KEY"),  
        api_version="2024-02-15-preview"
    )

    return client


def interaction(client, message_text):
    completion = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYNAME"),
        messages = message_text,
        temperature=0.7,
        max_tokens=800,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )

    return completion


def annotate_correctness(sample):
    """
    Evaluates question and answer pairs using GPT-3
    Returns a score for correctness.
    """
    question = sample['question']
    answer = sample['answer']
    assert len(answer) == 5
    pred = sample['pred']
    for i in range(5):
        try:
            message = [
                    {
                        "role": "system",
                        "content": 
                            "You are an intelligent chatbot designed for evaluating the factual accuracy of generative outputs for audio-based question-answer pairs. "
                            "Your task is to compare the predicted answer with the correct answer and determine if they are factually consistent. Here's how you can accomplish the task:"
                            "------"
                            "##INSTRUCTIONS: "
                            "- Focus on the factual consistency between the predicted answer and the correct answer. The predicted answer should not contain any misinterpretations or misinformation.\n"
                            "- The predicted answer must be factually accurate and align with the audio content.\n"
                            "- Consider synonyms or paraphrases as valid matches.\n"
                            "- Evaluate the factual accuracy of the prediction compared to the answer."
                    },
                    {
                        "role": "user",
                        "content":
                            "Please evaluate the following audio-based question-answer pair:\n\n"
                            f"Question: {question}\n"
                            f"Correct Answer: {answer[i]}\n"
                            f"Predicted Answer: {pred}\n\n"
                            "Provide your evaluation only as a factual accuracy score where the factual accuracy score is an integer value between 0 and 5, with 5 indicating the highest level of factual consistency. "
                            "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is the factual accuracy score in INTEGER, not STRING."
                            "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                            "For example, your response should look like this: {'score': 4.8}."
                    }
                ]
            completion = interaction(client, message)
            # Convert response to a Python dictionary.
            response_message = completion.choices[0].message.content
            response_dict = ast.literal_eval(response_message)
            if "score" not in sample:
                sample.update(response_dict)
            else:
                sample["score"] += response_dict["score"]
        except Exception as e:
            print(f"Error processing file '{key}': {e}")
            
    sample["score"] /= 5
    return sample



def annotate_detailed(sample):
    """
    Evaluates question and answer pairs using GPT-3
    Returns a score for correctness.
    """
    question = sample['question']
    answer = sample['answer']
    assert len(answer) == 5
    pred = sample['pred']
    for i in range(5):
        try:
            message = [
                        {
                            "role": "system",
                            "content":
                                "You are an intelligent chatbot designed for evaluating the detail orientation of generative outputs for video-based question-answer pairs. "
                                "Your task is to compare the predicted answer with the correct answer and determine its level of detail, considering both completeness and specificity. Here's how you can accomplish the task:"
                                "------"
                                "##INSTRUCTIONS: "
                                "- Check if the predicted answer covers all major points from the video. The response should not leave out any key aspects.\n"
                                "- Evaluate whether the predicted answer includes specific details rather than just generic points. It should provide comprehensive information that is tied to specific elements of the video.\n"
                                "- Consider synonyms or paraphrases as valid matches.\n"
                                "- Provide a single evaluation score that reflects the level of detail orientation of the prediction, considering both completeness and specificity."
                        },
                        {
                            "role": "user",
                            "content":
                                "Please evaluate the following video-based question-answer pair:\n\n"
                                f"Question: {question}\n"
                                f"Correct Answer: {answer[i]}\n"
                                f"Predicted Answer: {pred}\n\n"
                                "Provide your evaluation only as a detail orientation score where the detail orientation score is an integer value between 0 and 5, with 5 indicating the highest level of detail orientation. "
                                "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is the detail orientation score in INTEGER, not STRING."
                                "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                                "For example, your response should look like this: {'score': 4.8}."
                        }
                    ]
            completion = interaction(client, message)
            # Convert response to a Python dictionary.
            response_message = completion.choices[0].message.content
            response_dict = ast.literal_eval(response_message)
            if "score" not in sample:
                sample.update(response_dict)
            else:
                sample["score"] += response_dict["score"]
        except Exception as e:
            print(f"Error processing file '{key}': {e}")
    
    sample["score"] /= 5
    return sample


def main_correct(args):
    pred_contents = [eval(line) for line in open(args.pred_path, 'r').readlines()]
    
    results = []
    with Pool(processes=2) as pool:
        # tqdm用于显示进度条
        for result in tqdm(pool.imap(annotate_correctness, pred_contents), total=len(pred_contents)):
            results.append(result)

    with open(args.pred_path.replace("merge.json", "gpt_score_correct.json"), "w") as fp:
        for sample_set in results:
            fp.write(json.dumps(sample_set) + "\n")

    # Calculate average score
    score_sum = 0
    count = 0
    for sample in results:
        count += 1
        score_match = sample['score']
        score = int(score_match)
        score_sum += score
    average_score = score_sum / count

    print("Average score for correctness:", average_score)

def main_detail(args):
    pred_contents = [eval(line) for line in open(args.pred_path, 'r').readlines()]
    results = []
    with Pool(processes=2) as pool:
        # tqdm用于显示进度条
        for result in tqdm(pool.imap(annotate_detailed, pred_contents), total=len(pred_contents)):
            results.append(result)

    with open(args.pred_path.replace("merge.json", "gpt_score_detail.json"), "w") as fp:
        for sample_set in results:
            fp.write(json.dumps(sample_set) + "\n")

    # Calculate average score
    score_sum = 0
    count = 0
    for sample in results:
        count += 1
        score_match = sample['score']
        score = int(score_match)
        score_sum += score
    average_score = score_sum / count

    print("Average score for detailed:", average_score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="question-answer-generation-using-gpt-3")
    parser.add_argument("--pred-path", required=True, help="The path to file containing prediction.")
    parser.add_argument("--api-key", required=True, type=str, help="Azure Openai API key.")
    parser.add_argument("--api-endpoint", required=True, type=str, help="Azure Openai API endpoint.")
    parser.add_argument("--api-deployname", required=True, type=str, help="Azure Openai API deployname.")
    args = parser.parse_args()

    # Set the OpenAI API key.
    os.environ["AZURE_OPENAI_KEY"] = args.api_key
    os.environ["AZURE_OPENAI_ENDPOINT"] = args.api_endpoint
    os.environ["AZURE_OPENAI_DEPLOYNAME"] = args.api_deployname

    client = init()

    main_correct(args)

    main_detail(args)
