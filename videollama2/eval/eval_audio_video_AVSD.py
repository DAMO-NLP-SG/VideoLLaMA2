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


def annotate(sample):
    """
    Evaluates question and answer pairs using GPT-3
    Returns a score for correctness.
    """
    question = sample['question']
    answer = sample['answer']
    pred = sample['pred']
    try:
        message=[
                {
                    "role": "system",
                    "content": 
                        "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
                        "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
                        "------"
                        "##INSTRUCTIONS: "
                        "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
                        "- Consider synonyms or paraphrases as valid matches.\n"
                        "- Evaluate the correctness of the prediction compared to the answer."
                },
                {
                    "role": "user",
                    "content":
                        "Please evaluate the following video-based question-answer pair:\n\n"
                        f"Question: {question}\n"
                        f"Correct Answer: {answer}\n"
                        f"Predicted Answer: {pred}\n\n"
                        "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
                        "Please generate the response in the form of a Python dictionary string with keys 'binary_pred' and 'score', where value of 'binary_pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
                        "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                        "For example, your response should look like this: {'binary_pred': 'yes', 'score': 4}."
                }
            ]
        completion = interaction(client, message)
        # Convert response to a Python dictionary.
        response_message = completion.choices[0].message.content
        response_dict = ast.literal_eval(response_message)
        sample.update(response_dict)
        return sample

    except Exception as e:
        print(f"Error processing file '{key}': {e}")




def main(args):
    pred_contents = [eval(line) for line in open(args.pred_path, 'r').readlines()]
  
    results = []
    with Pool(processes=1) as pool:
        # tqdm用于显示进度条
        for result in tqdm(pool.imap(annotate, pred_contents), total=len(pred_contents)):
            results.append(result)

    with open(args.pred_path.replace("merge.json", "gpt_score.json"), "w") as fp:
        for sample_set in results:
            fp.write(json.dumps(sample_set) + "\n")
            
    # Calculate average score
    score_sum = 0
    count = 0
    corr = 0
    for sample in results:
        try:
            count += 1
            score_match = sample['score']
            score = int(score_match)
            score_sum += score
            if sample['binary_pred'] == "yes":
                corr += 1
        except:
            print(sample)
            print("json format error")
            continue
    average_score = score_sum / count
    acc = corr / count

    print("Average score for correctness:", average_score)
    print("Accuracy for correctness:", acc)


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

    main(args)
