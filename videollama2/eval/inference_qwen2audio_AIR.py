from io import BytesIO
from urllib.request import urlopen
import librosa
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
import os
import json
import math
import argparse
import warnings
import traceback
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

# NOTE: Ignore TypedStorage warning, which refers to this link~(https://github.com/pytorch/pytorch/issues/97207#issuecomment-1494781560)
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]



class AIRDataset(Dataset):

    audoi_formats = ['.wav', '.flac']

    def __init__(self, questions, processor):
        self.questions = questions
        self.processor = processor

    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        sample = self.questions[idx]

        audio_path  = sample['audio']
        
        question_id = sample['id']
        answer      = sample['answer']

        wrapped_question = sample['query']

        conversation = [
            {'role': 'system', 'content': 'You are a helpful assistant.'}, 
            {"role": "user", "content": [
                {"type": "audio", "audio_url": audio_path},
                {"type": "text", "text": wrapped_question},
            ]},
        ]
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios = []
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if ele["type"] == "audio":
                        audios.append(
                            librosa.load(
                                ele['audio_url'], 
                                sr=self.processor.feature_extractor.sampling_rate)[0]
                        )

        inputs = self.processor(text=text, audios=audios, return_tensors="pt", padding=True, sampling_rate=self.processor.feature_extractor.sampling_rate)
    
        return {
            'audio':       inputs,
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
    import csv

    # Initialize the model
    model = Qwen2AudioForConditionalGeneration.from_pretrained(args.model_path).to(args.device)
    processor = AutoProcessor.from_pretrained(args.model_path)
    
    gt_answer = {x["uniq_id"]: x for x in json.load(open(args.answer_file, "r"))}
    gt_questions = []
    with open(args.question_file, "r") as fp:
        for x in fp.readlines():
            gt_questions.append(json.loads(x))
            gt_questions[-1]["answer"] = gt_answer[gt_questions[-1]["id"]]["answer_gt"]
    gt_questions = get_chunk(gt_questions, args.num_chunks, args.chunk_idx)
    dataset = AIRDataset(gt_questions, processor)

    assert args.batch_size == 1, "Batch size must be 1 for inference"
   
    dataloader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn)

    answer_file = os.path.join(args.output_file)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    ans_file = open(answer_file, "w")

    # Iterate over each sample in the ground truth file
    for i, (audio_tensors, audio_names, questions, question_ids, answers) in enumerate(tqdm(dataloader)):
        audio_tensor = audio_tensors[0].to(args.device)
        audio_name   = audio_names[0]
        question     = questions[0]
        question_id  = question_ids[0]
        answer       = answers[0]

        generated_ids = model.generate(**audio_tensor, max_length=256)
        generated_ids = generated_ids[:, audio_tensor.input_ids.size(1):]
        response = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        sample_set = {'id': question_id, 'question': question, 'answer': answer, 'pred': response}
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
