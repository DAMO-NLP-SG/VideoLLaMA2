"""
A model worker executes the model.
"""
import os
import json
import time
import uuid
import asyncio
import requests
import argparse
import threading
from threading import Thread
from functools import partial
from typing import Iterator, List, Optional, Tuple

import uvicorn
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse

import torch
import decord
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
from transformers import TextIteratorStreamer

from videollama2.constants import WORKER_HEART_BEAT_INTERVAL
from videollama2.utils import (build_logger, server_error_msg, pretty_print_semaphore)
from videollama2.model.builder import load_pretrained_model
from videollama2.mm_utils import process_images, process_videos, load_image_from_base64, tokenizer_image_token, KeywordsStoppingCriteria, tokenizer_MMODAL_token
from videollama2.mm_utils import chunk_list, frame_expansion
from videollama2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_VIDEO_TOKEN, NUM_FRAMES, MMODAL_TOKEN_INDEX


GB = 1 << 30

worker_id = str(uuid.uuid4())[:6]
logger = build_logger("model_worker", f"model_worker_{worker_id}.log")
global_counter = 0

model_semaphore = None


# variable_content = os.getenv('MY_VARIABLE', '')
# KEYWORDS_LIST = set(variable_content.split('\n'))
KEYWORDS_LIST = []
path = 'assets/keywords.txt'
if os.path.exists(path):
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            
            KEYWORDS_LIST.append(line.strip()) 
else:
    KEYWORDS_LIST = []


KEYWORD_BLOCK_MESSAGE2 = "The output contains political, erotic and other unsafe content that violates local laws. Please re-enter your question."
KEYWORD_BLOCK_MESSAGE1 = "Your input question contains political, erotic and other unsafe content that violates local laws. Please re-enter your question."
STREAM_CHECK_MULTIPLE = 20


def heart_beat_worker(controller):

    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        controller.send_heart_beat()


def safety_check(text, history=None, ) -> Optional[str]:

    if len(KEYWORDS_LIST) > 0 and any(x in text.lower() for x in KEYWORDS_LIST):
        print('############')
        return KEYWORD_BLOCK_MESSAGE2
    
    return None


def input_safety_check(text) -> Optional[str]:
    if len(KEYWORDS_LIST) > 0 and any(x in text.lower() for x in KEYWORDS_LIST):
        print('######## Input keyword alarm triggered:', text)
        return KEYWORD_BLOCK_MESSAGE1
    return None


class ModelWorker:

    def __init__(self, controller_addr, worker_addr,
                 worker_id, no_register,
                 model_path, model_base, model_name,
                 load_8bit, load_4bit, device):
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        self.model_path = model_path
        if model_path.endswith("/"):
            model_path = model_path[:-1]
        if model_name is None:
            model_paths = model_path.split("/")
            if model_paths[-1].startswith('checkpoint-'):
                self.model_name = model_paths[-2] + "_" + model_paths[-1]
            else:
                self.model_name = model_paths[-1]
        else:
            self.model_name = model_name

        self.device = device
        logger.info(f"Loading the model {self.model_name} on worker {worker_id} ...")
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, model_base, self.model_name, load_8bit, load_4bit, device=self.device)
        self.is_multimodal = 'videollama2' in self.model_name.lower() or 'vlb' in self.model_name.lower()

        if not no_register:
            self.register_to_controller()
            self.heart_beat_thread = threading.Thread(
                target=heart_beat_worker, args=(self,))
            self.heart_beat_thread.start()

    def register_to_controller(self):
        logger.info("Register to controller")

        url = self.controller_addr + "/register_worker"
        data = {
            "worker_name": self.worker_addr,
            "check_heart_beat": True,
            "worker_status": self.get_status()
        }
        r = requests.post(url, json=data)
        assert r.status_code == 200

    def send_heart_beat(self):
        logger.info(f"Send heart beat. Models: {[self.model_name]}. "
                    f"Semaphore: {pretty_print_semaphore(model_semaphore)}. "
                    f"global_counter: {global_counter}")

        url = self.controller_addr + "/receive_heart_beat"

        while True:
            try:
                ret = requests.post(url, json={
                    "worker_name": self.worker_addr,
                    "queue_length": self.get_queue_length()}, timeout=5)
                exist = ret.json()["exist"]
                break
            except requests.exceptions.RequestException as e:
                logger.error(f"heart beat error: {e}")
            time.sleep(5)

        if not exist:
            self.register_to_controller()

    def get_queue_length(self):
        if model_semaphore is None:
            return 0
        else:
            return args.limit_model_concurrency - model_semaphore._value + (len(
                model_semaphore._waiters) if model_semaphore._waiters is not None else 0)

    def get_status(self):
        return {
            "model_names": [self.model_name],
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }

    @torch.inference_mode()
    def generate_stream(self, params):
        tokenizer, model, image_processor = self.tokenizer, self.model, self.image_processor

        prompt = params["prompt"]
        ori_prompt = prompt
        images_or_videos = params.get("images", None)
        #print("Input images:", images_or_videos)
        num_image_tokens = 0
        modal_list = []
        if images_or_videos is not None and len(images_or_videos) and self.is_multimodal:
            if len(images_or_videos) > 0:
                if len(images_or_videos) != prompt.count(DEFAULT_IMAGE_TOKEN) and len(images_or_videos) != (prompt.count(DEFAULT_VIDEO_TOKEN)):
                    raise ValueError("Number of images/videos does not match number of <image>/<video> tokens in prompt")
                
                try:
                    print("Load image...")
                    images_or_videos = [load_image_from_base64(image) for image in images_or_videos]
                    images_or_videos = process_images(images_or_videos, image_processor, model.config)
                    
                    modal_list = ["image"]
                    replace_token = DEFAULT_IMAGE_TOKEN
                    modal_token_index = MMODAL_TOKEN_INDEX["IMAGE"]
                except:
                    print("Load video instead...")
                    decord_vr = VideoReader(uri=images_or_videos[0], ctx=cpu(0))
                    duration = len(decord_vr)
                    if not "use_taug" in self.model_path:
                        frame_id_list = np.linspace(0, duration-1, 8, dtype=int)
                        video_frames = decord_vr.get_batch(frame_id_list).asnumpy()
                        images_or_videos = process_videos(video_frames, image_processor, model.config)
                    else:
                        print("Temporal augmentation activated!!!")
                        frame_id_list = np.linspace(0, duration-1, 8 * 2 * 2, dtype=int)
                        video_data = decord_vr.get_batch(frame_id_list)
                        video_frames = [Image.fromarray(f) for f in video_data.asnumpy()]
                        chunked_video_frames = chunk_list(video_frames, 2*2)
                        expanded_video_frames = [frame_expansion(frame_list, 2) for frame_list in chunked_video_frames]
                        images_or_videos = process_videos(expanded_video_frames, image_processor, model.config)

                    # frame_id_list = np.linspace(0, duration-1, NUM_FRAMES, dtype=int)
                    # images_or_videos = decord_vr.get_batch(frame_id_list).asnumpy()
                    # images_or_videos = process_videos(images_or_videos, image_processor, model.config)
                    #print("images_or_videos.shape:", images_or_videos.shape)
                    modal_list = ["video"]
                    replace_token = DEFAULT_VIDEO_TOKEN
                    modal_token_index = MMODAL_TOKEN_INDEX["VIDEO"]
                
                if type(images_or_videos) is list:
                    images_or_videos = [image.to(self.model.device, dtype=torch.float16) for image in images_or_videos]
                else:
                    images_or_videos = images_or_videos.to(self.model.device, dtype=torch.float16)
                    if modal_list[0] == "video":
                        print("Video:", images_or_videos.shape)
                        images_or_videos = [images_or_videos]
                    else:
                        print("Image:", images_or_videos.shape)


                #image_sizes = [image.size for image in images_or_videos]
                

                # if len(images_or_videos) % NUM_FRAMES == 0:
                #     images_or_videos = process_images(images_or_videos, image_processor, model.config)
                #     #images_or_videos = [image.to(self.model.device, dtype=torch.float16) for image in images_or_videos]
                #     #modal_list = ["image"] * len(images_or_videos)
                #     images_or_videos = images_or_videos.to(self.model.device, dtype=torch.float16)
                #     modal_list = ["video"]
                #     replace_token = DEFAULT_VIDEO_TOKEN
                # else:
                    
                if getattr(self.model.config, 'mm_use_im_start_end', False):
                    replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

                num_image_tokens = prompt.count(replace_token) * model.get_vision_tower().num_patches
            else:
                images = None
                modal_list = []
            image_args = {"images_or_videos": images_or_videos, "modal_list": modal_list}
        else:
            images = None
            image_args = {}
        print("image_args:", image_args)
        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        max_context_length = getattr(model.config, 'max_position_embeddings', 2048)
        max_new_tokens = min(int(params.get("max_new_tokens", 256)), 1024)
        stop_str = params.get("stop", None)
        do_sample = True if temperature > 0.001 else False

        #input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
        # tokenizer for our video-llama beta
        input_ids = tokenizer_MMODAL_token(prompt, tokenizer, modal_token_index, return_tensors='pt').unsqueeze(0).to(self.device)
        #print("Current prompt:", prompt)
        #print("input_ids.shape:", input_ids.shape)
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=15)

        max_new_tokens = min(max_new_tokens, max_context_length - input_ids.shape[-1] - num_image_tokens)

        if max_new_tokens < 1:
            yield json.dumps({"text": ori_prompt + "Exceeds max token length. Please start a new conversation, thanks.", "error_code": 0}).encode() + b"\0"
            return

        thread = Thread(target=model.generate, kwargs=dict(
            inputs=input_ids,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            stopping_criteria=[stopping_criteria],
            use_cache=True,
            **image_args
        ))
        thread.start()

        generated_text = ori_prompt
        token_count = 0
        for new_text in streamer:
            generated_text += new_text
            token_count += len(tokenizer.encode(new_text))
            if token_count >= STREAM_CHECK_MULTIPLE:
                safety_message = safety_check(generated_text)
                if safety_message:
                    print('####### Keyword alarm triggered:', generated_text)
                    yield json.dumps({"text": safety_message , "error_code": 1}).encode() + b"\0"
                    return  
                token_count = 0  # 
                

            if generated_text.endswith(stop_str):
                generated_text = generated_text[:-len(stop_str)]
            yield json.dumps({"text": generated_text, "error_code": 0}).encode() + b"\0"

    def generate_stream_gate(self, params):
        try:      
            input_text = params.get("prompt", "")
            safety_message = input_safety_check(input_text)
            if safety_message:
                yield json.dumps({"text": safety_message, "error_code": 1}).encode() + b"\0"
                return

            for x in self.generate_stream(params):
                yield x
        except ValueError as e:
            print("Caught ValueError:", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"
        except torch.cuda.CudaError as e:
            print("Caught torch.cuda.CudaError:", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"
        except Exception as e:
            print("Caught Unknown Error", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"


app = FastAPI()


def release_model_semaphore(fn=None):
    model_semaphore.release()
    if fn is not None:
        fn()


@app.post("/worker_generate_stream")
async def generate_stream(request: Request):
    global model_semaphore, global_counter
    global_counter += 1
    params = await request.json()

    if model_semaphore is None:
        model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)
    await model_semaphore.acquire()
    worker.send_heart_beat()
    generator = worker.generate_stream_gate(params)
    background_tasks = BackgroundTasks()
    background_tasks.add_task(partial(release_model_semaphore, fn=worker.send_heart_beat))
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_get_status")
async def get_status(request: Request):
    return worker.get_status()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str, default="http://localhost:21002")
    parser.add_argument("--controller-address", type=str, default="http://localhost:21001")
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--multi-modal", action="store_true", help="Multimodal mode is automatically detected with model name, please make sure `llava` is included in the model path.")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--stream-interval", type=int, default=1)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    if args.multi_modal:
        logger.warning("Multimodal mode is automatically detected with model name, please make sure `llava` is included in the model path.")

    worker = ModelWorker(args.controller_address,
                         args.worker_address,
                         worker_id,
                         args.no_register,
                         args.model_path,
                         args.model_base,
                         args.model_name,
                         args.load_8bit,
                         args.load_4bit,
                         args.device)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
