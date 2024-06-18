import ast
import math
import base64
from io import BytesIO

import torch
import decord
import imageio
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
from moviepy.editor import VideoFileClip
from transformers import StoppingCriteria

from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from scenedetect.stats_manager import StatsManager

from .constants import NUM_FRAMES, MAX_FRAMES, NUM_FRAMES_PER_SECOND, MMODAL_INDEX_TOKEN, IMAGE_TOKEN_INDEX


def merge_scenes(cut_list, cut_scores, scene_list,num_frames,max_scene_num=4, num_frame_per_scene=8, min_frames_per_scene=30):
    if len(scene_list) == len(cut_list) and len(scene_list) == 0:
        frame_ids = np.linspace(0, num_frames-1, num_frame_per_scene, dtype=int)  # only one scene for current video
        return [frame_ids]

    scene_list, cut_results = merge_scenes_not_exeed_max_scene_num(cut_list,cut_scores,scene_list, max_scene_num)

    prev_cut_point = 0
    list_of_scene_frames = [] 
    for (cur_cut_point, _) in cut_results:
        frame_ids = list(np.linspace(prev_cut_point, cur_cut_point-1, num_frame_per_scene, dtype=int))
        list_of_scene_frames.append(frame_ids)
        prev_cut_point = cur_cut_point
    if cur_cut_point < num_frames:
        frame_ids = np.linspace(cur_cut_point, num_frames-1, num_frame_per_scene, dtype=int)
        list_of_scene_frames.append(frame_ids)

    return list_of_scene_frames


def merge_scenes_not_exeed_max_scene_num(cut_list,cut_scores, scene_list, max_scene_num):
    cut_frames = [ele.get_frames() for ele in cut_list]
    cut_results = list(zip(cut_frames, cut_scores))
    while len(scene_list) > max_scene_num:
        min_idx = np.argmin(cut_scores)
        cut_frames = [ele for idx, ele in enumerate(cut_frames) if idx != min_idx]
        cut_scores = [ele for idx, ele in enumerate(cut_scores) if idx != min_idx]

        # merge scene list
        num_scenes = len(scene_list)
        #print("Current min_idx:", min_idx)
        s1 = scene_list[min_idx]
        s2 = scene_list[min_idx+1]
        new_scene = (s1[0], s2[1])
        if min_idx == 0:
            # merge the first two scenes
            new_scene_list = [new_scene] + scene_list[2:]
        elif min_idx == num_scenes - 1:
            # # merge the last two scenes
            new_scene_list = scene_list[:min_idx-1] + [new_scene]
        else:
            new_scene_list = scene_list[:min_idx] + [new_scene] + scene_list[min_idx+2:]
        scene_list = new_scene_list
        cut_results = list(zip(cut_frames, cut_scores))
    return scene_list, cut_results


def split_video_into_scenes(video_path, threshold=27.0, max_scene_num=10, num_frame_per_scene=8):
    # Open video, create a scene manager, and add a detector.
    video = open_video(video_path)
    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)
    detector = ContentDetector(threshold=threshold)
    scene_manager.add_detector(detector)
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()
    cut_list = scene_manager.get_cut_list()
    num_frames = video.duration.get_frames()
    if len(scene_list) == len(cut_list) and len(scene_list) == 0:
        frame_ids = np.linspace(0, num_frames-1, num_frame_per_scene, dtype=int)  # only one scene for current video
        return [frame_ids]
    assert len(scene_list) == len(cut_list) + 1, f"inconsistent lengths for scene list ({len(scene_list)}) vs. cut list ({len(cut_list)})"
    cut_frames = [ele.get_frames() for ele in cut_list]
    cut_scores = [stats_manager.get_metrics(f, ["delta_lum"])[0] for f in cut_frames]
    cut_results = list(zip(cut_frames, cut_scores))
    #print(f"Original cut scores: {cut_scores}, original scene list: {scene_list}")
    while len(scene_list) > max_scene_num:
        min_idx = np.argmin(cut_scores)
        cut_frames = [ele for idx, ele in enumerate(cut_frames) if idx != min_idx]
        cut_scores = [ele for idx, ele in enumerate(cut_scores) if idx != min_idx]

        # merge scene list
        num_scenes = len(scene_list)
        #print("Current min_idx:", min_idx)
        s1 = scene_list[min_idx]
        s2 = scene_list[min_idx+1]
        new_scene = (s1[0], s2[1])
        if min_idx == 0:
            # merge the first two scenes
            new_scene_list = [new_scene] + scene_list[2:]
        elif min_idx == num_scenes - 1:
            # # merge the last two scenes
            new_scene_list = scene_list[:min_idx-1] + [new_scene]
        else:
            new_scene_list = scene_list[:min_idx] + [new_scene] + scene_list[min_idx+2:]
        scene_list = new_scene_list
        cut_results = list(zip(cut_frames, cut_scores))
    #print(f"Cut scores after merging: {cut_scores}, scene list: {scene_list}")
    prev_cut_point = 0
    list_of_scene_frames = [] 
    for (cur_cut_point, _) in cut_results:
        frame_ids = list(np.linspace(prev_cut_point, cur_cut_point-1, num_frame_per_scene, dtype=int))
        list_of_scene_frames.append(frame_ids)
        prev_cut_point = cur_cut_point
    if cur_cut_point < num_frames:
        frame_ids = np.linspace(cur_cut_point, num_frames-1, num_frame_per_scene, dtype=int)
        list_of_scene_frames.append(frame_ids)
    # print(f"Finally got {len(list_of_scene_frames)} scenes where we evenly sampled {num_frame_per_scene} frames for each scene")
    return list_of_scene_frames


def select_best_resolution(original_size, possible_resolutions):
    """
    Selects the best resolution from a list of possible resolutions based on the original size.
    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].
    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float('inf')
    for width, height in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution
        if effective_resolution > max_effective_resolution or (effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)
    return best_fit


def resize_and_pad_image(image, target_resolution):
    """
    Resize and pad an image to a target resolution while maintaining aspect ratio.
    Args:
        image (PIL.Image.Image): The input image.
        target_resolution (tuple): The target resolution (width, height) of the image.
    Returns:
        PIL.Image.Image: The resized and padded image.
    """
    original_width, original_height = image.size
    target_width, target_height = target_resolution
    scale_w = target_width / original_width
    scale_h = target_height / original_height
    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)
    # Resize the image
    resized_image = image.resize((new_width, new_height))
    new_image = Image.new('RGB', (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))
    return new_image


def divide_to_patches(image, patch_size):
    """
    Divides an image into patches of a specified size.
    Args:
        image (PIL.Image.Image): The input image.
        patch_size (int): The size of each patch.
    Returns:
        list: A list of PIL.Image.Image objects representing the patches.
    """
    patches = []
    width, height = image.size
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            patch = image.crop(box)
            patches.append(patch)
    return patches


def get_anyres_image_grid_shape(image_size, grids, patch_size):
    """
    Calculate the shape of the image patch grid after the preprocessing for images of any resolution.
    Args:
        image_size (tuple): The size of the input image in the format (width, height).
        grids (str, List[tuple[int]]): Patch segmentation grid.
        patch_size (int): The size of each image patch.
    Returns:
        tuple: The shape of the image patch grid in the format (width, height).
    """
    if type(grids) is list:
        possible_resolutions = [(x * patch_size, y * patch_size) for x, y in grids]
    else:
        possible_resolutions = [(x * patch_size, y * patch_size) for x, y in ast.literal_eval(grids)]
    width, height = select_best_resolution(image_size, possible_resolutions)
    return width // patch_size, height // patch_size


def process_anyres_image(image, grids, patch_size):
    """
    Process an image with variable resolutions.
    Args:
        image (PIL.Image.Image): The input image to be processed.
        grids (str, List[tuple[int]]): Patch segmentation grid.
        patch_size (int): The size of the patches to be extracted.
    Returns:
        torch.Tensor: A tensor containing the processed image patches.
    """
    if type(grids) is list:
        possible_resolutions = [(x * patch_size, y * patch_size) for x, y in grids]
    else:
        possible_resolutions = [(x * patch_size, y * patch_size) for x, y in ast.literal_eval(grids)]
    best_resolution = select_best_resolution(image.size, possible_resolutions)
    image_padded = resize_and_pad_image(image, best_resolution)
    patches = divide_to_patches(image_padded, patch_size)
    image_original_resize = resize_and_pad_image(image, (patch_size, patch_size))
    image_patches = [image_original_resize] + patches
    return image_patches


def chunk_list(input_list, chunk_size):
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]


def frame_expansion(frame_list, n):
    assert len(frame_list) == n * n
    width, height = frame_list[0].width, frame_list[0].height
    expanded_width = n * width
    expanded_height = n * height
    expanded_frame = Image.new('RGB', (expanded_width, expanded_height))
    for i in range(n):
        for j in range(n):
            frame = frame_list[i * n + j]
            coordinate = (j*width, i*height)
            expanded_frame.paste(frame, coordinate)
    return expanded_frame


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def process_images(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    #print("Current image_aspect_ratio:", image_aspect_ratio)
    if image_aspect_ratio == 'pad':
        for image in images:
            image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            new_images.append(image)
    else:
        return image_processor(images, return_tensors='pt')['pixel_values']
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images


def process_videos(frames, image_processor, model_cfg):
    # this function only used during inference
    # image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    # new_frames = []
    # print("Current image_aspect_ratio:", image_aspect_ratio)
    # if image_aspect_ratio == 'pad':
    #     for image in frames:
    #         image = Image.fromarray(image)
    #         image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
    #         image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
    #         new_frames.append(image)
    # else:
    #     return image_processor(frames, return_tensors='pt')['pixel_values']
    # if all(x.shape == new_frames[0].shape for x in new_frames):
    #     new_frames = torch.stack(new_frames, dim=0)
    new_frames = image_processor.preprocess(frames, return_tensors='pt')['pixel_values']  # do not pad for video frames
    return new_frames


def create_photo_grid(arr, rows=None, cols=None):
    """
    Create a photo grid from a 4D numpy array with shape [t, h, w, c].

    Parameters:
        arr (numpy.ndarray): Input array with shape [t, h, w, c].
        rows (int): Optional. Number of rows in the grid. If not set, it will be determined based on `cols` or the square root of `t`.
        cols (int): Optional. Number of columns in the grid. If not set, it will be determined based on `rows` or the square root of `t`.

    Returns:
        numpy.ndarray: A 3D numpy array representing the photo grid.
    """

    if isinstance(arr, list):
        if isinstance(arr[0], Image.Image):
            arr = np.stack([np.array(img) for img in arr])
        elif isinstance(arr[0], np.ndarray):
            arr = np.stack(arr)
        else:
            raise ValueError("Invalid input type. Expected list of Images or numpy arrays.")

    t, h, w, c = arr.shape
    
    # Calculate the number of rows and columns if not provided
    if rows is None and cols is None:
        rows = math.ceil(math.sqrt(t))
        cols = math.ceil(t / rows)
    elif rows is None:
        rows = math.ceil(t / cols)
    elif cols is None:
        cols = math.ceil(t / rows)

    # Check if the grid can hold all the images
    if rows * cols < t:
        raise ValueError(f"Not enough grid cells ({rows}x{cols}) to hold all images ({t}).")
    
    # Create the grid array with appropriate height and width
    grid_height = h * rows
    grid_width = w * cols
    grid = np.zeros((grid_height, grid_width, c), dtype=arr.dtype)
    
    # Fill the grid with images
    for i in range(t):
        row_idx = i // cols
        col_idx = i % cols
        grid[row_idx*h:(row_idx+1)*h, col_idx*w:(col_idx+1)*w, :] = arr[i]
    
    return grid


def process_image(image_path, processor, aspect_ratio='pad', num_frames=NUM_FRAMES, image_grid=False):
    image = Image.open(image_path).convert('RGB')

    if image_grid:
        pg = np.stack([np.array(image)] * num_frames)
        grid_h = grid_w = math.ceil(math.sqrt(num_frames))
        pg = create_photo_grid(pg, grid_h, grid_w)
        images = [pg, np.array(image)]
    else:
        images = [np.array(image)]

    if aspect_ratio == 'pad':
        images = [Image.fromarray(f) for f in images]
        images = [expand2square(image, tuple(int(x*255) for x in processor.image_mean)) for image in images]
    else:
        images = [Image.fromarray(f) for f in images]

    images = processor.preprocess(images, return_tensors='pt')['pixel_values']
    return images


def process_video(video_path, processor, aspect_ratio='pad', num_frames=NUM_FRAMES, image_grid=False, sample_scheme='uniform'):
    def frame_sample(duration, mode='uniform', local_fps=None):
        if mode == 'uniform':
            # Calculate the size of each segment from which a frame will be extracted
            seg_size = float(duration - 1) / num_frames

            frame_ids = []
            for i in range(num_frames):
                # Calculate the start and end indices of each segment
                start = int(np.round(seg_size * i))
                end = int(np.round(seg_size * (i + 1)))
                # Append the middle index of the segment to the list
                frame_ids.append((start + end) // 2)

            return frame_ids
            # NOTE: old version
            # return np.linspace(0, duration-1, num_frames, dtype=int)
        elif mode == 'fps':
            assert local_fps is not None
            segment_len = min(local_fps // NUM_FRAMES_PER_SECOND, duration)
            return np.arange(segment_len // 2, duration, segment_len, dtype=int)
        else:
            raise ImportError(f'Unsupported frame sampling mode: {mode}')

    if isinstance(video_path, str):
        if video_path.endswith('.gif'):
            video_gif = imageio.get_reader(video_path)
            duration, local_fps = len(video_gif), 10

            frame_id_list = frame_sample(duration, mode=sample_scheme, local_fps=local_fps)
            # limit the max input frames
            if len(frame_id_list) > MAX_FRAMES:
                frame_id_list = np.linspace(0, duration-1, MAX_FRAMES, dtype=int)
            video_data = [frame for index, frame in enumerate(video_gif) if index in frame_id_list]
        # added by lixin4ever, include the support of .webm files from sthsthv2
        elif video_path.endswith('.webm'):
            video_webm = VideoFileClip(video_path)
            video_frames = np.array(list(video_webm.iter_frames()))

            duration, local_fps = len(video_frames), video_webm.fps

            frame_id_list = frame_sample(duration, mode=sample_scheme, local_fps=local_fps)
            # limit the max input frames
            if len(frame_id_list) > MAX_FRAMES:
                frame_id_list = np.linspace(0, duration-1, MAX_FRAMES, dtype=int)
            video_data = video_frames[frame_id_list]
        else:
            # NOTE: num_threads=1 is required to avoid deadlock in multiprocessing
            decord_vr = VideoReader(uri=video_path, ctx=cpu(0), num_threads=1) 
            duration, local_fps = len(decord_vr), float(decord_vr.get_avg_fps())
        
            frame_id_list = frame_sample(duration, mode=sample_scheme, local_fps=local_fps)
            # limit the max input frames
            if len(frame_id_list) > MAX_FRAMES:
                frame_id_list = np.linspace(0, duration-1, MAX_FRAMES, dtype=int)
            try:
                video_data = decord_vr.get_batch(frame_id_list).numpy()
            except:
                video_data = decord_vr.get_batch(frame_id_list).asnumpy()

            # if self.data_args.use_temp_aug:
            #     frame_id_list = np.linspace(0, duration-1, num_frames * 2 * 2, dtype=int)
            #     video_data = decord_vr.get_batch(frame_id_list)
            #     video_frames = [Image.fromarray(f) for f in video_data.numpy()]
            #     chunked_video_frames = chunk_list(video_frames, 2*2)
            #     video_data = [frame_expansion(frame_list, 2) for frame_list in chunked_video_frames]
    elif isinstance(video_path, np.ndarray):
        assert len(video_path) == num_frames
        video_data = video_path
    elif isinstance(video_path, list):
        assert len(video_path) == num_frames
        video_data = np.stack([np.array(x) for x in video_path])

    if image_grid:
        grid_h = grid_w = math.ceil(math.sqrt(num_frames))
        pg = create_photo_grid(video_data, grid_h, grid_w)
        video_data = [pg, *video_data]

    if aspect_ratio == 'pad':
        images = [Image.fromarray(f.numpy() if isinstance(f, torch.Tensor) else f) for f in video_data]
        images = [expand2square(image, tuple(int(x*255) for x in processor.image_mean)) for image in images]
        video = processor.preprocess(images, return_tensors='pt')['pixel_values']
    else:
        images = [Image.fromarray(f.numpy() if isinstance(f, torch.Tensor) else f) for f in video_data]
        video = processor.preprocess(images, return_tensors='pt')['pixel_values']

    return video


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def tokenizer_MMODAL_token(prompt, tokenizer, MMODAL_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split(f'<{MMODAL_INDEX_TOKEN[MMODAL_token_index].lower()}>')]
    num_prompt_chunks = len(prompt.split(f'<{MMODAL_INDEX_TOKEN[MMODAL_token_index].lower()}>'))

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [MMODAL_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]
    
    def call_for_batch(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if (output_ids[0, -keyword_id.shape[0]:] == keyword_id).all():
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False
    
    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        outputs = []
        for i in range(output_ids.shape[0]):
            outputs.append(self.call_for_batch(output_ids[i].unsqueeze(0), scores))
        return all(outputs)
