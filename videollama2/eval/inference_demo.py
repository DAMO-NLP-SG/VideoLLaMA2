import sys
sys.path.append('./')
from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init
import argparse

def inference(args):

    model_path = args.model_path
    model, processor, tokenizer = model_init(model_path)

    if args.modal_type == "a":
        model.model.vision_tower = None
    elif args.modal_type == "v":
        model.model.audio_tower = None
    elif args.modal_type == "av":
        pass
    else:
        raise NotImplementedError

    audio_video_path = "/mnt/data/xyf/00000368.mp4"
    #/mnt/data/xyf/WBS4I.mp4
    #"/mnt/data/xyf/xyf/Y--ZHUMfueO0.flac"
    #"/mnt/data/xyf/AVQA/AVQA/scratch/shared/beegfs/hchen/train_data/VGGSound_final/video/"
    preprocess = processor['audio' if args.modal_type == "a" else "video"]
    if args.modal_type == "a":
        audio_video_tensor = preprocess(audio_video_path)
    else:
        audio_video_tensor = preprocess(audio_video_path, va=True if args.modal_type == "av" else False)
    question = f"Please describe the video with visual and audio information."
    #Please describe the video with visual and audio information.
    #Please describe the video.
    #Please describe the audio.
    output = mm_infer(
        audio_video_tensor,
        question,
        model=model,
        tokenizer=tokenizer,
        modal='audio' if args.modal_type == "a" else "video",
        do_sample=False,
    )

    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', help='', required=True)
    parser.add_argument('--modal-type', choices=["a", "v", "av"], help='', required=True)
    args = parser.parse_args()

    inference(args)


