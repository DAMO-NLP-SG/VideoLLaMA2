import base64
import dataclasses
from io import BytesIO
from enum import auto, Enum
from typing import List, Tuple

from PIL import Image
from .constants import LOGDIR, NUM_FRAMES


class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()
    PLAIN = auto()
    LLAMA2 = auto()
    QWEN = auto()

@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"

    skip_next: bool = False
    modality: str = "image"

    def get_prompt(self):
        messages = self.messages
        modality_token = f"<{self.modality}>"
        if len(messages) > 0 and type(messages[0][1]) is tuple:
            messages = self.messages.copy()
            init_role, init_msg = messages[0].copy()
            init_msg = init_msg[0].replace(modality_token, "").strip()
            if 'mmtag' in self.version:
                messages[0] = (init_role, init_msg)
                messages.insert(0, (self.roles[0], "<Image><image></Image>"))
                messages.insert(1, (self.roles[1], "Received."))
            else:
                messages[0] = (init_role, f"{modality_token}\n" + init_msg)

        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.LLAMA2:
            wrap_sys = lambda msg: f"<<SYS>>\n{msg}\n<</SYS>>\n\n"
            wrap_inst = lambda msg: f"[INST] {msg} [/INST]"
            ret = ""

            for i, (role, message) in enumerate(messages):
                if i == 0:
                    assert message, "first message should not be none"
                    assert role == self.roles[0], "first message should come from user"
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    if i == 0: message = wrap_sys(self.system) + message
                    if i % 2 == 0:
                        message = wrap_inst(message)
                        ret += self.sep + message
                    else:
                        ret += " " + message + " " + self.sep2
                else:
                    ret += ""
            ret = ret.lstrip(self.sep)
        elif self.sep_style == SeparatorStyle.QWEN:
            ret = ""
            # 1. Add system prompt
            ret += self.system + self.sep + "\n"
            # 2. Iterate message
            for i, (role, message) in enumerate(messages):
                if i == 0:
                    assert message, "first message should not be none"
                    assert role == self.roles[0], "first message should come from user"
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    # 2.1 Add role and message
                    ret += role + message + self.sep + "\n"
                else:
                    # 2.2 Add generation prompt
                    ret += role
        elif self.sep_style == SeparatorStyle.PLAIN:
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + seps[i % 2]
                else:
                    ret += role
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

        return ret

    def append_message(self, role, message):
        self.messages.append([role, message])

    def process_image(self, image, image_process_mode, return_pil=False, image_format='PNG', max_len=800, min_len=400):
        if image_process_mode == "Pad":
            def expand2square(pil_img, background_color=(122, 116, 104)):
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
            image = expand2square(image)
        elif image_process_mode in ["Default", "Crop"]:
            pass
        elif image_process_mode == "Resize":
            image = image.resize((336, 336))
        else:
            raise ValueError(f"Invalid image_process_mode: {image_process_mode}")
        if max(image.size) > max_len:
            max_hw, min_hw = max(image.size), min(image.size)
            aspect_ratio = max_hw / min_hw
            shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
            longest_edge = int(shortest_edge * aspect_ratio)
            W, H = image.size
            if H > W:
                H, W = longest_edge, shortest_edge
            else:
                H, W = shortest_edge, longest_edge
            image = image.resize((W, H))
        if return_pil:
            return image
        else:
            buffered = BytesIO()
            image.save(buffered, format=image_format)
            img_b64_str = base64.b64encode(buffered.getvalue()).decode()
            return img_b64_str


    def get_videos(self, return_pil=False):
        video_frames = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    from decord import VideoReader, cpu
                    import numpy as np
                    # here video is the file path of input video
                    msg, video, image_process_mode = msg
                    if not return_pil:
                        # return filepath
                        video_frames.append(video)
                    else:
                        # read video using decord.VideoReader
                        decord_vr = VideoReader(uri=video, ctx=cpu(0))
                        duration = len(decord_vr)
                        frame_id_list = np.linspace(0, duration-1, NUM_FRAMES, dtype=int)
                        # convert the extracted image frames into PIL objects
                        all_images = [Image.fromarray(f) for f in decord_vr.get_batch(frame_id_list).asnumpy()]
                        video_frames.extend([self.process_image(image, image_process_mode, return_pil=return_pil) for image in all_images])
        return video_frames


    def get_images(self, return_pil=False):
        images = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    msg, image, image_process_mode = msg
                    image = self.process_image(image, image_process_mode, return_pil=return_pil)
                    images.append(image)

                    # import base64
                    # from io import BytesIO
                    # from PIL import Image
                    # # here image is a PIL object
                    # msg, image, image_process_mode = msg
                    # if image_process_mode == "Pad":
                    #     def expand2square(pil_img, background_color=(122, 116, 104)):
                    #         width, height = pil_img.size
                    #         if width == height:
                    #             return pil_img
                    #         elif width > height:
                    #             result = Image.new(pil_img.mode, (width, width), background_color)
                    #             result.paste(pil_img, (0, (width - height) // 2))
                    #             return result
                    #         else:
                    #             result = Image.new(pil_img.mode, (height, height), background_color)
                    #             result.paste(pil_img, ((height - width) // 2, 0))
                    #             return result
                    #     image = expand2square(image)
                    # elif image_process_mode in ["Default", "Crop"]:
                    #     pass
                    # elif image_process_mode == "Resize":
                    #     image = image.resize((336, 336))
                    # else:
                    #     raise ValueError(f"Invalid image_process_mode: {image_process_mode}")
                    # max_hw, min_hw = max(image.size), min(image.size)
                    # aspect_ratio = max_hw / min_hw
                    # max_len, min_len = 800, 400
                    # shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
                    # longest_edge = int(shortest_edge * aspect_ratio)
                    # W, H = image.size
                    # if longest_edge != max(image.size):
                    #     if H > W:
                    #         H, W = longest_edge, shortest_edge
                    #     else:
                    #         H, W = shortest_edge, longest_edge
                    #     image = image.resize((W, H))
                    # if return_pil:
                    #     images.append(image)
                    # else:
                    #     buffered = BytesIO()
                    #     image.save(buffered, format="PNG")
                    #     img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                    #     images.append(img_b64_str)
        return images

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    # import base64
                    # from io import BytesIO
                    # from PIL import Image
                    # msg, image, image_process_mode = msg
                    # max_hw, min_hw = max(image.size), min(image.size)
                    # aspect_ratio = max_hw / min_hw
                    # max_len, min_len = 800, 400
                    # shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
                    # longest_edge = int(shortest_edge * aspect_ratio)
                    # W, H = image.size
                    # if H > W:
                    #     H, W = longest_edge, shortest_edge
                    # else:
                    #     H, W = shortest_edge, longest_edge
                    # image = image.resize((W, H))
                    # buffered = BytesIO()
                    # image.save(buffered, format="JPEG")
                    # img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                    # img_str = f'<img src="data:image/png;base64,{img_b64_str}" alt="user upload image" />'
                    # display image/video in the textbox
                    msg, image_or_video, image_process_mode = msg
                    ##print("imagebox:", image)
                    if isinstance(image_or_video, Image.Image):
                        # image is PIL object
                        img_b64_str = self.process_image(image_or_video, "Default", return_pil=False, image_format='JPEG')
                        img_str = f'<img src="data:image/jpeg;base64,{img_b64_str}" alt="user upload image" />'
                        msg = img_str + msg.replace('<image>', '').strip()
                    else:
                        # video is file path
                        vid_str = f'<video controls playsinline width="500" style="display: inline-block;" src="./file={image_or_video}"></video><br>'
                        msg = vid_str + msg.replace('<video>', '').strip()
                    ret.append([msg, None])
                else:
                    ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version)

    def dict(self):
        if (self.modality == "image" and len(self.get_images()) > 0) or \
            (self.modality == "video" and len(self.get_videos()) > 0):
            return {
                "system": self.system,
                "roles": self.roles,
                "messages": [[x, y[0] if type(y) is tuple else y] for x, y in self.messages],
                "offset": self.offset,
                "sep": self.sep,
                "sep2": self.sep2,
                "modality": self.modality
            }
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }


conv_vicuna_v0 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(
        ("Human", "What are the key differences between renewable and non-renewable energy sources?"),
        ("Assistant",
            "Renewable energy sources are those that can be replenished naturally in a relatively "
            "short amount of time, such as solar, wind, hydro, geothermal, and biomass. "
            "Non-renewable energy sources, on the other hand, are finite and will eventually be "
            "depleted, such as coal, oil, and natural gas. Here are some key differences between "
            "renewable and non-renewable energy sources:\n"
            "1. Availability: Renewable energy sources are virtually inexhaustible, while non-renewable "
            "energy sources are finite and will eventually run out.\n"
            "2. Environmental impact: Renewable energy sources have a much lower environmental impact "
            "than non-renewable sources, which can lead to air and water pollution, greenhouse gas emissions, "
            "and other negative effects.\n"
            "3. Cost: Renewable energy sources can be more expensive to initially set up, but they typically "
            "have lower operational costs than non-renewable sources.\n"
            "4. Reliability: Renewable energy sources are often more reliable and can be used in more remote "
            "locations than non-renewable sources.\n"
            "5. Flexibility: Renewable energy sources are often more flexible and can be adapted to different "
            "situations and needs, while non-renewable sources are more rigid and inflexible.\n"
            "6. Sustainability: Renewable energy sources are more sustainable over the long term, while "
            "non-renewable sources are not, and their depletion can lead to economic and social instability.\n")
    ),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_llava_plain = Conversation(
    system="",
    roles=("", ""),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.PLAIN,
    sep="",
    sep2="\n"
)

conv_llava_v0_mmtag = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant is able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
           "The visual content will be provided with the following format: <Image>visual content</Image>.",
    roles=("Human", "Assistant"),
    messages=(
    ),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
    version="v0_mmtag",
)

conv_llava_v0 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(
    ),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_vicuna_v1 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_llava_v1_mmtag = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant is able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
           "The visual content will be provided with the following format: <Image>visual content</Image>.",
    roles=("USER", "ASSISTANT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
    version="v1_mmtag",
)

conv_llava_v1 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_llava_llama2 = Conversation(
    system="You are a helpful language and vision assistant. "
           "You are able to understand the visual content that the user provides, "
           "and assist the user with a variety of tasks using natural language.",
    roles=("USER", "ASSISTANT"),
    version="llama2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA2,
    sep="<s>",
    sep2="</s>",
)

conv_llama2 = Conversation(
    system="""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.""",
    roles=("USER", "ASSISTANT"),
    version="llama2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA2,
    sep="<s>",
    sep2="</s>",
)

conv_mistral = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="llama2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA2,
    sep="",
    sep2="</s>",
)

conv_qwen = Conversation(
    system="<|im_start|>system\nYou are a helpful assistant.",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.QWEN,
    sep="<|im_end|>",
    version="qwen",
)

conv_qwen_plain = Conversation(
    system="",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.PLAIN,
    sep="<|im_end|>",
    sep2="<|im_end|>",
    version="qwen_plain",
)

default_conversation = conv_mistral
conv_templates = {
    "default": conv_vicuna_v0,
    # pretrain template
    "plain": conv_llava_plain,
    # llava v0
    "v0": conv_vicuna_v0,
    "v0_plain": conv_llava_plain,
    "v0_mmtag": conv_llava_v0_mmtag,
    "llava_v0": conv_llava_v0,
    # llava v1
    "v1": conv_vicuna_v1,
    "v1_mmtag": conv_llava_v1_mmtag,
    "llava_v1": conv_llava_v1,
    "vicuna_v1": conv_vicuna_v1,
    # llava v1.5
    "llava_llama2": conv_llava_llama2,
    # llama2
    "llama2": conv_llama2,
    # mistral
    "mistral": conv_mistral,
    # qwen
    "qwen": conv_qwen,
    "qwen_plain": conv_qwen_plain,
}


if __name__ == "__main__":
    print(default_conversation.get_prompt())
