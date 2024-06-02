# Adopted from https://github.com/haotian-liu/LLaVA. Below is the original copyright:
# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

import sys
sys.path.append('./')

from videollama2.train import train

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
