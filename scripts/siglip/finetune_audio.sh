#!/bin/bash

# Environment Variables
ARG_WORLD_SIZE=${1:-1}
ARG_NPROC_PER_NODE=${2:-8}
ARG_MASTER_ADDR="127.0.0.1"
ARG_MASTER_PORT=16666
ARG_RANK=0

# Multiple conditions
if [ ! -n "$WORLD_SIZE" ] || [ ! -n "$NPROC_PER_NODE" ]; then
    WORLD_SIZE=$ARG_WORLD_SIZE
    NPROC_PER_NODE=$ARG_NPROC_PER_NODE
fi
if [ ! -n "$MASTER_ADDR" ] || [ ! -n "$MASTER_PORT" ] || [ ! -n "$RANK" ]; then
    MASTER_ADDR=$ARG_MASTER_ADDR
    MASTER_PORT=$ARG_MASTER_PORT
    RANK=$ARG_RANK
fi

echo "WORLD_SIZE: $WORLD_SIZE"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"

# Training Arguments
GLOBAL_BATCH_SIZE=128
LOCAL_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=$[$GLOBAL_BATCH_SIZE/($WORLD_SIZE*$NPROC_PER_NODE*$LOCAL_BATCH_SIZE)]

# Log Arguments
export TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=videollama2qwen2_vllava
RUN_NAME=videollama2qwen2_vllava
DATA_DIR=datasets
OUTP_DIR=work_dirs
export NPROC_PER_NODE=1
torchrun --nnodes $WORLD_SIZE \
    --nproc_per_node $NPROC_PER_NODE  \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --node_rank $RANK \
    videollama2/train_flash_attn.py \
    --deepspeed scripts/zero2.json \
    --model_type videollama2_qwen2 \
    --model_path /mnt/data/xyf/tmpfs_models/videollama2qwen2_finetune_7b_siglip_16f \
    --data_path_a /mnt/data/xyf/stage2_filter_vgg.json \
    --audio_tower /mnt/data/xyf/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt \
    --pretrain_mm_mlp_adapter_a /mnt/data/xyf/va2/output/stage1_mlp_videollm_qwen2_new_beats_1024/mm_projector_a.bin \
    --mm_projector_a_type mlp2x_gelu \
    --tune_mm_mlp_adapter_a True \
    --tune_audio_tower True \
    --tune_adapter_llm False \
    --bf16 True \
    --tf32 True \
    --fp16 False \
    --output_dir /mnt/data/xyf/va2/output/sft_mlp \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --freeze_backbone True \
    --report_to wandb \