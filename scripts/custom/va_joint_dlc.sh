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
export WANDB_PROJECT=audio_visual_stage3_qwen2
RUN_NAME=audio_visual_stage3_qwen2
DATA_DIR=datasets
OUTP_DIR=work_dirs
torchrun --nnodes $WORLD_SIZE \
    --nproc_per_node $NPROC_PER_NODE  \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --node_rank $RANK \
    videollama2/train.py \
    --deepspeed scripts/zero2.json \
    --model_type videollama2_qwen2 \
    --model_path DAMO-NLP-SG/VideoLLaMA2.1-7B-16F \
    --data_folder ${DATA_DIR} \
    --data_path ${DATA_DIR}/stage3_video_audio.json,${DATA_DIR}/stage2_audio_subset_new.json,${DATA_DIR}/stage2_video_subset.json \
    --vision_tower google/siglip-so400m-patch14-384 \
    --audio_tower $OUTP_DIR/audio_tower.bin \
    --pretrain_mm_mlp_adapter_a $OUTP_DIR/mm_projector_a.bin \
    --mm_projector_type stc_connector_v35 \
    --mm_projector_a_type mlp2x_gelu \
    --va True \
    --tune_audio_tower True \
    --tune_adapter_llm True \
    --tune_mm_mlp_adapter_a True \
    --mm_vision_select_layer -2 \
    --image_aspect_ratio pad \
    --num_frames 16 \
    --bf16 True \
    --tf32 True \
    --fp16 False \
    --output_dir $OUTP_DIR/${WANDB_PROJECT}/VideoLLaMA2.1-7B-AV \
    --num_train_epochs 2 \
    --per_device_train_batch_size $LOCAL_BATCH_SIZE \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
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
    --report_to tensorboard \
    --run_name $RUN_NAME \
