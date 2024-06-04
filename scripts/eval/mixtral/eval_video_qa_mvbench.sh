set -x

EVAL_DATA_DIR=/mnt/base/chengzesen/dataset/videollm_eval
OUTPUT_DIR=eval
CKPT_NAME=videollama2-mixtral8x7b-ep3_1200st
# CKPT_NAME=videollama2-16f-ep3
CKPT=publish_models/${CKPT_NAME}
# CKPT_NAME=VideoLLaMA2
# CKPT=DAMO-NLP-SG/${CKPT_NAME}

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

# divide data via the number of GPUs per task
GPUS_PER_TASK=2
CHUNKS=$((${#GPULIST[@]}/$GPUS_PER_TASK))

output_file=${OUTPUT_DIR}/mvbench/answers/${CKPT_NAME}_merge.json

for IDX in $(seq 0 $((CHUNKS-1))); do
    gpu_devices=$(IFS=,; echo "${GPULIST[*]:$(($IDX*$GPUS_PER_TASK)):$GPUS_PER_TASK}")
    TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${gpu_devices} python3 videollama2/eval/run_inference_video_qa_batch.py \
        --dataset mvbench \
        --model-path ${CKPT} \
        --video-folder ${EVAL_DATA_DIR}/mvbench/video \
        --question-file ${EVAL_DATA_DIR}/mvbench/json \
        --answer-file ${OUTPUT_DIR}/mvbench/answers/${CKPT_NAME}/${CHUNKS}_${IDX}.json \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --batch-size 1 \
        --conv-mode llama_2 &
done

wait

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${OUTPUT_DIR}/mvbench/answers/${CKPT_NAME}/${CHUNKS}_${IDX}.json >> "$output_file"
done

python3 videollama2/eval/eval_video_qa_mvbench.py \
    --pred_path ${output_file} \
