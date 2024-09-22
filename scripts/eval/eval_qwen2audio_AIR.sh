set -x

EVAL_DATA_DIR=eval
OUTPUT_DIR=eval_output
# CKPT=DAMO-NLP-SG/VideoLLaMA2-7B
CKPT=/mnt/data/xyf/Qwen2-Audio-7B-Instruct
CKPT_NAME=$(echo $CKPT | rev | cut -d'/' -f1 | rev)

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

# divide data via the number of GPUs per task
GPUS_PER_TASK=1
CHUNKS=$((${#GPULIST[@]}/$GPUS_PER_TASK))

output_file=${OUTPUT_DIR}/AIR/answers/${CKPT_NAME}/merge.json

if [ ! -f "$output_file" ]; then
    for IDX in $(seq 0 $((CHUNKS-1))); do
        # select the GPUs for the task
        gpu_devices=$(IFS=,; echo "${GPULIST[*]:$(($IDX*$GPUS_PER_TASK)):$GPUS_PER_TASK}")
        TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${gpu_devices} python3 videollama2/eval/inference_qwen2audio_AIR.py \
            --model-path ${CKPT} \
            --dataset AIR \
            --video-folder /mnt/data/xyf/AIR-Bench-Dataset/Chat/ \
            --question-file /mnt/data/xyf/AIR-Bench-Dataset/airbench_level_3_eval.jsonl \
            --answer-file /mnt/data/xyf/AIR-Bench-Dataset/Chat/Chat_meta.json \
            --output-file ${OUTPUT_DIR}/AIR/answers/${CKPT_NAME}/${CHUNKS}_${IDX}.json \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX
    done

    wait

    # Clear out the output file if it exists.
    > "$output_file"

    #Loop through the indices and concatenate each file.
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat ${OUTPUT_DIR}/AIR/answers/${CKPT_NAME}/${CHUNKS}_${IDX}.json >> "$output_file"
    done
fi

python videollama2/eval/eval_audio_AIR.py \
    --pred-path /mnt/workspace/fisher/VideoLLaMA2_backup/eval_output/AIR/answers/Qwen2-Audio-7B-Instruct/merge.json \
    --answer-file /mnt/data/xyf/AIR-Bench-Dataset/Chat/Chat_meta.json \
    --api-key f68a11a54a064caa851e290258d52cce \
    --api-endpoint https://vl-australiaeast.openai.azure.com \
    --api-deployname gpt35-turbo-0613
