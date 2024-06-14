set -x

EVAL_DATA_DIR=/mnt/chengzs/dataset/videollm_eval
OUTPUT_DIR=eval
# CKPT_NAME=videollama2-mixtral8x7b-ep3_1200st
# CKPT_NAME=videollama2-16f-ep3
# CKPT=publish_models/${CKPT_NAME}
CKPT_NAME=VideoLLaMA2
CKPT=ClownRat/${CKPT_NAME}
# CKPT=DAMO-NLP-SG/${CKPT_NAME}

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

# divide data via the number of GPUs per task
GPUS_PER_TASK=1
CHUNKS=$((${#GPULIST[@]}/$GPUS_PER_TASK))

output_file=${OUTPUT_DIR}/videochatgpt_gen/answers/correctness/${CKPT_NAME}/merge.json

if [ ! -f "$output_file" ]; then
    for IDX in $(seq 0 $((CHUNKS-1))); do
        # select the GPUs for the task
        gpu_devices=$(IFS=,; echo "${GPULIST[*]:$(($IDX*$GPUS_PER_TASK)):$GPUS_PER_TASK}")
        TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3 videollama2/new_eval/inference_video_oqa_vcgpt_general.py \
            --model-path ${CKPT} \
            --video-folder ${EVAL_DATA_DIR}/videochatgpt_gen/Test_Videos \
            --question-file ${EVAL_DATA_DIR}/videochatgpt_gen/generic_qa.json \
            --answer-file ${OUTPUT_DIR}/videochatgpt_gen/answers/correctness/${CKPT_NAME}/${CHUNKS}_${IDX}.json \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX &
    done

    wait

    # Clear out the output file if it exists.
    > "$output_file"

    #Loop through the indices and concatenate each file.
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat ${OUTPUT_DIR}/videochatgpt_gen/answers/correctness/${CKPT_NAME}/${CHUNKS}_${IDX}.json >> "$output_file"
    done

    mkdir -p ${OUTPUT_DIR}/videochatgpt_gen/answers/detail/${CKPT_NAME}
    mkdir -p ${OUTPUT_DIR}/videochatgpt_gen/answers/context/${CKPT_NAME}
    cp ${output_file} ${OUTPUT_DIR}/videochatgpt_gen/answers/detail/${CKPT_NAME}/merge.json
    cp ${output_file} ${OUTPUT_DIR}/videochatgpt_gen/answers/context/${CKPT_NAME}/merge.json
fi


AZURE_API_KEY="35632dae7dd94d0a93338db373c63893"
AZURE_API_ENDPOINT=https://damo-openai-gpt4v-test.openai.azure.com
AZURE_API_DEPLOYNAME=gpt-35-turbo

python3 videollama2/new_eval/eval_video_oqa_vcgpt_1_correctness.py \
    --pred-path ${output_file} \
    --output-dir ${OUTPUT_DIR}/videochatgpt_gen/answers/correctness/${CKPT_NAME}/gpt \
    --output-json ${OUTPUT_DIR}/videochatgpt_gen/answers/correctness/${CKPT_NAME}/results.json \
    --api-key "35632dae7dd94d0a93338db373c63893" \
    --api-endpoint https://damo-openai-gpt4v-test.openai.azure.com \
    --api-deployname gpt-35-turbo \
    --num-tasks 4
