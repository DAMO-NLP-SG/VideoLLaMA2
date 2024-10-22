set -x

EVAL_DATA_DIR=eval
OUTPUT_DIR=eval_output
CKPT=DAMO-NLP-SG/VideoLLaMA2.1-7B-16F
CKPT_NAME=$(echo $CKPT | rev | cut -d'/' -f1 | rev)

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

# divide data via the number of GPUs per task
GPUS_PER_TASK=1
CHUNKS=$((${#GPULIST[@]}/$GPUS_PER_TASK))

output_file=${OUTPUT_DIR}/videochatgpt_gen/answers/detail/${CKPT_NAME}/merge.json

if [ ! -f "$output_file" ]; then
    for IDX in $(seq 0 $((CHUNKS-1))); do
        # select the GPUs for the task
        gpu_devices=$(IFS=,; echo "${GPULIST[*]:$(($IDX*$GPUS_PER_TASK)):$GPUS_PER_TASK}")
        TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${gpu_devices} python3 videollama2/eval/run_inference_video_qa_gpt_general.py \
            --model-path ${CKPT} \
            --video-folder ${EVAL_DATA_DIR}/videochatgpt_gen/Test_Videos \
            --question-file ${EVAL_DATA_DIR}/videochatgpt_gen/generic_qa.json \
            --answer-file ${OUTPUT_DIR}/videochatgpt_gen/answers/detail/${CKPT_NAME}/${CHUNKS}_${IDX}.json \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX &
    done

    wait

    # Clear out the output file if it exists.
    > "$output_file"

    #Loop through the indices and concatenate each file.
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat ${OUTPUT_DIR}/videochatgpt_gen/answers/detail/${CKPT_NAME}/${CHUNKS}_${IDX}.json >> "$output_file"
    done

    mkdir -p ${OUTPUT_DIR}/videochatgpt_gen/answers/correctness/${CKPT_NAME}
    mkdir -p ${OUTPUT_DIR}/videochatgpt_gen/answers/context/${CKPT_NAME}
    cp ${output_file} ${OUTPUT_DIR}/videochatgpt_gen/answers/correctness/${CKPT_NAME}/merge.json
    cp ${output_file} ${OUTPUT_DIR}/videochatgpt_gen/answers/context/${CKPT_NAME}/merge.json
fi


AZURE_API_KEY=your_key
AZURE_API_ENDPOINT=your_endpoint
AZURE_API_DEPLOYNAME=your_deployname

python3 videollama2/eval/eval_video_oqa_vcgpt_2_detailed_orientation.py \
    --pred-path ${output_file} \
    --output-dir ${OUTPUT_DIR}/videochatgpt_gen/answers/detail/${CKPT_NAME}/gpt \
    --output-json ${OUTPUT_DIR}/videochatgpt_gen/answers/detail/${CKPT_NAME}/results.json \
    --api-key $AZURE_API_KEY \
    --api-endpoint $AZURE_API_ENDPOINT \
    --api-deployname $AZURE_API_DEPLOYNAME \
    --num-tasks 4
