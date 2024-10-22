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

output_file=${OUTPUT_DIR}/msvc/answers/${CKPT_NAME}/merge.json

# judge if the number of json lines is 0
if [ ! -f "$output_file" ] || [ $(cat "$output_file" | wc -l) -eq 0 ]; then
    rm -f ${OUTPUT_DIR}/msvc/answers/${CKPT_NAME}/*.json
fi

if [ ! -f "$output_file" ]; then
    for IDX in $(seq 0 $((CHUNKS-1))); do
        # select the GPUs for the task
        gpu_devices=$(IFS=,; echo "${GPULIST[*]:$(($IDX*$GPUS_PER_TASK)):$GPUS_PER_TASK}")
        TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${gpu_devices} python3 videollama2/eval/inference_video_cap_msvc.py \
          --model-path ${CKPT} \
          --video-folder ${EVAL_DATA_DIR}/msvc \
          --question-file ${EVAL_DATA_DIR}/msvc/msvc.json \
          --output-file ${OUTPUT_DIR}/msvc/answers/${CKPT_NAME}/${CHUNKS}_${IDX}.json \
          --num-chunks $CHUNKS \
          --chunk-idx $IDX &
    done

    wait

    # Clear out the output file if it exists.
    > "$output_file"

    #Loop through the indices and concatenate each file.
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat ${OUTPUT_DIR}/msvc/answers/${CKPT_NAME}/${CHUNKS}_${IDX}.json >> "$output_file"
    done
fi


AZURE_API_KEY=your_key
AZURE_API_ENDPOINT=your_endpoint
AZURE_API_DEPLOYNAME=your_deployname

python3 videollama2/eval/eval_video_cap_msvc_correctness.py \
    --pred-path $output_file \
    --output-dir ${OUTPUT_DIR}/msvc/answers/${CKPT_NAME}/correctness_gpt \
    --output-json ${OUTPUT_DIR}/msvc/answers/${CKPT_NAME}/correctness_results.json \
    --api-key $AZURE_API_KEY \
    --api-endpoint $AZURE_API_ENDPOINT \
    --api-deployname $AZURE_API_DEPLOYNAME \
    --num-tasks 4 \

python3 videollama2/eval/eval_video_cap_msvc_detailedness.py \
    --pred-path $output_file \
    --output-dir ${OUTPUT_DIR}/msvc/answers/${CKPT_NAME}/detailedness_gpt \
    --output-json ${OUTPUT_DIR}/msvc/answers/${CKPT_NAME}/detailedness_results.json \
    --api-key $AZURE_API_KEY \
    --api-endpoint $AZURE_API_ENDPOINT \
    --api-deployname $AZURE_API_DEPLOYNAME \
    --num-tasks 4 \
