EVAL_DATA_DIR=/mnt/base/chengzesen/dataset/videollm_eval
OUTPUT_DIR=eval/
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

output_file=${OUTPUT_DIR}/MSVD_Zero_Shot_QA/answers/${CKPT_NAME}/merge.json

if [ ! -f "$output_file" ]; then
    for IDX in $(seq 0 $((CHUNKS-1))); do
        # select the GPUs for the task
        gpu_devices=$(IFS=,; echo "${GPULIST[*]:$(($IDX*$GPUS_PER_TASK)):$GPUS_PER_TASK}")
        TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${gpu_devices} python3 videollama2/eval/run_inference_video_qa_gpt.py \
            --model-path ${CKPT} \
            --video-folder ${EVAL_DATA_DIR}/MSVD_Zero_Shot_QA/videos \
            --question-file ${EVAL_DATA_DIR}/MSVD_Zero_Shot_QA/test_q.json \
            --answer-file ${EVAL_DATA_DIR}/MSVD_Zero_Shot_QA/test_a.json \
            --output-file ${OUTPUT_DIR}/MSVD_Zero_Shot_QA/answers/${CKPT_NAME}/${CHUNKS}_${IDX}.json \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX \
            --conv-mode llama_2 &
    done

    wait

    # Clear out the output file if it exists.
    > "$output_file"

    #Loop through the indices and concatenate each file.
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat ${OUTPUT_DIR}/MSVD_Zero_Shot_QA/answers/${CKPT_NAME}/${CHUNKS}_${IDX}.json >> "$output_file"
    done
fi


AZURE_API_KEY="35632dae7dd94d0a93338db373c63893"
AZURE_API_ENDPOINT=https://damo-openai-gpt4v-test.openai.azure.com
AZURE_API_DEPLOYNAME=gpt-35-turbo

python3 videollama2/eval/eval_video_qa_gpt.py \
    --pred-path ${output_file} \
    --output-dir ${OUTPUT_DIR}/MSVD_Zero_Shot_QA/answers/${CKPT_NAME}/gpt \
    --output-json ${OUTPUT_DIR}/MSVD_Zero_Shot_QA/answers/${CKPT_NAME}/results.json \
    --api-key $AZURE_API_KEY \
    --api-endpoint $AZURE_API_ENDPOINT \
    --api-deployname $AZURE_API_DEPLOYNAME \
    --num-tasks 4
