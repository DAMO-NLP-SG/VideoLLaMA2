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

output_file=${OUTPUT_DIR}/videomme/answers/${CKPT_NAME}/merge.json

# judge if the number of json lines is 0
if [ ! -f "$output_file" ] || [ $(cat "$output_file" | wc -l) -eq 0 ]; then
    rm -f ${OUTPUT_DIR}/videomme/answers/${CKPT_NAME}/*.json
fi

if [ ! -f "$output_file" ]; then
    for IDX in $(seq 0 $((CHUNKS-1))); do
        # select the GPUs for the task
        gpu_devices=$(IFS=,; echo "${GPULIST[*]:$(($IDX*$GPUS_PER_TASK)):$GPUS_PER_TASK}")
        TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${gpu_devices} python3 videollama2/new_eval/inference_video_mcqa_videomme.py \
            --model-path ${CKPT} \
            --video-folder ${EVAL_DATA_DIR}/videomme/videos \
            --question-file ${EVAL_DATA_DIR}/videomme/Video-MME.json \
            --answer-file ${OUTPUT_DIR}/videomme/answers/${CKPT_NAME}/${CHUNKS}_${IDX}.json \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX &
    done

    wait

    # Clear out the output file if it exists.
    > "$output_file"

    echo "[" >> "$output_file"

    #Loop through the indices and concatenate each file.
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat ${OUTPUT_DIR}/videomme/answers/${CKPT_NAME}/${CHUNKS}_${IDX}.json >> "$output_file"
    done

    sed -i '$s/.$//' $output_file

    echo "]" >> "$output_file"
fi


python videollama2/new_eval/eval_video_mcqa_videomme.py \
    --results_file $output_file \
    --video_duration_type "short,medium,long" \
    --return_categories_accuracy \
    --return_sub_categories_accuracy \
    --return_task_types_accuracy \
    --skip_missing \
