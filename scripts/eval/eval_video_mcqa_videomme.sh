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

output_file=${OUTPUT_DIR}/videomme/answers/${CKPT_NAME}/merge.json
output_sub_file=${OUTPUT_DIR}/videomme/answers/${CKPT_NAME}/merge_sub.json

# judge if the number of json lines is 0
if [ ! -f "$output_file" ] || [ $(cat "$output_file" | wc -l) -eq 0 ]; then
    rm -f ${OUTPUT_DIR}/videomme/answers/${CKPT_NAME}/*.json
fi


if [ ! -f "$output_file" ]; then
    for IDX in $(seq 0 $((CHUNKS-1))); do
        # select the GPUs for the task
        gpu_devices=$(IFS=,; echo "${GPULIST[*]:$(($IDX*$GPUS_PER_TASK)):$GPUS_PER_TASK}")
        TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${gpu_devices} python3 videollama2/eval/inference_video_mcqa_videomme.py \
            --model-path ${CKPT} \
            --video-folder ${EVAL_DATA_DIR}/videomme/videos \
            --subtitle-folder ${EVAL_DATA_DIR}/videomme/subtitles \
            --question-file ${EVAL_DATA_DIR}/videomme/test-00000-of-00001.parquet \
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

    # Clear out the output file if it exists.
    > "$output_sub_file"

    echo "[" >> "$output_sub_file"

    #Loop through the indices and concatenate each file.
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat ${OUTPUT_DIR}/videomme/answers/${CKPT_NAME}/${CHUNKS}_${IDX}_sub.json >> "$output_sub_file"
    done

    sed -i '$s/.$//' $output_sub_file

    echo "]" >> "$output_sub_file"
fi


python videollama2/eval/eval_video_mcqa_videomme.py \
    --results_file $output_file \
    --video_duration_type "short,medium,long" \
    --return_categories_accuracy \
    --return_sub_categories_accuracy \
    --return_task_types_accuracy \
    --skip_missing \

python videollama2/eval/eval_video_mcqa_videomme.py \
    --results_file $output_sub_file \
    --video_duration_type "short,medium,long" \
    --return_categories_accuracy \
    --return_sub_categories_accuracy \
    --return_task_types_accuracy \
    --skip_missing \
