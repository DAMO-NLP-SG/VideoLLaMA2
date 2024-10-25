set -x

EVAL_DATA_DIR=eval
OUTPUT_DIR=eval_output
CKPT=DAMO-NLP-SG/VideoLLaMA2.1-7B-AV
CKPT_NAME=$(echo $CKPT | rev | cut -d'/' -f1 | rev)

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

# divide data via the number of GPUs per task
GPUS_PER_TASK=1
CHUNKS=$((${#GPULIST[@]}/$GPUS_PER_TASK))

output_file=${OUTPUT_DIR}/AVSD/answers/${CKPT_NAME}/merge.json

if [ ! -f "$output_file" ]; then
    for IDX in $(seq 0 $((CHUNKS-1))); do
        # select the GPUs for the task
        gpu_devices=$(IFS=,; echo "${GPULIST[*]:$(($IDX*$GPUS_PER_TASK)):$GPUS_PER_TASK}")
        TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${gpu_devices} python3 videollama2/eval/inference_audio_video.py \
            --model-path ${CKPT} \
            --dataset AVSD \
            --video-folder ${EVAL_DATA_DIR}/AVSD/Charades_v1_480 \
            --question-file ${EVAL_DATA_DIR}/AVSD/instruction_val.json \
            --answer-file ${EVAL_DATA_DIR}/AVSD/instruction_val.json \
            --output-file ${OUTPUT_DIR}/AVSD/answers/${CKPT_NAME}/${CHUNKS}_${IDX}.json \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX
    done

    wait

    # Clear out the output file if it exists.
    > "$output_file"

    #Loop through the indices and concatenate each file.
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat ${OUTPUT_DIR}/AVSD/answers/${CKPT_NAME}/${CHUNKS}_${IDX}.json >> "$output_file"
    done
fi

python videollama2/eval/eval_audio_video_AVSD.py \
    --pred-path /mnt/data/xyf/VideoLLaMA2_backup/eval_output/AVSD/answers/vlb_audio_visual_stage3_tuning_projector_beats_qwen2_videollm_ep2/merge.json \
    --api-key $AZURE_API_KEY \
    --api-endpoint $AZURE_API_ENDPOINT \
    --api-deployname $AZURE_API_DEPLOYNAME
