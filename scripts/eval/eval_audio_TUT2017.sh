set -x

EVAL_DATA_DIR=eval
OUTPUT_DIR=eval_output
# CKPT=DAMO-NLP-SG/VideoLLaMA2-7B
CKPT=/mnt/data/xyf/va2/output/videollama2_audio_visual_stage3_a_v_va_256_16f
#videollama2_audio_visual_stage3_a_v_va_512
#videollama2_audio_visual_stage3_allmodality_256
#videollama2_audio_stage2_mlp_mistral_videollm_ep2_64_updated
#vlb_audio_stage2_mlp_mistral_videollm_ep2_sft_128
#vlb_audio_stage2_mlp_mistral_videollm_ep2_sft_64
#vlb_audio_stage2_mlp_mistral_videollm_ep2_64
#vlb_audio_stage2_mlp_encoder_beats_qwen2_new_videollm_ep2_64
CKPT_NAME=$(echo $CKPT | rev | cut -d'/' -f1 | rev)

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

# divide data via the number of GPUs per task
GPUS_PER_TASK=1
CHUNKS=$((${#GPULIST[@]}/$GPUS_PER_TASK))

output_file=${OUTPUT_DIR}/TUT2017/answers/${CKPT_NAME}/merge.json

if [ ! -f "$output_file" ]; then
    for IDX in $(seq 0 $((CHUNKS-1))); do
        # select the GPUs for the task
        gpu_devices=$(IFS=,; echo "${GPULIST[*]:$(($IDX*$GPUS_PER_TASK)):$GPUS_PER_TASK}")
        TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${gpu_devices} python3 videollama2/eval/inference_audio.py \
            --model-path ${CKPT} \
            --dataset TUT2017 \
            --video-folder /mnt/data/xyf/TUT2017 \
            --question-file /mnt/data/xyf/TUT2017/tut2017_eval.jsonl \
            --answer-file /mnt/data/xyf/TUT2017/tut2017_eval.jsonl \
            --output-file ${OUTPUT_DIR}/TUT2017/answers/${CKPT_NAME}/${CHUNKS}_${IDX}.json \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX &
    done

    wait

    # Clear out the output file if it exists.
    > "$output_file"

    #Loop through the indices and concatenate each file.
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat ${OUTPUT_DIR}/TUT2017/answers/${CKPT_NAME}/${CHUNKS}_${IDX}.json >> "$output_file"
    done
fi

python videollama2/eval/eval_audio_TUT2017.py \
    --pred-path ${output_file}
