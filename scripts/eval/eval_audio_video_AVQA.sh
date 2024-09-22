set -x

EVAL_DATA_DIR=eval
OUTPUT_DIR=eval_output
# CKPT=DAMO-NLP-SG/VideoLLaMA2-7B
CKPT=/mnt/data/xyf/va2/output/videollama2_audio_visual_stage3_a_v_va_256_16f
#videollama2_audio_visual_stage3_a_v_va_512
#videollama2_audio_visual_stage3_allmodality_128
#videollama2_audio_visual_stage3_allmodality_256
#videollama2_audio_visual_stage3_tuning_projector_beats_mistral_avinstruct
#videollama2_audio_visual_stage3_allmodality_256
#vlb_audio_visual_stage3_tuning_projector_beats_mistral_videollm_ep2_64_new
#vlb_audio_visual_stage3_tuning_projector_beats_mistral_videollm_ep2_64
#vlb_audio_visual_stage3_tuning_projector_beats_qwen2_videollm_ep2_bs128
#vlb_audio_visual_stage3_tuning_projector_beats_qwen2_videollm_ep2
CKPT_NAME=$(echo $CKPT | rev | cut -d'/' -f1 | rev)

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

# divide data via the number of GPUs per task
GPUS_PER_TASK=1
CHUNKS=$((${#GPULIST[@]}/$GPUS_PER_TASK))

output_file=${OUTPUT_DIR}/AVQA/answers/${CKPT_NAME}/merge.json

if [ ! -f "$output_file" ]; then
    for IDX in $(seq 0 $((CHUNKS-1))); do
        # select the GPUs for the task
        gpu_devices=$(IFS=,; echo "${GPULIST[*]:$(($IDX*$GPUS_PER_TASK)):$GPUS_PER_TASK}")
        TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${gpu_devices} python3 videollama2/eval/inference_audio_video.py \
            --model-path ${CKPT} \
            --dataset AVQA \
            --video-folder /mnt/data/xyf/AVQA_music/MUSIC-AVQA-videos \
            --question-file /mnt/data/xyf/AVQA_music/AVQA_music_test.json \
            --answer-file /mnt/data/xyf/AVQA_music/AVQA_music_test.json \
            --output-file ${OUTPUT_DIR}/AVQA/answers/${CKPT_NAME}/${CHUNKS}_${IDX}.json \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX
    done

    wait

    # Clear out the output file if it exists.
    > "$output_file"

    #Loop through the indices and concatenate each file.
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat ${OUTPUT_DIR}/AVQA/answers/${CKPT_NAME}/${CHUNKS}_${IDX}.json >> "$output_file"
    done
fi

python3 videollama2/eval/eval_audio_video_AVQA.py \
    --pred-path ${output_file}
