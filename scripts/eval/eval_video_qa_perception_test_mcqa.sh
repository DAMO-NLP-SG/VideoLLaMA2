EVAL_DATA_DIR=eval/
OUTPUT_DIR=eval/
CKPT_NAME=VideoLLaMA2
CKPT=DAMO-NLP-SG/${CKPT_NAME}

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

output_file=${OUTPUT_DIR}/perception_test_mcqa/answers/${CKPT_NAME}_merge.json

for IDX in $(seq 0 $((CHUNKS-1))); do
  TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3 videollama2/eval/run_inference_video_qa_perception_test_mcqa.py \
      --model-path ${CKPT} \
      --video-folder ${EVAL_DATA_DIR}/perception_test_mcqa/videos \
      --question-file ${EVAL_DATA_DIR}/perception_test_mcqa/mc_question_test.json \
      --answer-file ${OUTPUT_DIR}/perception_test_mcqa/answers/${CKPT_NAME}_${CHUNKS}_${IDX}.json \
      --num-chunks $CHUNKS \
      --chunk-idx $IDX \
      --batch-size 1 \
      --conv-mode llama_2 &
done

wait

# Clear out the output file if it exists.
> "$output_file"

echo "{" >> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${OUTPUT_DIR}/perception_test_mcqa/answers/${CKPT_NAME}_${CHUNKS}_${IDX}.json >> "$output_file"
done

sed -i '$s/.$//' $output_file

echo "}" >> "$output_file"
