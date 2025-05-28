#!/bin/bash
SPLIT=valid
# Path settings
ROOT=../..
DATA_PATH=${ROOT}/dataset
MODEL_SRC=${ROOT}/exp/Context_aware_exp
VISUAL_DECODER_PATH=${ROOT}/checkpoints/visual_decoder/Qwen2.5-32B-Instruct

# Trained model checkpoint
MODEL_PATH=

# Extract filename and directory for organization
FILENAME=$(basename $MODEL_PATH)
DIRNAME=$(dirname $MODEL_PATH)

# Loop over different repetition_penalty values
for VARI in 6 
do
    OUT_PATH=${DIRNAME}/${FILENAME}-${SPLIT}-infer-repetition_penalty-${VARI}
    mkdir -p ${OUT_PATH}

    echo "Running inference with repetition_penalty=${VARI}..."

    # Inference command
    export PYTHONPATH="${ROOT}/fairseq:$PYTHONPATH"
    CUDA_VISIBLE_DEVICES=0 python -B ${MODEL_SRC}/infer.py \
        --config-dir ${MODEL_SRC}/conf \
        --config-name infer \
        hydra.run.dir=${OUT_PATH} \
        common.user_dir=${MODEL_SRC} \
        dataset.gen_subset=${SPLIT} \
        override.data=${DATA_PATH} \
        override.label_dir=${DATA_PATH} \
        generation.beam=5 \
        generation.lenpen=0 \
        dataset.max_tokens=3000 \
        override.visual_decoder_path=${VISUAL_DECODER_PATH} \
        common_eval.path=${MODEL_PATH} \
        common_eval.results_path=${OUT_PATH} \
        save.infer_result=${OUT_PATH}/infer_result.json \
        generation.lm_weight=0 \
        override.repetition_penalty=${VARI}
done
