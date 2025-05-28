#! /bin/bash
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

LANG=en    # language direction (e.g 'en' for VSR task / 'en-es' for En to Es VST task)
USE_BLEU=false
DATA_PATH=/home/liuzehua/task/VSR/VSP-LLM/dataset/CNVSRC_Single_icl
SPLIT=test
# set paths
ROOT=/home/liuzehua/task/VSR/VSP-LLM
MODEL_SRC=${ROOT}/exp/model_v4
VISUAL_DECODER_PATH=/home/liuzehua/task/VSR/VSP-LLM/checkpoints/visual_decoder/Qwen2.5-32B-Instruct

MODEL_PATH=/home/liuzehua/task/VSR/VSP-LLM/result/Single-90h/Interspeech_2025/Qwen2.5-32B-Instruct-before_context-30s/17-10-17/model-v4/model/checkpoint_best.pt # path to trained model
# 使用 basename 命令获取文件名
FILENAME=$(basename $MODEL_PATH)
# 使用 dirname 命令获取目录名
DIRNAME=$(dirname $MODEL_PATH)

for VARI in  6 4 8
do
    OUT_PATH=${DIRNAME}/${FILENAME}-${SPLIT}-infer-lm_${lm_weight}-repetition_penalty-${VARI}   # output path to save
    mkdir -p ${OUT_PATH}

    # start decoding
    export PYTHONPATH="${ROOT}/fairseq:$PYTHONPATH"
    CUDA_VISIBLE_DEVICES=5 python -B ${MODEL_SRC}/infer.py \
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
            override.eval_bleu=${USE_BLEU} \
            override.visual_decoder_path=${VISUAL_DECODER_PATH} \
            common_eval.path=${MODEL_PATH} \
            common_eval.results_path=${OUT_PATH} \
            save.infer_result=${OUT_PATH}/infer_result.json \
            generation.lm_weight=${lm_weight} \
            override.repetition_penalty=${VARI} 

done
# OUT_PATH=${DIRNAME}/${FILENAME}-${SPLIT}-infer   # output path to save
# mkdir -p ${OUT_PATH}


# # start decoding
# export PYTHONPATH="${ROOT}/fairseq:$PYTHONPATH"
# CUDA_VISIBLE_DEVICES=2 python -B ${MODEL_SRC}/infer.py \
#     --config-dir ${MODEL_SRC}/conf \
#     --config-name infer \
#         hydra.run.dir=${OUT_PATH} \
#         common.user_dir=${MODEL_SRC} \
#         dataset.gen_subset=${SPLIT} \
#         override.data=${DATA_PATH} \
#         override.label_dir=${DATA_PATH} \
#         generation.beam=20 \
#         generation.lenpen=0 \
#         dataset.max_tokens=3000 \
#         override.eval_bleu=${USE_BLEU} \
#         override.visual_decoder_path=${VISUAL_DECODER_PATH} \
#         common_eval.path=${MODEL_PATH} \
#         common_eval.results_path=${OUT_PATH} \
#         save.infer_result=${OUT_PATH}/infer_result.json \
#         override.repetition_penalty=
#         # 2>&1 | tee ${OUT_PATH}/infer_output.log