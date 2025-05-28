#! /bin/bash
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

LANG=en    # language direction (e.g 'en' for VSR task / 'en-es' for En to Es VST task)
USE_BLEU=false
DATA_PATH=/home/liuzehua/task/VSR/VSP-LLM/dataset/CNVSRC_Single
SPLIT=valid300
# set paths
ROOT=/home/liuzehua/task/VSR/VSP-LLM
MODEL_SRC=${ROOT}/exp/model_v7_1
VISUAL_DECODER_PATH=/home/liuzehua/task/VSR/VSP-LLM/checkpoints/visual_decoder/Qwen2.5-7B

MODEL_PATH=/home/liuzehua/task/VSR/VSP-LLM/main_log/2024-11-26/19-08-22/model-v7_1/model/checkpoint_best.pt # path to trained model
# 使用 basename 命令获取文件名
FILENAME=$(basename $MODEL_PATH)
# 使用 dirname 命令获取目录名
DIRNAME=$(dirname $MODEL_PATH)


ITER=4
# 遍历不同的 repetition_penalty 参数值
for VARI in 10 8 6 4 15 2
do
    OUT_PATH=${DIRNAME}/${FILENAME}-${SPLIT}-infer-lm_0-repetition_penalty-${VARI}-iter-${ITER} # output path to save
    mkdir -p ${OUT_PATH}

    # start decoding
    export PYTHONPATH="${ROOT}/fairseq:$PYTHONPATH"
    CUDA_VISIBLE_DEVICES=1 python -B ${MODEL_SRC}/infer.py \
        --config-dir ${MODEL_SRC}/conf \
        --config-name infer \
            hydra.run.dir=${OUT_PATH} \
            common.user_dir=${MODEL_SRC} \
            dataset.gen_subset=${SPLIT} \
            override.data=${DATA_PATH} \
            override.label_dir=${DATA_PATH} \
            generation.beam=10 \
            generation.lenpen=0 \
            dataset.max_tokens=3000 \
            override.eval_bleu=${USE_BLEU} \
            override.visual_decoder_path=${VISUAL_DECODER_PATH} \
            common_eval.path=${MODEL_PATH} \
            common_eval.results_path=${OUT_PATH} \
            save.infer_result=${OUT_PATH}/infer_result \
            generation.lm_weight=0 \
            override.repetition_penalty=${VARI} \
            override.iter_time=${ITER}
done

