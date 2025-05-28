DEBUG_MODE=true


# 设置CUDA_VISIBLE_DEVICES为一个字符串变量，可以包含多个GPU编号
CUDA_DEVICES="1"
# 计算逗号分隔的数字数量，即GPU数量
IFS=',' read -ra ADDR <<< "$CUDA_DEVICES"
DEVICE_COUNT=${#ADDR[@]}


current_date=$(date +"%Y-%m-%d")
current_time=$(date +"%H-%M-%S")


# 根据DEBUG_MODE选择输出路径
if [ "$DEBUG_MODE" = false ]; then
    SAVE_PATH=/home/liuzehua/task/VSR/VSP-LLM/main_log/${current_date}/${current_time}/model-v7_1
else
    SAVE_PATH=/home/liuzehua/task/VSR/VSP-LLM/main_log/temp
fi


mkdir -p ${SAVE_PATH}
export DEBUG_MODE
# export PYTHONUNBUFFERED=1
# GPU 数量 --nproc_per_node  总的机器数量--nnodes
export HYDRA_FULL_ERROR=0
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES torchrun --master_port=29506 --nproc_per_node=$DEVICE_COUNT --nnodes=1 --node_rank=0 main.py \
        input.gpu_nums=$DEVICE_COUNT  save.save_path=${SAVE_PATH} \
        --config-path /home/liuzehua/task/VSR/VSP-LLM/exp/model_v7_1/conf \
        --config-name vsp-llm-90h-single \
        2>&1 | tee ${SAVE_PATH}/train_output.log





