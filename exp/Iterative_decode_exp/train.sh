DEBUG_MODE=false

# Set the GPUs to be used for training
CUDA_DEVICES="0"
# Automatically calculate the number of GPUs
IFS=',' read -ra ADDR <<< "$CUDA_DEVICES"
DEVICE_COUNT=${#ADDR[@]}

# Generate a save path based on the current date and time
current_date=$(date +"%Y-%m-%d")
current_time=$(date +"%H-%M-%S")
SAVE_PATH=main_log/${current_date}/${current_time}/Iterative_decode
mkdir -p ${SAVE_PATH}

visual_decoder_path=../../checkpoints/visual_decoder/    # please download qwen model here(https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e)
visual_encoder_path=../../checkpoints/visual_encoder/    # please download the pretrained vsr encoder here(https://www.modelscope.cn/models/chenchen2121/CNVSRC2024Baseline/files  model_name: model_avg_cncvs_cnvsrc-single.pth)
export DEBUG_MODE
export NCCL_P2P_DISABLE=1

# if input.icl_data=False ,then no context information will be trained, if input.icl_data=True ,the training process will be complemented with context information
export HYDRA_FULL_ERROR=0
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES torchrun --master_port=29503 --nproc_per_node=$DEVICE_COUNT --nnodes=1 --node_rank=0 main.py \
        input.gpu_nums=$DEVICE_COUNT  save.save_path=${SAVE_PATH} input.icl_data=True  input.label_dir=../../dataset\
        usr_dir=../.. input.visual_decoder_path= ${visual_decoder_path}  input.visual_encoder_path=${visual_encoder_path}\
        --config-path ./conf \
        --config-name vsr-llm-90h-single \
        2>&1 | tee ${SAVE_PATH}/train_output.log




