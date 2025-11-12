EXP_NAME="train_stage1"

python train_stage1-use_consecutive_flow-reduced_dataset-rl-sampling_server.py \
 --pretrained_model_name_or_path="./ckpts/stable-video-diffusion-img2vid-xt-1-1" \
 --controlnet_model_name_or_path="/mnt/ssd6/thong/MOFA-Video/Training/logs-use_consecutive_flow-reduced_dataset/train_stage1/checkpoint-7500/controlnet" 
