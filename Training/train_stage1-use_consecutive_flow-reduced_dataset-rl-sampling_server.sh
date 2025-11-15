EXP_NAME="train_stage1"

python train_stage1-use_consecutive_flow-reduced_dataset-rl-sampling_server.py \
 --pretrained_model_name_or_path="/home/minhthan001/Video/video_research/Training/ckpts/stable-video-diffusion-img2vid-xt-1-1" \
 --controlnet_model_name_or_path="/projects_vol/gp_slab/minhthan001/ckpts_mofa/checkpoint-7500/controlnet" 
