EXP_NAME="train_stage1"

accelerate launch --main_process_port=27657 train_stage1-use_consecutive_flow-reduced_dataset-rl-training_client.py \
 --pretrained_model_name_or_path="./ckpts/stable-video-diffusion-img2vid-xt-1-1" \
 --controlnet_model_name_or_path="/mnt/ssd6/thong/MOFA-Video/Training/logs-use_consecutive_flow-reduced_dataset/train_stage1/checkpoint-7500/controlnet" \
 --output_dir="logs-use_consecutive_flow-reduced_dataset-rl/${EXP_NAME}/" \
 --width=384 \
 --height=384 \
 --seed=42 \
 --learning_rate=2e-5 \
 --mixed_precision="fp16" \
 --gradient_accumulation_steps=8 \
 --num_frames=25 \
 --sample_stride=4 
#  --resume_from_checkpoint latest
