EXP_NAME="train_stage1"

export CUDA_VISIBLE_DEVICES=1

accelerate launch --main_process_port=27657 train_stage1-use_consecutive_flow-reduced_dataset-rl-training_client.py \
 --pretrained_model_name_or_path="./ckpts/stable-video-diffusion-img2vid-xt-1-1" \
 --controlnet_model_name_or_path="/projects_vol/gp_slab/minhthan001/ckpts_mofa/checkpoint-7500/controlnet" \
 --output_dir="logs-use_consecutive_flow-reduced_dataset-rl/${EXP_NAME}/" \
 --width=256 \
 --height=256 \
 --seed=42 \
 --learning_rate=3e-6 \
 --mixed_precision="fp16" \
 --gradient_accumulation_steps=2 \
 --num_frames=25 \
 --sample_stride=4 \
 --sample_num_batches_per_epoch=2 \
 --clip_range=0.05 \
 --kl_penalty=0.5 \
 --target_kl=0.02 \
 --reference_update_freq=5 \
 --num_inner_epochs=2 
#  --resume_from_checkpoint latest
