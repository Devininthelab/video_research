#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to fine-tune Stable Video Diffusion."""
import argparse
from collections import defaultdict
import logging
import math
import os
import shutil
from pathlib import Path
from urllib.parse import urlparse

import accelerate
import numpy as np
import PIL
from PIL import Image
import torch
import torchvision
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import RandomSampler
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from einops import rearrange

import diffusers
from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler, UNetSpatioTemporalConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available, load_image
from diffusers.utils.import_utils import is_xformers_available

from train_utils.dataset import WebVid10M

from models.unet_spatio_temporal_condition_controlnet import UNetSpatioTemporalConditionControlNetModel
from pipeline.pipeline import FlowControlNetPipeline
from models.svdxt_featureflow_forward_controlnet_s2d_fixcmp_norefine import FlowControlNet

from train_utils.unimatch.unimatch.unimatch import UniMatch
from train_utils.unimatch.utils.flow_viz import flow_to_image

from utils.scheduling_ddim_with_logprob import DDIMSchedulerWithLogProb
from pipeline.pipeline_with_logprob import FlowControlNetPipelineWithLogProb
import lpips


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.24.0.dev0")

logger = get_logger(__name__, log_level="INFO")


class LpipsRewardFunction:
    def __init__(self, device):
        self.device = device
        self.lpips_model = lpips.LPIPS(net='alex').to(device)
        self.lpips_model.eval()
        logger.info("Initialized LPIPS reward function")

    def compute_lpips_reward(self, generated_frames, gt_frames):
        lpips_scores = []
        with torch.no_grad():
            for i in range(len(generated_frames)):
                gen_frame = torch.from_numpy(generated_frames[i]).permute(2, 0, 1).float() / 127.5 - 1.0
                gt_frame = torch.from_numpy(gt_frames[i]).permute(2, 0, 1).float() / 127.5 - 1.0

                gen_frame = gen_frame.unsqueeze(0).to(self.device)
                gt_frame = gt_frame.unsqueeze(0).to(self.device)
            
                lpips_dist = self.lpips_model(gen_frame, gt_frame)
                lpips_scores.append(lpips_dist.item())
            
        mean_lpips = np.mean(lpips_scores)
        reward = np.exp(-mean_lpips * 2.0)
        return reward
    
    def __call__(
        self,
        frames,
        gt_frames=None,
        prompts=None
    ):
        rewards = []
        # pil_frames = [Image.fromarray(frame.astype(np.uint8)) for frame in frames]
        primary_reward = self.compute_lpips_reward(frames, gt_frames)
        total_reward = primary_reward
        return total_reward




def preprocess_size(image1, image2, padding_factor=32):
    '''
        img: [b, c, h, w]
        preprares images for optical flow estimation
        ensures width > height (model requirement)
        resizes to [384, 512] for flow prediction 
        scale flow vectors back to original dimensions
    '''
    original_dtype = image1.dtype
    transpose_img = False
    # the model is trained with size: width > height
    if image1.size(-2) > image1.size(-1):
        image1 = torch.transpose(image1, -2, -1)
        image2 = torch.transpose(image2, -2, -1)
        transpose_img = True

    # inference_size = [int(np.ceil(image1.size(-2) / padding_factor)) * padding_factor,
    #                 int(np.ceil(image1.size(-1) / padding_factor)) * padding_factor]
        
    inference_size = [384, 512]

    assert isinstance(inference_size, list) or isinstance(inference_size, tuple)
    ori_size = image1.shape[-2:]

    # resize before inference
    if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
        image1 = F.interpolate(image1, size=inference_size, mode='bilinear',
                                align_corners=True).to(original_dtype)
        image2 = F.interpolate(image2, size=inference_size, mode='bilinear',
                                align_corners=True).to(original_dtype)
    
    return image1, image2, inference_size, ori_size, transpose_img


def postprocess_size(flow_pr, inference_size, ori_size, transpose_img):

    if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
        flow_pr = F.interpolate(flow_pr, size=ori_size, mode='bilinear',
                                align_corners=True)
        flow_pr[:, 0] = flow_pr[:, 0] * ori_size[-1] / inference_size[-1]
        flow_pr[:, 1] = flow_pr[:, 1] * ori_size[-2] / inference_size[-2]

    if transpose_img:
        flow_pr = torch.transpose(flow_pr, -2, -1)
    
    return flow_pr


@torch.no_grad()
def get_optical_flows(unimatch, video_frame):
    '''
        video_frame: [b, t, c, h, w]
        extract optical flow from consecutive video frames using Unimatch
        return flow from frame 0 to all subsequent frames

        return: flows: [b, t-1, 2, h, w]
    '''

    original_dtype = video_frame.dtype
    video_frame = (video_frame * 255).to(original_dtype)

    # print(video_frame.dtype)

    flows = []
    for i in range(video_frame.shape[1] - 1):
        image1, image2 = video_frame[:, 0], video_frame[:, i + 1]
        # print(image1.dtype)
        image1_r, image2_r, inference_size, ori_size, transpose_img = preprocess_size(image1, image2)
        # print(image1_r.dtype)
        results_dict_r = unimatch(image1_r, image2_r,
            attn_type='swin',
            attn_splits_list=[2, 8],
            corr_radius_list=[-1, 4],
            prop_radius_list=[-1, 1],
            num_reg_refine=6,
            task='flow',
            pred_bidir_flow=False,
            )
        flow_r = results_dict_r['flow_preds'][-1]  # [b, 2, H, W]
        # print(flow_r.shape)
        flow = postprocess_size(flow_r, inference_size, ori_size, transpose_img)
        flows.append(flow.unsqueeze(1))  # [b, 1, 2, h, w]
    
    flows = torch.cat(flows, dim=1).to(torch.float16)  # [b, t, 2, h, w]
    return flows


def create_iterator(sample_size, sample_dataset):
    while True:
        sample_loader = torch.utils.data.DataLoader(
            dataset= sample_dataset,
            batch_size=sample_size,
            drop_last=True
        )

        for item in sample_loader:
            yield item




min_value = 0.002
max_value = 700
image_d = 64
noise_d_low = 32
noise_d_high = 64
sigma_data = 0.5








def tensor_to_vae_latent(t, vae):
    video_length = t.shape[1]

    t = rearrange(t, "b f c h w -> (b f) c h w")
    latents = vae.encode(t).latent_dist.sample()
    latents = rearrange(latents, "(b f) c h w -> b f c h w", f=video_length)
    latents = latents * vae.config.scaling_factor

    return latents


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train Stable Diffusion XL for InstructPix2Pix."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--num_frames",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--height",
        type=int,
        default=320,
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=500,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the text/image prompt"
            " multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--per_gpu_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--conditioning_dropout_prob",
        type=float,
        default=0.1,
        help="Conditioning dropout probability. Drops out the conditionings (image and edit prompt) used in training InstructPix2Pix. See section 3.2.1 in the paper: https://arxiv.org/abs/2211.09800.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--use_ema", action="store_true", help="Whether to use EMA model."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=3,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )

    parser.add_argument(
        "--pretrain_unet",
        type=str,
        default=None,
        help="use weight for unet block",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=128,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--sample_stride",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--num_batches_per_epoch",
        type=int,
        default=4,
        help="Number of batches to sample per epoch.",
    )
    parser.add_argument(
        "--num_inner_epochs",
        type=int,
        default=2,
        help="Number of inner epochs for PPO training.",
    )
    parser.add_argument(
        "--clip_range",
        type=float,
        default=1e-4,
        help="PPO clip range.",
    )
    

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

def _get_add_time_ids(
        fps,
        motion_bucket_ids,  # Expecting a list of tensor floats
        noise_aug_strength,
        dtype,
        batch_size,
        unet=None,
    ):
        """
        Creates temporal conditioning embeddings
        Includes FPS, motion bucket ID, and noise augmentation strength
        """
        # # Ensure motion_bucket_ids is a tensor with the correct shape
        # if not isinstance(motion_bucket_ids, torch.Tensor):
        #     # motion_bucket_ids = torch.tensor(motion_bucket_ids, dtype=dtype)
    
        # # Reshape motion_bucket_ids if necessary
        # if motion_bucket_ids.dim() == 1:
        #     motion_bucket_ids = motion_bucket_ids.view(-1, 1)

        motion_bucket_ids = torch.tensor([motion_bucket_ids], dtype=dtype).unsqueeze(0).repeat(batch_size, 1)
    
        # Check for batch size consistency
        if motion_bucket_ids.size(0) != batch_size:
            raise ValueError("The length of motion_bucket_ids must match the batch_size.")
    
        add_time_ids = [fps, noise_aug_strength]
    
        # Concatenate fps and noise_aug_strength with motion_bucket_ids along the second dimension
        add_time_ids = torch.tensor(add_time_ids, dtype=dtype).repeat(batch_size, 1)
        add_time_ids = torch.cat([add_time_ids, motion_bucket_ids.to(add_time_ids)], dim=1)
    
        # Checking the dimensions of the added time embedding
        passed_add_embed_dim = unet.config.addition_time_embed_dim * add_time_ids.size(1)
        expected_add_embed_dim = unet.add_embedding.linear_1.in_features
    
        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, "
                f"but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. "
                "Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )
    
        return add_time_ids



def main():
    args = parse_args()

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir)
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )

    generator = torch.Generator(
        device=accelerator.device).manual_seed(42)

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load models
    feature_extractor = CLIPImageProcessor.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="feature_extractor", revision=args.revision
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="image_encoder", revision=args.revision, variant="fp16"
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant="fp16")
    unet = UNetSpatioTemporalConditionControlNetModel.from_pretrained(
        args.pretrained_model_name_or_path if args.pretrain_unet is None else args.pretrain_unet,
        subfolder="unet",
        low_cpu_mem_usage=True,
        variant="fp16",
    )
    if args.controlnet_model_name_or_path:
        logger.info("Loading existing controlnet weights")
        controlnet = FlowControlNet.from_pretrained(args.controlnet_model_name_or_path)
    else:
        logger.info("Initializing controlnet weights from unet")
        controlnet = FlowControlNet.from_unet(unet)
        
    # Freeze vae and image_encoder and unet
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.requires_grad_(False) # Turn off controlent for now

    # Define Unimatch
    unimatch = UniMatch(feature_channels=128,
        num_scales=2,
        upsample_factor=4,
        num_head=1,
        ffn_dim_expansion=4,
        num_transformer_layers=6,
        reg_refine=True,
        task='flow').to('cuda')
    checkpoint = torch.load('./train_utils/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth')
    unimatch.load_state_dict(checkpoint['model'])
    unimatch.eval()
    unimatch.requires_grad_(False)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    image_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()

    if args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
        except ImportError:
            raise ImportError("Please install bitsandbytes to use 8-bit Adam.")
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        controlnet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Dataset
    train_dataset = WebVid10M(
        meta_path="/projects_vol/gp_slab/minhthan001/data_webvid_reduce/reduced_clean_results_2M_train.csv",
        data_dir="/projects_vol/gp_slab/minhthan001/data_webvid_reduce/reduced_WebVid",
        sample_stride=args.sample_stride,
        sample_n_frames=args.num_frames, 
        sample_size=[args.height, args.width]
    )

    # Create sample dataloader for PPO
    sample_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.per_gpu_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    # Prepare with accelerator
    unet, optimizer, sample_dataloader, controlnet = accelerator.prepare(
        unet, optimizer, sample_dataloader, controlnet
    )

    # Initialize Pipeline and Scheduler
    scheduler = DDIMSchedulerWithLogProb.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="scheduler"
    )
    
    pipeline = FlowControlNetPipelineWithLogProb(
        unet=unet,
        controlnet=controlnet,
        vae=vae,
        image_encoder=image_encoder,
        feature_extractor=feature_extractor,
        scheduler=scheduler,
    )
    pipeline.set_progress_bar_config(disable=True)

    # Reward Function
    reward_fn = LpipsRewardFunction(accelerator.device)

    # Training Loop
    logger.info("***** Running PPO training *****")
    global_step = 0

    # Create iterator from prepared dataloader
    sample_iterator = iter(sample_dataloader)
    
    for epoch in range(args.num_train_epochs):
        #################### SAMPLING ####################
        # 1. Sampling Phase
        pipeline.unet.eval()
        pipeline.controlnet.eval()
        samples = []
        
        logger.info(f"Epoch {epoch}: Sampling...")
        # turn off grad
        
        for i in tqdm(
                range(args.num_batches_per_epoch),
                desc=f"Epoch {epoch}: sampling",
                disable=not accelerator.is_local_main_process,
                position=0,
            ):
            # Get batch from iterator, restart if exhausted
            try:
                batch = next(sample_iterator)
            except StopIteration:
                sample_iterator = iter(sample_dataloader)
                batch = next(sample_iterator)
            
            pixel_values = batch["pixel_values"].to(accelerator.device, weight_dtype) # [b, t, c, h, w]
            
            # Extract flows
            flows = get_optical_flows(unimatch, pixel_values) # [1, T-1, 2, h, w]
            print("flows shape:", flows.shape) # flows shape: torch.Size([1, 24, 2, 384, 384])
            
            # Prepare inputs for pipeline
            # pixel_values: [1, t, c, h, w]
            # image: first frame [1, c, h, w]
            image = pixel_values[:, 0]
            # print("image shape:", image.shape) # image shape: torch.Size([1, 3, 384, 384])
            # print(type(image)) # <class 'torch.Tensor'>
            pil_images = [Image.fromarray((img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)) for img in image]
            # print("pil_images len:", len(pil_images)) # pil_images len: 1
            
            # controlnet_condition: first frame
            controlnet_condition = pil_images
            
            # controlnet_flow: flows
            controlnet_flow = flows
            
            with torch.no_grad():
                # Generate video
                output = pipeline(
                    image=pil_images,
                    controlnet_condition=controlnet_condition,
                    controlnet_flow=controlnet_flow,
                    height=args.height,
                    width=args.width,
                    num_frames=args.num_frames,
                    num_inference_steps=25, # Fixed for now
                    output_type="pt", # Return tensors
                    return_dict=True,
                    batch_size=len(pil_images)
                )
            
            # print("FOR DEBUG")
            # print("All latents of shape", [lat.shape for lat in output.all_latents]) # torch.Size([1, 25 (num frames), 4, 48, 48]); 26 latents include the first sampling latents from N(0,1)
            # print("All log probs of shape", [logp.shape for logp in output.all_log_probs]) # List for 25 timesteps

            timesteps = pipeline.scheduler.timesteps.repeat(
                1, 1
            )  # (batch_size, num_steps)
            # print("Timesteps shape:", timesteps.shape) # Timesteps shape: torch.Size([1, 25])

            log_probs = torch.stack(output.all_log_probs, dim=1)  # (bs, num_steps) = (1, 25)
            latents = torch.stack(output.all_latents, dim=1)  # (bs, num_steps, C, H, W) = (1, 26, 25, 4, 48, 48)
            # print("Latents shape:", latents.shape)
            # print("Log probs shape:", log_probs.shape)

            
            generated_frames = output.frames[0]
            # # save genearted frames for debug
            os.makedirs("./debug_generated_frames", exist_ok=True)
            for k, frame in enumerate(generated_frames):
                # Move channels to the last dimension (H, W, C)
                # Scale from [0, 1] to [0, 255] and convert to uint8
                frame_np = (frame.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
                
                # Convert to PIL image and save
                img = Image.fromarray(frame_np)
                img.save(f"./debug_generated_frames/debug_generated_frame_{i}_{k}.png")

            generated_frames_np = (
                generated_frames
                .permute(0, 2, 3, 1)   # [T, H, W, C]
                .cpu()
                .numpy()
            )
            generated_frames_np = (generated_frames_np * 255).astype(np.uint8)
            gt_frames = (pixel_values[0].permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
    
        
            # Compute rewards
            reward = reward_fn.compute_lpips_reward(generated_frames_np, gt_frames) # [1,]
            reward = torch.tensor(reward, device=accelerator.device, dtype=weight_dtype)
            
            samples.append({
                "timesteps": timesteps,
                "latents": latents[:, :-1],   # each entry is the latent before timestep t
                "next_latents": latents[:, 1:],  # each entry is the latent after timestep t
                "log_probs": log_probs,
                "rewards": reward,
                # Condition for the model
                "flows": controlnet_flow,
                "controlnet_condition": controlnet_condition,
                "image": image,
            })
        
        # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
        samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}
        # gather rewards across processes
        rewards = accelerator.gather(samples["rewards"]).cpu().numpy()


        if accelerator.is_local_main_process:
            logger.info(
                f"[Epoch {epoch}] "
                f"reward_mean={rewards.mean():.4f}, "
                f"reward_std={rewards.std():.4f}"
            )
        # log rewards
        accelerator.log(
            {
                "reward": rewards,
                "epoch": epoch,
                "reward_mean": rewards.mean(),
                "reward_std": rewards.std(),
            },
            step=global_step,
        )
        
        # Compute advantages
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # ungather advantages; we only need to keep the entries corresponding to the samples on this process
        samples["advantages"] = (
            torch.as_tensor(advantages)
            .reshape(accelerator.num_processes, -1)[accelerator.process_index]
            .to(accelerator.device)
        )
        del samples["rewards"]  # we don't need rewards anymore in the training phase

        total_batch_size, num_timesteps = samples["timesteps"].shape
        assert (
            total_batch_size
            == 1 * args.num_batches_per_epoch # = arg.batch_Size * args.num_batches_per_epoch
        )
        # assert num_timesteps == args.num_inference_steps  == (25)
            
        print("Total batch size:", total_batch_size)
        #################### PPO TRAINING ####################
        logger.info(f"Epoch {epoch}: Training...")
        for inner_epoch in range(args.num_inner_epochs):
            # shuffle samples along batch dimension
            perm = torch.randperm(total_batch_size, device=accelerator.device)
            samples = {k: v[perm] for k, v in samples.items()}

            # shuffle along time dimension independently for each sample
            perms = torch.stack(
                [
                    torch.randperm(num_timesteps, device=accelerator.device)
                    for _ in range(total_batch_size)
                ]
            )

            for key in ["timesteps", "latents", "next_latents", "log_probs"]:
                samples[key] = samples[key][
                    torch.arange(total_batch_size, device=accelerator.device)[:, None],
                    perms,
                ]

            # rebatch for training
            samples_batched = {
                k: v.reshape(-1, args.per_gpu_batch_size, *v.shape[1:])
                for k, v in samples.items()
            }

            # dict of lists -> list of dicts for easier iteration
            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]
            print(samples_batched)

            # # 2. Training Phase
            # pipeline.controlnet.train() # Only training controlnet
            # info = defaultdict(list)
            # for i, sample in tqdm(
            #     list(enumerate(samples_batched)),
            #     desc=f"Epoch {epoch}.{inner_epoch}: training",
            #     position=0,
            #     disable=not accelerator.is_local_main_process,
            # ):  
            #     # Prepare inputs
            #     for j in tqdm(
            #         range(25), # num_inference_steps: HARDCODE here, fix later
            #         desc="Timestep",
            #         position=1,
            #         leave=False,
            #         disable=not accelerator.is_local_main_process,
            #     ):
            #         latents_t = sample["latents"][:, j].detach()     # x_t 
            #         # latents_prev = sample["latents"][:, j-1].detach() # x_t-1 
            #         old_log_prob = sample["log_probs"][:, j].detach()

            #         with accelerator.accumulate(pipeline.controlnet):
            #             latent_model_input = latents_t
            #             down_block_res_samples, mid_block_res_sample, controlnet_flow, _ = controlnet(
            #                 latent_model_input,
            #                 sample["timesteps"][:, j],
            #                 encoder_hidden_states=image_embeddings, # Text/Image embeddings
            #                 controlnet_cond=controlnet_condition,
            #                 added_time_ids=added_time_ids, # Giữ nguyên logic tính time_ids của bạn
            #                 return_dict=False,
            #             )

            #             noise_pred = unet(
            #                 latent_model_input,
            #                 timesteps,
            #                 encoder_hidden_states=image_embeddings,
            #                 down_block_additional_residuals=down_block_res_samples, # Inject ControlNet
            #                 mid_block_additional_residual=mid_block_res_sample,     # Inject ControlNet
            #                 added_time_ids=added_time_ids,
            #                 return_dict=False,
            #             )[0]
                                                                        
                                                                        
                        
                


            
                
            #     # Backprop
            #     accelerator.backward(loss_sum)
            #     optimizer.step()
            #     optimizer.zero_grad()
                
        # Save checkpoint
        if epoch % 1 == 0:
            accelerator.save_state(os.path.join(args.output_dir, f"checkpoint-{epoch}"))

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        controlnet = accelerator.unwrap_model(controlnet)
        pipeline = FlowControlNetPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            image_encoder=accelerator.unwrap_model(image_encoder),
            vae=accelerator.unwrap_model(vae),
            unet=unet,
            controlnet=controlnet,
            revision=args.revision,
        )
        pipeline.save_pretrained(args.output_dir)

    accelerator.end_training()


if __name__ == "__main__":
    main()
