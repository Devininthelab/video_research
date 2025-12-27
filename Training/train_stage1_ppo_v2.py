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
from diffusers import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel
from utils.scheduling_ddim_with_logprob import DDIMSchedulerWithLogProb
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

from reward.lpips_flow import LpipsRewardFunction
from pipeline.pipeline_with_logprob import FlowControlNetPipelineWithLogProb
from collections import defaultdict
import gc


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.24.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def clear_memory():
    """Clear GPU memory cache"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()



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
def get_optical_flows(unimatch, video_frame, device='cuda'):
    '''
        video_frame: [b, t, c, h, w]
        extract optical flow from consecutive video frames using Unimatch
        return flow from frame 0 to all subsequent frames

        return: flows: [b, t-1, 2, h, w]
    '''
    # Move unimatch to GPU only when needed
    unimatch = unimatch.to(device)
    
    original_dtype = video_frame.dtype
    video_frame = (video_frame * 255).to(original_dtype)

    flows = []
    for i in range(video_frame.shape[1] - 1):
        image1, image2 = video_frame[:, 0], video_frame[:, i + 1]
        image1_r, image2_r, inference_size, ori_size, transpose_img = preprocess_size(image1, image2)
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
        flow = postprocess_size(flow_r, inference_size, ori_size, transpose_img)
        flows.append(flow.unsqueeze(1))  # [b, 1, 2, h, w]
        
        # Clear intermediate tensors
        del image1_r, image2_r, results_dict_r, flow_r
    
    flows = torch.cat(flows, dim=1).to(torch.float16)  # [b, t, 2, h, w]
    
    # Move unimatch back to CPU to free GPU memory
    unimatch = unimatch.to('cpu')
    clear_memory()
    
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


# WILL NOT NEED FOR PPO LOGIC
# copy from https://github.com/crowsonkb/k-diffusion.git
def stratified_uniform(shape, group=0, groups=1, dtype=None, device=None):
    """Draws stratified samples from a uniform distribution."""
    if groups <= 0:
        raise ValueError(f"groups must be positive, got {groups}")
    if group < 0 or group >= groups:
        raise ValueError(f"group must be in [0, {groups})")
    n = shape[-1] * groups
    offsets = torch.arange(group, n, groups, dtype=dtype, device=device)
    u = torch.rand(shape, dtype=dtype, device=device)
    return (offsets + u) / n

# WILL NOT NEED FOR PPO LOGIC
def rand_cosine_interpolated(shape, image_d, noise_d_low, noise_d_high, sigma_data=1., min_value=1e-3, max_value=1e3, device='cpu', dtype=torch.float32):
    """Draws samples from an interpolated cosine timestep distribution (from simple diffusion)."""

    def logsnr_schedule_cosine(t, logsnr_min, logsnr_max):
        t_min = math.atan(math.exp(-0.5 * logsnr_max))
        t_max = math.atan(math.exp(-0.5 * logsnr_min))
        return -2 * torch.log(torch.tan(t_min + t * (t_max - t_min)))

    def logsnr_schedule_cosine_shifted(t, image_d, noise_d, logsnr_min, logsnr_max):
        shift = 2 * math.log(noise_d / image_d)
        return logsnr_schedule_cosine(t, logsnr_min - shift, logsnr_max - shift) + shift

    def logsnr_schedule_cosine_interpolated(t, image_d, noise_d_low, noise_d_high, logsnr_min, logsnr_max):
        logsnr_low = logsnr_schedule_cosine_shifted(
            t, image_d, noise_d_low, logsnr_min, logsnr_max)
        logsnr_high = logsnr_schedule_cosine_shifted(
            t, image_d, noise_d_high, logsnr_min, logsnr_max)
        return torch.lerp(logsnr_low, logsnr_high, t)

    logsnr_min = -2 * math.log(min_value / sigma_data)
    logsnr_max = -2 * math.log(max_value / sigma_data)
    u = stratified_uniform(
        shape, group=0, groups=1, dtype=dtype, device=device
    )
    logsnr = logsnr_schedule_cosine_interpolated(
        u, image_d, noise_d_low, noise_d_high, logsnr_min, logsnr_max)
    return torch.exp(-logsnr / 2) * sigma_data


min_value = 0.002
max_value = 700
image_d = 64
noise_d_low = 32
noise_d_high = 64
sigma_data = 0.5


def _resize_with_antialiasing(input, size, interpolation="bicubic", align_corners=True):
    h, w = input.shape[-2:]
    factors = (h / size[0], w / size[1])

    # First, we have to determine sigma
    # Taken from skimage: https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/transform/_warps.py#L171
    sigmas = (
        max((factors[0] - 1.0) / 2.0, 0.001),
        max((factors[1] - 1.0) / 2.0, 0.001),
    )

    # Now kernel size. Good results are for 3 sigma, but that is kind of slow. Pillow uses 1 sigma
    # https://github.com/python-pillow/Pillow/blob/master/src/libImaging/Resample.c#L206
    # But they do it in the 2 passes, which gives better results. Let's try 2 sigmas for now
    ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))

    # Make sure it is odd
    if (ks[0] % 2) == 0:
        ks = ks[0] + 1, ks[1]

    if (ks[1] % 2) == 0:
        ks = ks[0], ks[1] + 1

    input = _gaussian_blur2d(input, ks, sigmas)

    output = torch.nn.functional.interpolate(
        input, size=size, mode=interpolation, align_corners=align_corners)
    return output


def _compute_padding(kernel_size):
    """Compute padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]

    # for even kernels we need to do asymmetric padding :(
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]

        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front

        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding


def _filter2d(input, kernel):
    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...].to(
        device=input.device, dtype=input.dtype)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    height, width = tmp_kernel.shape[-2:]

    padding_shape: list[int] = _compute_padding([height, width])
    input = torch.nn.functional.pad(input, padding_shape, mode="reflect")

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    # convolve the tensor with the kernel.
    output = torch.nn.functional.conv2d(
        input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

    out = output.view(b, c, h, w)
    return out


def _gaussian(window_size: int, sigma):
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]])

    batch_size = sigma.shape[0]

    x = (torch.arange(window_size, device=sigma.device,
         dtype=sigma.dtype) - window_size // 2).expand(batch_size, -1)

    if window_size % 2 == 0:
        x = x + 0.5

    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))

    return gauss / gauss.sum(-1, keepdim=True)


def _gaussian_blur2d(input, kernel_size, sigma):
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], dtype=input.dtype)
    else:
        sigma = sigma.to(dtype=input.dtype)

    ky, kx = int(kernel_size[0]), int(kernel_size[1])
    bs = sigma.shape[0]
    kernel_x = _gaussian(kx, sigma[:, 1].view(bs, 1))
    kernel_y = _gaussian(ky, sigma[:, 0].view(bs, 1))
    out_x = _filter2d(input, kernel_x[..., None, :])
    out = _filter2d(out_x, kernel_y[..., None])

    return out


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
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
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
        help="Number of samples to generate for compute advantages.",
    )
    parser.add_argument(
        "--num_inner_epochs",
        type=int,
        default=1,
        help="Number of inner epochs used for update for PPO training.",
    )
    parser.add_argument(
        "--clip_range",
        type=float,
        default=1e-4,
        help="PPO clip range.",
    )
    parser.add_argument(
        "--adv_clip_max",
        type=float,
        default=5.0,
        help="Advantage clipping max value.",
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=1,
        help="Frequency of saving model checkpoints (in epochs).",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def main():
    args = parse_args()

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir)
    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
     #   log_with=args.report_to,
        project_config=accelerator_project_config,
        # kwargs_handlers=[ddp_kwargs]
    )

    generator = torch.Generator(
        device=accelerator.device).manual_seed(23123134)

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # Make one log on every process with the configuration for debugging.
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

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDIMSchedulerWithLogProb.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler")
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
        assert False, "must specify controlnet model path"
        
    # Freeze vae and image_encoder
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.requires_grad_(False)

    # Define Unimatch for optical flow prediction - keep on CPU by default to save GPU memory
    unimatch = UniMatch(feature_channels=128,
        num_scales=2,
        upsample_factor=4,
        num_head=1,
        ffn_dim_expansion=4,
        num_transformer_layers=6,
        reg_refine=True,
        task='flow').to('cpu') # Keep on CPU, move to GPU only when needed
    checkpoint = torch.load('./train_utils/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth', map_location='cpu')
    unimatch.load_state_dict(checkpoint['model'])
    unimatch.eval()
    unimatch.requires_grad_(False)
    del checkpoint  # Clear checkpoint from memory
    clear_memory()


    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move image_encoder and vae to gpu and cast to weight_dtype
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    #controlnet.to(accelerator.device, dtype=weight_dtype)
    # Create EMA for the unet.
    if args.use_ema:
        ema_controlnet = EMAModel(unet.parameters(
        ), model_cls=UNetSpatioTemporalConditionModel, model_config=unet.config)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly")
    
    # Enable gradient checkpointing for unet to save memory
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if args.use_ema:
                ema_controlnet.save_pretrained(os.path.join(output_dir, "controlnet_ema"))

            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "controlnet"))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(
                    input_dir, "unet_ema"), UNetSpatioTemporalConditionModel)
                ema_controlnet.load_state_dict(load_model.state_dict())
                ema_controlnet.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = FlowControlNet.from_pretrained(
                    input_dir, subfolder="controlnet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()
        

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps *
            args.per_gpu_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    controlnet.requires_grad_(True)
    parameters_list = []

    optimizer = optimizer_cls(
        controlnet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # check para
    if accelerator.is_main_process:
        rec_txt1 = open('rec_para.txt', 'w')
        rec_txt2 = open('rec_para_train.txt', 'w')
        for name, para in controlnet.named_parameters():
            if para.requires_grad is False:
                rec_txt1.write(f'{name}\n')
            else:
                rec_txt2.write(f'{name}\n')
        rec_txt1.close()
        rec_txt2.close()
    # DataLoaders creation:
    args.global_batch_size = args.per_gpu_batch_size * accelerator.num_processes

    train_dataset = WebVid10M(
        meta_path='/projects_vol/gp_slab/minhthan001/data_webvid_reduce/reduced_clean_results_2M_train.csv',
        data_dir="/projects_vol/gp_slab/minhthan001/data_webvid_reduce/reduced_WebVid",
        sample_stride=args.sample_stride,
        sample_n_frames=args.num_frames, 
        sample_size=[args.height, args.width]
        )
    sampler = RandomSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.per_gpu_batch_size,
        num_workers=args.num_workers,
    )
    # Data Flow:
    # Load video frames from WebVid10M dataset
    # Encode first frame with CLIP → conditioning
    # Encode all frames to VAE latents
    # Extract optical flows between frames (frame 0 → all others)
    # Add noise to latents (diffusion forward process)
    # ControlNet processes flows and injects features into UNet
    # UNet denoises with ControlNet guidance
    # Compute loss against clean latents

    # LEAVE IT FOR NOW
    # test_dataset = WebVid10M(
    #     meta_path='/apdcephfs/share_1290939/0_public_datasets/WebVid/metadata/metadata_2048_val.csv',
    #     sample_size=[args.height, args.width],
    #     sample_n_frames=args.num_frames, 
    #     sample_stride=args.sample_stride
    #     )
    # test_loader = create_iterator(1, test_dataset)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, lr_scheduler, train_dataloader, controlnet = accelerator.prepare(
        unet, optimizer, lr_scheduler, train_dataloader, controlnet
    )

    if args.use_ema:
        ema_controlnet.to(accelerator.device)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(
        args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("SVDXtend", config=vars(args))
    
    # Initialize reward function
    reward_fn = LpipsRewardFunction(accelerator.device)

    # create pipeline for sampling
    pipeline = FlowControlNetPipelineWithLogProb(
        unet=unet,
        controlnet=controlnet,
        vae=vae,
        image_encoder=image_encoder,
        feature_extractor=feature_extractor,
        scheduler=noise_scheduler,
    )
    pipeline.set_progress_bar_config(disable=True)

    # Train!
    total_batch_size = args.per_gpu_batch_size * \
        accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_gpu_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps),
                        disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        ########################### SAMPLING #############################
        pipeline.controlnet.eval()
        samples = []
        logger.info(f"Epoch {epoch}: Sampling...")
        
        # Clear memory before sampling
        clear_memory()
        
        # CONSIDER LATER AS NEED TO WRAP INTO ACCELERATOR
        # Create iterator from dataloader
        train_dataloader_iter = iter(train_dataloader)
        
        for i in tqdm(range(args.num_batches_per_epoch), desc=f"Epoch {epoch}: sampling", disable=not accelerator.is_local_main_process, position=0):
            batch = next(train_dataloader_iter)
            pixel_values = batch["pixel_values"].to(accelerator.device, weight_dtype)
            flows = get_optical_flows(unimatch, pixel_values, device=accelerator.device) # [1, T-1, 2, h, w]
            print("flows shape:", flows.shape) # flows shape: torch.Size([1, 24, 2, 384, 384])
            
            # Prepare inputs for pipeline
            # pixel_values: [1, t, c, h, w]
            # image: first frame [1, c, h, w]
            # Use the same tensor for both to avoid duplication
            controlnet_condition = pixel_values[:, 0]  # [1, c, h, w]
            
            # Convert to PIL only when needed by pipeline
            pil_images = [Image.fromarray((img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)) for img in controlnet_condition]
            
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
            # print("All latents of shape", [lat.shape for lat in output.all_latents]) # torch.Size([1, 26, (num frames), 4, 48, 48]); 26 latents include the first sampling latents from N(0,1)
            # print("All log probs of shape", [logp.shape for logp in output.all_log_probs]) # List for 25 timesteps

            timesteps = pipeline.scheduler.timesteps.repeat(
                1, 1
            )  # (batch_size, num_steps)
            # print("Timesteps shape:", timesteps.shape) # Timesteps shape: torch.Size([1, 25])

            log_probs = torch.stack(output.all_log_probs, dim=1)  # (bs, num_steps) = (1, 25)
            latents = torch.stack(output.all_latents, dim=1)  # (bs, num_steps, C, H, W) = (1, 26, 25, 4, 48, 48)
            image_latents = output.image_latents
            controlnet_condition = output.controlnet_condition
            controlnet_flow = output.controlnet_flow
            added_time_ids = output.added_time_ids
            image_embeddings = output.image_embeddings
            guidance_scale = output.guidance_scale 
            # print("Controlnet flow shape:", controlnet_flow.shape) # Controlnet flow shape: torch.Size([1, 24, 2, 384, 384])
            # print("Controlnet condition shape:", controlnet_condition.shape) # Controlnet condition shape:
            print("Image latents shape:", image_latents.shape) # Image latents shape: torch.Size([1, 4, 48, 48])
            # print("Added time ids shape:", added_time_ids.shape) # Added time ids shape: torch.Size([1, 3])
            # print("Latents shape:", latents.shape)
            # print("Log probs shape:", log_probs.shape)
            # print("Image embeddings shape:", image_embeddings.shape)
            # print("Guidance scale shape:", guidance_scale.shape) # torch.Size([1, 25, 1, 1, 1])
            # print("Guidance scale value:", guidance_scale) # guidance scale for each frame is different, increase linearly 

            
            generated_frames = output.frames[0].cpu()  # Move to CPU immediately
            # # save genearted frames for debug
            os.makedirs("./debug_generated_frames", exist_ok=True)
            for k, frame in enumerate(generated_frames):
                # Move channels to the last dimension (H, W, C)
                # Scale from [0, 1] to [0, 255] and convert to uint8
                frame_np = (frame.permute(1, 2, 0).numpy() * 255).astype("uint8")
                
                # Convert to PIL image and save
                img = Image.fromarray(frame_np)
                img.save(f"./debug_generated_frames/debug_generated_frame_{i}_{k}.png")
                del frame_np, img  # Free memory immediately

            generated_frames_np = (
                generated_frames
                .permute(0, 2, 3, 1)   # [T, H, W, C]
                .numpy()
            )
            generated_frames_np = (generated_frames_np * 255).astype(np.uint8)
            gt_frames = (pixel_values[0].permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
    
        
            # Compute rewards
            reward = reward_fn.compute_lpips_reward(generated_frames_np, gt_frames) 
            reward = torch.tensor(reward, device=accelerator.device, dtype=weight_dtype).unsqueeze(0)  # (1, )
            
            # Clear intermediate tensors
            del generated_frames, generated_frames_np, gt_frames, output, pixel_values, flows

            # print("Timesteps shape:", timesteps.shape) # Timesteps shape: torch.Size([1, 25])
            # print("Latents shape:", latents.shape) # Latents shape: torch.Size([1, 26, 25, 4, 48, 48])
            # print("Log probs shape:", log_probs.shape)  # Log probs shape: torch.Size([1, 25])
            # print("Reward shape:", reward.shape) # Reward shape: torch.Size([1])
            # print("Flow shape:", controlnet_flow.shape) # Flow shape: torch.Size([1, 24, 2, 384, 384])
            # print("Controlnet condition shape:", controlnet_condition.shape)# Controlnet condition shape: torch.Size([1, 3, 384, 384])
            # print("Image shape:", image.shape)# Image shape: torch.Size([1, 3, 384, 384])
            
            print("Image latents at 1208", image_latents.shape)

            samples.append({
                "timesteps": timesteps.detach(),
                "latents": latents[:, :-1].detach(),   # each entry is the latent before timestep t
                "next_latents": latents[:, 1:].detach(),  # each entry is the latent after timestep t
                "log_probs": log_probs.detach(),
                "rewards": reward.detach(),
                # Condition for the model - detach to save memory
                "controlnet_flow": controlnet_flow.detach(),
                "controlnet_condition": controlnet_condition.detach(),
                "added_time_ids": added_time_ids.detach(),
                "image_embeddings": image_embeddings.detach(),
                "image_latents": image_latents.detach(),
                "guidance_scale": guidance_scale.detach(),
            })
            
            # Delete references to free memory
            del timesteps, latents, log_probs, reward, controlnet_flow, controlnet_condition
            del added_time_ids, image_embeddings, image_latents, guidance_scale, pil_images
            
            # Clear memory after each batch
            clear_memory()

        # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
        samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}
        
        # Clear memory after collating samples
        clear_memory()
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
        print("Total batch size:", total_batch_size)

        #####################END OF SAMPLING ##########################



        
        #################### PPO TRAINING ####################
        logger.info(f"Epoch {epoch}: Training...")
        controlnet.train()
        for inner_epoch in range(args.num_inner_epochs):            
            # shuffle samples along batch dimension: CONSIDER CAN REMOVE SINCE IT DOES NOTHING???
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
            # Keys with time dimension that were shuffled
            time_dependent_keys = ["timesteps", "latents", "next_latents", "log_probs", "advantages"]
            samples_batched = {}
            
            for k, v in samples.items():
                if k in time_dependent_keys:
                    # These have time dimension: (total_batch_size, num_timesteps, ...)
                    samples_batched[k] = v.reshape(-1, args.per_gpu_batch_size, *v.shape[1:])
                else:
                    # Conditioning tensors without time dimension: (total_batch_size, ...)
                    # Keep them as (total_batch_size, ...) to match the doubled batch for CFG
                    samples_batched[k] = v.reshape(-1, args.per_gpu_batch_size * 2, *v.shape[1:])

            print("After rebatch:")
            print(samples_batched["timesteps"].shape)  # (num_batches, bs_per_gpu, num_steps)
            # If it is 4 then should be (4, 1, 25) since bs per gpu is 1

            # dict of lists -> list of dicts for easier iteration
            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]
            # for b in samples_batched:
            #     for k, v in b.items():
            #         print(f"{k} shape: {v.shape}")
            #         if k == "timesteps":
            #             print(v)
            
            print("Len of samples_batched:", len(samples_batched))  # Should be num_batches_per_epoch

            info = defaultdict(list) 
            
            for i, sample in tqdm(list(enumerate(samples_batched)), desc=f"Epoch {epoch}.{inner_epoch}: training", position=0, disable=not accelerator.is_local_main_process):
                controlnet_condition = sample["controlnet_condition"]
                controlnet_flow = sample["controlnet_flow"]
                added_time_ids = sample["added_time_ids"]
                image_latents = sample["image_latents"]
                print("Image latents shape (train):", image_latents.shape)
                image_embeddings = sample["image_embeddings"]
                guidance_scale = sample["guidance_scale"]


                # Looping over 25 timesteps
                num_train_timesteps = sample["timesteps"].shape[-1]
                for j in tqdm(range(num_train_timesteps), desc=f"Inner step {i}", position=1, leave=False, disable=not accelerator.is_local_main_process):    
                    with accelerator.accumulate(controlnet):
                        latents = sample["latents"][:, j, :, :, :]  # (bs, 1, num_frames, C, H, W)
                        next_latents = sample["next_latents"][:, j, :, :, :]  # (bs, 1, num_frames, C, H, W)
                        timesteps = sample["timesteps"][:, j]  # (bs, 1)
                        old_log_probs = sample["log_probs"][:, j]  # (bs, 1)

                        # always do cfg 
                        latent_model_input = torch.cat([latents] * 2)
                        latent_model_input = noise_scheduler.scale_model_input(latent_model_input, timesteps)

                        latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)

                        down_block_res_samples, mid_block_res_sample, controlnet_flow, _ = controlnet(
                            latent_model_input,
                            timesteps,
                            encoder_hidden_states=image_embeddings,
                            controlnet_cond=controlnet_condition,
                            controlnet_flow=controlnet_flow,
                            added_time_ids=added_time_ids,
                            conditioning_scale=1.0, # fixed to 1.0 since the pipeline use this
                            guess_mode=False,
                            return_dict=False,
                        )

                        # predict the noise residual
                        noise_pred = unet(
                            latent_model_input,
                            timesteps,
                            encoder_hidden_states=image_embeddings,
                            down_block_additional_residuals=down_block_res_samples,
                            mid_block_additional_residual=mid_block_res_sample,
                            added_time_ids=added_time_ids,
                            return_dict=False,
                        )[0]

                        # perform cfg:
                        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                        latents_and_log_prob = noise_scheduler.step(noise_pred, timesteps, latents)
                        latents = latents_and_log_prob.prev_sample
                        log_prob = latents_and_log_prob.log_prob 

                        # ppo logic
                        advantages = torch.clamp(
                            sample["advantages"],
                            -args.adv_clip_max,
                            args.adv_clip_max,
                        )
                        ratio = torch.exp(log_prob - old_log_probs)
                        unclipped_loss = -advantages * ratio
                        clipped_loss = -advantages * torch.clamp(
                            ratio,
                            1.0 - args.clip_range,
                            1.0 + args.clip_range,
                        )
                        loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                        info["approx_kl"].append(
                            0.5
                            * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2)
                        )
                        info["clipfrac"].append(
                            torch.mean(
                                (
                                    torch.abs(ratio - 1.0) > args.clip_range
                                ).float()
                            )
                        )
                        info["loss"].append(loss)

                        # backward pass
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(
                                controlnet.parameters(), args.max_grad_norm
                            )
                        optimizer.step()
                        optimizer.zero_grad()
                        
                        # Clear intermediate tensors
                        del latents, next_latents, latent_model_input, down_block_res_samples
                        del mid_block_res_sample, noise_pred, noise_pred_uncond, noise_pred_cond
                        del latents_and_log_prob, log_prob, ratio, unclipped_loss, clipped_loss, loss

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        assert (j == num_train_timesteps - 1) and (
                            i + 1
                        ) % args.gradient_accumulation_steps == 0
                        # log training-related stuff
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        accelerator.log(info, step=global_step)
                        global_step += 1
                        info = defaultdict(list)
                        
                        # Clear cache after each optimization step
                        clear_memory()
            # make sure we did an optimization step at the end of the inner epoch
            assert accelerator.sync_gradients
        
        # Clear memory after training epoch
        clear_memory()

        if epoch != 0 and epoch % args.save_freq == 0 and accelerator.is_main_process:
            accelerator.save_state()
    
                
        
        #################### END OF PPO TRAINING ####################
        
        # Checkpointing and validation (once per outer epoch)
        if accelerator.is_main_process:
            # save checkpoints!
            if epoch % (args.checkpointing_steps // args.num_batches_per_epoch + 1) == 0:
                # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                if args.checkpoints_total_limit is not None:
                    checkpoints = os.listdir(args.output_dir)
                    checkpoints = [
                        d for d in checkpoints if d.startswith("checkpoint")]
                    checkpoints = sorted(
                        checkpoints, key=lambda x: int(x.split("-")[1]))

                    # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                    if len(checkpoints) >= args.checkpoints_total_limit:
                        num_to_remove = len(
                            checkpoints) - args.checkpoints_total_limit + 1
                        removing_checkpoints = checkpoints[0:num_to_remove]

                        logger.info(
                            f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                        )
                        logger.info(
                            f"removing checkpoints: {', '.join(removing_checkpoints)}")

                        for removing_checkpoint in removing_checkpoints:
                            removing_checkpoint = os.path.join(
                                args.output_dir, removing_checkpoint)
                            shutil.rmtree(removing_checkpoint)

                save_path = os.path.join(
                    args.output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_path)
                logger.info(f"Saved state to {save_path}")
        
        if global_step >= args.max_train_steps:
            break
    

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        controlnet = accelerator.unwrap_model(controlnet)
        if args.use_ema:
            ema_controlnet.copy_to(controlnet.parameters())

        pipeline = FlowControlNetPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            image_encoder=accelerator.unwrap_model(image_encoder),
            vae=accelerator.unwrap_model(vae),
            unet=unet,
            controlnet=controlnet,
            revision=args.revision,
        )
        pipeline.save_pretrained(args.output_dir)

        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )
    accelerator.end_training()


if __name__ == "__main__":
    main()
