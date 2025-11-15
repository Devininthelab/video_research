import argparse
import logging
import math
import os
import time
import shutil
from pathlib import Path
from collections import defaultdict
from contextlib import nullcontext
import copy
import socket
import struct
import zlib

import numpy as np
import pickle
import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from tqdm.auto import tqdm
from tqdm import trange
from einops import rearrange

import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

import diffusers
from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from models.unet_spatio_temporal_condition_controlnet import UNetSpatioTemporalConditionControlNetModel
from models.svdxt_featureflow_forward_controlnet_s2d_fixcmp_norefine import FlowControlNet
from pipeline.pipeline import FlowControlNetPipeline
from train_utils.dataset import WebVid10M
from train_utils.unimatch.unimatch.unimatch import UniMatch
from train_stage1 import get_optical_flows
import structlog
logger = structlog.get_logger()


COMPRESSION_LEVEL = 6
CHUNK_SIZE = 4 * 1024 * 1024


def send_large_data(sock, data, compress=True):
    serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)

    if compress:
        compressed = zlib.compress(serialized, level=COMPRESSION_LEVEL)
        data_to_send = compressed
        is_compressed = 1
    else:
        data_to_send = serialized
        is_compressed = 0
    
    total_size = len(data_to_send)
    sock.sendall(struct.pack('!BI', is_compressed, total_size))

    offset = 0
    while offset < total_size:
        chunk = data_to_send[offset:offset+CHUNK_SIZE]
        sock.sendall(chunk)
        offset += len(chunk)
    
    logger.info(f"Sent {total_size / 1024 / 1024:.2f} MB (compressed: {compress})")


def recv_large_data(sock):
    metadata = sock.recv(5)
    is_compressed, total_size = struct.unpack("!BI", metadata)

    logger.info(f"Receiving {total_size / 1024 / 1024:.2f} MB")
    data = b''
    while len(data) < total_size:
        chunk_size = min(CHUNK_SIZE, total_size - len(data))
        chunk = sock.recv(chunk_size)
        if not chunk:
            raise ConnectionError("Connection closed while receiving data")
        
        data += chunk
        logger.info(f"Progress: {len(data) / 1024 / 1024:.1f} / {total_size / 1024 / 1024:.1f} MB")
    
    if is_compressed:
        logger.info("Decompressing data")
        data = zlib.decompress(data)
    
    logger.info("Deserializing data")
    return pickle.loads(data)


class SamplingClient:
    """Client to communicate with sampling server"""
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.socket = None

    def connect(self):
        # establish tcp connection
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2 * 1024 * 1024)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 2 * 1024 * 1024)
        self.socket.connect((self.host, self.port))
        logger.info(f"Connected to sampling server at {self.host}:{self.port}")

    def disconnect(self):
        if self.socket:
            cmd = 'CLOSE'
            self.socket.sendall(struct.pack('!I', len(cmd)))
            self.socket.sendall(cmd.encode('utf-8'))
            self.socket.close()

    # def request_samples(self, epoch, num_batches):
    #     """Request samples from the server"""
    #     cmd = 'SAMPLE'
    #     self.socket.sendall(struct.pack('!I', len(cmd)))
    #     self.socket.sendall(cmd.encode('utf-8'))
    #     self.socket.sendall(struct.pack('!I', epoch))
    #     self.socket.sendall(struct.pack('!I', num_batches))

    #     # receive samples
    #     samples = recv_large_data(self.socket)
        
    #     for sample in samples:
    #         sample['pixel_values'] = sample['pixel_values']
    #         sample['flows'] = sample['flows']

    #     return samples

    def get_samples(self, num_samples):
        # request samples from server queue
        cmd = "GET_SAMPLES"
        self.socket.sendall(struct.pack('!I', len(cmd)))
        self.socket.sendall(cmd.encode('utf-8'))
        self.socket.sendall(struct.pack('!I', num_samples))

        # receive samples
        logger.info(f"Requesting {num_samples} samples from server")
        samples = recv_large_data(self.socket)
        logger.info(f"Received {len(samples)} samples")

        # convert to tensors if needed
        for sample in samples:
            sample['pixel_values'] = sample['pixel_values']
            sample['flows'] = sample['flows']
        
        return samples


    def update_server_weights(self, state_dict):
        # sends updated ControlNet weights
        """Send updated weights to the server"""
        cmd = 'UPDATE_WEIGHTS'
        self.socket.sendall(struct.pack('!I', len(cmd)))
        self.socket.sendall(cmd.encode('utf-8'))

        logger.info("Sending weight update to server")
        send_large_data(self.socket, state_dict, compress=True)

        response = self.socket.recv(2)
        success = response == b'OK'
        if success:
            logger.info("Server weights updated successfully")
        return success

    
    def get_queue_size(self):
        #checks the server queue and status
        cmd = 'QUEUE_SIZE'
        self.socket.sendall(struct.pack('!I', len(cmd)))
        self.socket.sendall(cmd.encode('utf-8'))
        size = struct.unpack('!I', self.socket.recv(4))[0]
        return size

    


def parse_args():
    parser = argparse.ArgumentParser(description="DDPO training for motion-aware video generation")

    parser.add_argument('--server_host', type=str, default='localhost')
    parser.add_argument('--server_port', type=int, default=9999)
    
    # Model arguments
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--controlnet_model_name_or_path", type=str, required=True)
    parser.add_argument("--revision", type=str, default=None)
    
    # Video generation parameters
    parser.add_argument("--num_frames", type=int, default=25)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--sample_stride", type=int, default=1)
    
    # DDPO specific arguments
    parser.add_argument("--num_epochs", type=int, default=10000000)
    parser.add_argument("--num_inner_epochs", type=int, default=4, help="PPO inner epochs")
    parser.add_argument("--sample_num_batches_per_epoch", type=int, default=4)
    parser.add_argument("--train_gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--weight_update_freq", type=int, default=20, help="Update server weights every N epochs")
    parser.add_argument("--reference_update_freq", type=int, default=10, help="Update reference policy every N epochs")
    
    # PPO hyperparameters
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--clip_range", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--clip_range_value", type=float, default=0.0001, help="Value function clip range")
    parser.add_argument("--gamma", type=float, default=1.0, help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--adv_clip_max", type=float, default=5.0, help="Advantage clipping")
    parser.add_argument("--kl_penalty", type=float, default=0.1, help="KL penalty coefficient")
    parser.add_argument("--target_kl", type=float, default=0.01, help="Target KL divergence")
    parser.add_argument("--kl_warmup_steps", type=int, default=100, help="Steps to warmup KL penalty")
    
    # Reward function arguments
    parser.add_argument("--use_temporal_consistency_reward", action="store_true")
    parser.add_argument("--temporal_consistency_weight", type=float, default=0.2)
    
    # Optimizer arguments
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-4)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    
    # Training configuration
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./ddpo_outputs")
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--report_to", type=str, default="wandb")
    
    # Checkpointing
    parser.add_argument("--save_freq", type=int, default=50, help="Save checkpoint every N epochs")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--num_checkpoint_limit", type=int, default=5)
    
    # Inference parameters
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--motion_bucket_id", type=int, default=127)
    parser.add_argument("--fps", type=int, default=7)
    parser.add_argument("--noise_aug_strength", type=float, default=0.02)
    
    args = parser.parse_args()
    return args


def encode_image(pixel_values, feature_extractor, image_encoder, weight_dtype, device):
    """Encode image for conditioning."""
    pixel_values = pixel_values * 2.0 - 1.0
    from train_stage1 import _resize_with_antialiasing
    pixel_values = _resize_with_antialiasing(pixel_values, (224, 224))
    pixel_values = (pixel_values + 1.0) / 2.0
    
    pixel_values = feature_extractor(
        images=pixel_values,
        do_normalize=True,
        do_center_crop=False,
        do_resize=False,
        do_rescale=False,
        return_tensors="pt",
    ).pixel_values
    
    pixel_values = pixel_values.to(device=device, dtype=weight_dtype)
    image_embeddings = image_encoder(pixel_values).image_embeds
    return image_embeddings.unsqueeze(1)


def _get_add_time_ids(
    fps,
    motion_bucket_ids,  # Expecting a list of tensor floats
    noise_aug_strength,
    dtype,
    batch_size,
    unet=None,
):
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


MIN_VALUE = 0.002
MAX_VALUE = 700
IMAGE_D = 64
NOISE_D_LOW = 32
NOISE_D_HIGH = 64
SIGMA_DATA = 0.5


def ppo_step_with_reference_policy(
    unet,
    controlnet,
    reference_controlnet,
    optimizer,
    accelerator,
    samples,
    args,
    weight_dtype,
    device,
    feature_extractor,
    image_encoder,
    vae
):  

    total_loss = 0.0
    total_policy_loss = 0.0
    total_kl_loss = 0.0
    total_clip_fraction = 0.0
    total_approx_kl = 0.0

    rewards_list = [float(s['reward']) for s in samples]
    rewards_tensor = torch.tensor(rewards_list, dtype=torch.float32, device=device)
    
    # compute advantages with better normalization
    adv = rewards_tensor - rewards_tensor.mean()
    adv_std = adv.std(unbiased=False)
    # Use a larger epsilon to prevent over-normalization
    if adv_std.item() > 1e-6:
        adv = adv / (adv_std + 1e-5)
    else:
        # If all rewards are identical, set advantages to zero
        logger.warning(f"All rewards are nearly identical (std={adv_std.item():.2e}), setting advantages to zero")
        adv = torch.zeros_like(adv)
    adv = torch.clamp(adv, -args.adv_clip_max, args.adv_clip_max)

    # reference policy forward pass
    reference_controlnet.eval()
    reference_outputs = [] # store reference outputs for each noised sample

    with torch.no_grad():
        for sample in tqdm(samples):
            pixel_values = sample['pixel_values'].to(device)
            flows = sample['flows'].to(device)

            from train_stage1 import tensor_to_vae_latent
            latents = tensor_to_vae_latent(pixel_values, vae)

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]

            sigmas = rand_cosine_interpolated(
                shape=[bsz,],
                image_d=IMAGE_D,
                noise_d_low=NOISE_D_LOW,
                noise_d_high=NOISE_D_HIGH,
                sigma_data=SIGMA_DATA,
                min_value=MIN_VALUE,
                max_value=MAX_VALUE,
                device=latents.device,
                dtype=latents.dtype
            )

            sigmas_reshaped = sigmas.clone()
            while len(sigmas_reshaped.shape) < len(latents.shape):
                sigmas_reshaped = sigmas_reshaped.unsqueeze(-1)
            
            train_noise_aug = 0.02
            small_noise_latents = latents + noise * train_noise_aug
            conditional_latents = small_noise_latents[:, 0, :, :, :] # take the first frame only for controlnet, bs, f, c, h, w
            conditional_latents = conditional_latents / vae.config.scaling_factor

            noisy_latents = latents + noise * sigmas_reshaped
            timesteps = torch.Tensor([0.25 * sigma.log() for sigma in sigmas]).to(latents.device)
            inp_noisy_latents = noisy_latents / ((sigmas_reshaped ** 2 + 1) ** 0.5)
            
            cond_image = pixel_values[:, 0, :, :, :]
            encoder_hidden_states = encode_image(
                cond_image.float(), feature_extractor, image_encoder, weight_dtype, device
            ) # conditional image
            
            added_time_ids = _get_add_time_ids(
                6, 127, train_noise_aug, encoder_hidden_states.dtype, bsz, unet
            )
            added_time_ids = added_time_ids.to(latents.device)
            conditional_latents = conditional_latents.unsqueeze(1).repeat(1, noisy_latents.shape[1], 1, 1, 1)
            inp_noisy_latents_cat = torch.cat([inp_noisy_latents, conditional_latents], dim=2)
            
            down_block_res_samples, mid_block_res_sample, _, _ = reference_controlnet(
                inp_noisy_latents_cat,
                timesteps,
                encoder_hidden_states,
                added_time_ids=added_time_ids,
                controlnet_cond=cond_image,
                controlnet_flow=flows,
                return_dict=False
            )
            
            model_pred_ref = unet(
                inp_noisy_latents_cat,
                timesteps,
                encoder_hidden_states,
                added_time_ids=added_time_ids,
                down_block_additional_residuals=[
                    s.to(dtype=weight_dtype) for s in down_block_res_samples
                ],
                mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype)
            ).sample

            reference_outputs.append({
                'model_pred': model_pred_ref,
                'noise': noise,
                'sigmas': sigmas,
                'sigmas_reshaped': sigmas_reshaped,
                'latents': latents,
                'noisy_latents': noisy_latents
            })
            
    
    controlnet.train()
    # current policy forward pass + loss
    for idx, sample in enumerate(samples):
        pixel_values = sample['pixel_values'].to(device)
        flows = sample['flows'].to(device)
        advantage = adv[idx]

        ref_data = reference_outputs[idx]
        latents = ref_data['latents']
        noise = ref_data['noise']
        sigmas = ref_data['sigmas']
        sigmas_reshaped = ref_data['sigmas_reshaped']
        noisy_latents = ref_data['noisy_latents']

        bsz = latents.shape[0]
        
        train_noise_aug = 0.02
        small_noise_latents = latents + noise * train_noise_aug
        conditional_latents = small_noise_latents[:, 0, :, :, :]
        conditional_latents = conditional_latents / vae.config.scaling_factor
        
        timesteps = torch.Tensor([0.25 * sigma.log() for sigma in sigmas]).to(latents.device)
        inp_noisy_latents = noisy_latents / ((sigmas_reshaped ** 2 + 1) ** 0.5)
        
        cond_image = pixel_values[:, 0, :, :, :]
        encoder_hidden_states = encode_image(
            cond_image.float(), feature_extractor, image_encoder, weight_dtype, device
        )
        
        added_time_ids = _get_add_time_ids(
            6, 127, train_noise_aug, encoder_hidden_states.dtype, bsz, unet
        )
        added_time_ids = added_time_ids.to(latents.device)
        conditional_latents = conditional_latents.unsqueeze(1).repeat(1, noisy_latents.shape[1], 1, 1, 1)
        inp_noisy_latents_cat = torch.cat([inp_noisy_latents, conditional_latents], dim=2)
        
        down_block_res_samples, mid_block_res_sample, _, _ = controlnet(
            inp_noisy_latents_cat,
            timesteps,
            encoder_hidden_states,
            added_time_ids=added_time_ids,
            controlnet_cond=cond_image,
            controlnet_flow=flows,
            return_dict=False
        )
        
        model_pred = unet(
            inp_noisy_latents_cat,
            timesteps,
            encoder_hidden_states,
            added_time_ids=added_time_ids,
            down_block_additional_residuals=[
                s.to(dtype=weight_dtype) for s in down_block_res_samples
            ],
            mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype)
        ).sample
        
        c_out = -sigmas_reshaped / ((sigmas_reshaped ** 2 + 1) ** 0.5)
        c_skip = 1 / (sigmas_reshaped ** 2 + 1)
        denoised_latents = model_pred * c_out + c_skip * noisy_latents
        model_pred_ref = ref_data['model_pred']
        denoised_latents_ref = model_pred_ref * c_out + c_skip * noisy_latents

        weighing = (1 + sigmas_reshaped ** 2) * (sigmas_reshaped ** -2.0)
        weighing = torch.clamp(weighing, max=1000.0)
        mse_current = (weighing * (denoised_latents.float() - latents.float()) ** 2).reshape(bsz, -1).mean(dim=1)
        mse_ref = (weighing * (denoised_latents_ref.float() - latents.float()) ** 2).reshape(bsz, -1).mean(dim=1)

        log_prob_current = -mse_current
        log_prob_ref = -mse_ref.detach()

        ratio = torch.exp(log_prob_current - log_prob_ref.detach())
        clipped_ratio = torch.clamp(ratio, 1.0 - args.clip_range, 1.0 + args.clip_range)
        policy_loss = -torch.min(ratio * advantage.detach(), clipped_ratio * advantage.detach()).mean()

        kl_penalty = F.mse_loss(denoised_latents.float(), denoised_latents_ref.float())
        with torch.no_grad():
            clip_fraction = (torch.abs(ratio - 1.0) > args.clip_range).float().mean()
            total_clip_fraction += clip_fraction.item()
            # Approximate KL divergence
            approx_kl = ((ratio - 1) - torch.log(ratio)).mean()
            total_approx_kl += approx_kl.item()

        loss = policy_loss + args.kl_penalty * kl_penalty
        loss = loss / args.train_gradient_accumulation_steps
        accelerator.backward(loss)

        total_loss += loss.item()
        total_policy_loss += policy_loss.item()
        total_kl_loss += kl_penalty.item()
    
    if args.max_grad_norm > 0:
        accelerator.clip_grad_norm_(controlnet.parameters(), args.max_grad_norm)
    
    optimizer.step()
    optimizer.zero_grad()
    
    return {
        'loss': total_loss / len(samples),
        'policy_loss': total_policy_loss / len(samples),
        'kl_loss': total_kl_loss / len(samples),
        'clip_fraction': total_clip_fraction / len(samples),
        'approx_kl': total_approx_kl / len(samples)
    }



if __name__ == '__main__':
    args = parse_args()
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
        log_with=args.report_to
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Loading models")
    noise_scheduler = EulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder='scheduler'
    )

    feature_extractor = CLIPImageProcessor.from_pretrained(
        args.pretrained_model_name_or_path, subfolder='feature_extractor'
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder='image_encoder', variant='fp16'
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.pretrained_model_name_or_path, subfolder='vae', variant='fp16'
    )
    unet = UNetSpatioTemporalConditionControlNetModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder='unet', variant='fp16'
    )
    controlnet = FlowControlNet.from_pretrained(args.controlnet_model_name_or_path)
    reference_controlnet = copy.deepcopy(controlnet)
    reference_controlnet.requires_grad_(False)
    reference_controlnet.eval()

    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    # set trainable parameters
    controlnet.requires_grad_(True)

    # Define Unimatch for optical flow prediction
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

    # setup dtype
    weight_dtype = torch.float32
    if accelerator.mixed_precision == 'fp16':
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == 'bf16':
        weight_dtype = torch.bfloat16
    
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unimatch.to(accelerator.device, dtype=weight_dtype)

    optimizer = torch.optim.AdamW(
        list(controlnet.parameters()),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon
    )
    unet, controlnet, optimizer = accelerator.prepare(unet, controlnet, optimizer)

    # connect to sampling server
    client = SamplingClient(args.server_host, args.server_port)
    client.connect()

    if accelerator.is_main_process:
        accelerator.init_trackers("ddpo_video_generation", config=vars(args))
    
    logger.info("***** Running DDPO training *****")
    logger.info(f"Num epochs = {args.num_epochs}")
    logger.info(f"Weight updates frequency = every {args.weight_update_freq} epochs")

    global_step = 0
    reward_ema = None
    kl_ema = None
    
    for epoch in trange(args.num_epochs):
        logger.info(f"Epoch {epoch + 1}/{args.num_epochs}")

        if accelerator.is_main_process:
            try:
                queue_size = client.get_queue_size()
                logger.info(f"Server queue size: {queue_size}")
            except:
                logger.warning(f"Could not get queue size")

        # request samples from server
        logger.info("Requesting samples from server")
        # samples = client.request_samples(epoch, args.sample_num_batches_per_epoch)
        samples = client.get_samples(args.sample_num_batches_per_epoch) # sample_num_batches_per_epoch = 4

        if not samples:
            logger.warning("No samples available, waiting")
            time.sleep(5)
            continue

        rewards = [s['reward'] for s in samples]
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        logger.info(f"Mean reward: {mean_reward:.4f} +/- {std_reward:.4f}")

        # train with ppo
        logger.info("Training with PPO")
        controlnet.train()

        for inner_epoch in range(args.num_inner_epochs): # 4 = num_inner_epochs
            metrics = ppo_step_with_reference_policy(unet, controlnet, reference_controlnet, optimizer, accelerator, samples, args, weight_dtype, accelerator.device, feature_extractor, image_encoder, vae)
            
            # Update KL EMA
            if kl_ema is None:
                kl_ema = metrics['approx_kl']
            else:
                kl_ema = 0.9 * kl_ema + 0.1 * metrics['approx_kl']
            
            # Adaptive early stopping based on KL divergence
            if metrics['approx_kl'] > 1.5 * args.target_kl:
                logger.warning(f"Early stopping at inner epoch {inner_epoch} due to high KL ({metrics['approx_kl']:.4f} > {1.5 * args.target_kl:.4f})")
                break
            
            global_step += 1
            if inner_epoch == args.num_inner_epochs - 1 and accelerator.is_main_process:
                # Update reward EMA
                if reward_ema is None:
                    reward_ema = mean_reward
                else:
                    reward_ema = 0.9 * reward_ema + 0.1 * mean_reward
                
                current_lr = optimizer.param_groups[0]['lr']
                accelerator.log({
                    'train/loss': metrics['loss'],
                    'train/policy_loss': metrics['policy_loss'],
                    'train/kl_loss': metrics['kl_loss'],
                    'train/approx_kl': metrics['approx_kl'],
                    'train/kl_ema': kl_ema,
                    'train/clip_fraction': metrics['clip_fraction'],
                    'train/mean_reward': mean_reward,
                    'train/std_reward': std_reward,
                    'train/reward_ema': reward_ema,
                    'train/learning_rate': current_lr,
                    'epoch': epoch
                }, step=global_step)
            
        # Update reference policy more frequently to prevent divergence
        if (epoch + 1) % args.reference_update_freq == 0:
            logger.info(f"Updating client's reference controlnet (epoch {epoch+1})")
            controlnet_state = accelerator.unwrap_model(controlnet).state_dict()
            reference_controlnet.load_state_dict(controlnet_state)
            reference_controlnet.eval()
            logger.info("Client's reference controlnet updated")
        
        # Update server weights less frequently
        if (epoch + 1) % args.weight_update_freq == 0 and accelerator.is_main_process:
            logger.info(f"Updating server weights (epoch {epoch+1})")
            controlnet_state = accelerator.unwrap_model(controlnet).state_dict()
            client.update_server_weights(controlnet_state)
        
        if (epoch + 1) % args.save_freq == 0 and accelerator.is_main_process:
            save_path = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch+1}")
            checkpoints = sorted(
                [d for d in os.listdir(args.output_dir) if d.startswith('checkpoint-epoch-')],
                key=lambda x: int(x.split('-')[-1])
            )
            if len(checkpoints) >= args.num_checkpoint_limit:
                for old_checkpoint in checkpoints[:-args.num_checkpoint_limit + 1]:
                    shutil.rmtree(os.path.join(args.output_dir, old_checkpoint))
                
            accelerator.save_state(save_path)
            logger.info(f"Saved checkpoint to {save_path}")
    
    client.disconnect()
    accelerator.end_training()