#!/usr/bin/env python
# coding=utf-8
"""Minimal script to test video generation with MOFA-Video."""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from tqdm import tqdm

from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import AutoencoderKLTemporalDecoder
from utils.scheduling_ddim_fix import DDIMScheduler
from utils.scheduling_ddim_with_logprob import DDIMSchedulerWithLogProb

from models.unet_spatio_temporal_condition_controlnet import UNetSpatioTemporalConditionControlNetModel
from pipeline.pipeline_with_logprob import FlowControlNetPipelineWithLogProb
from models.svdxt_featureflow_forward_controlnet_s2d_fixcmp_norefine import FlowControlNet
from train_utils.unimatch.unimatch.unimatch import UniMatch
from train_utils.unimatch.utils.flow_viz import flow_to_image
from train_utils.dataset import WebVid10M


def preprocess_size(image1, image2, padding_factor=32):
    '''Prepare images for optical flow estimation'''
    transpose_img = False
    # the model is trained with size: width > height
    if image1.size(-2) > image1.size(-1):
        image1 = torch.transpose(image1, -2, -1)
        image2 = torch.transpose(image2, -2, -1)
        transpose_img = True
        
    inference_size = [384, 512]
    ori_size = image1.shape[-2:]

    # resize before inference
    if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
        image1 = F.interpolate(image1, size=inference_size, mode='bilinear', align_corners=True)
        image2 = F.interpolate(image2, size=inference_size, mode='bilinear', align_corners=True)
    
    return image1, image2, inference_size, ori_size, transpose_img


def postprocess_size(flow_pr, inference_size, ori_size, transpose_img):
    '''Postprocess optical flow to original size'''
    if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
        flow_pr = F.interpolate(flow_pr, size=ori_size, mode='bilinear', align_corners=True)
        flow_pr[:, 0] = flow_pr[:, 0] * ori_size[-1] / inference_size[-1]
        flow_pr[:, 1] = flow_pr[:, 1] * ori_size[-2] / inference_size[-2]

    if transpose_img:
        flow_pr = torch.transpose(flow_pr, -2, -1)
    
    return flow_pr


@torch.no_grad()
def get_optical_flows(unimatch, video_frame):
    '''
    Extract optical flow from video frames using Unimatch.
    Args:
        video_frame: [b, t, c, h, w] in range [0, 1]
    Returns:
        flows: [b, t-1, 2, h, w] - flow from frame 0 to all subsequent frames
    '''
    video_frame = video_frame * 255

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
    
    flows = torch.cat(flows, dim=1).to(torch.float16)  # [b, t-1, 2, h, w]
    return flows


def parse_args():
    parser = argparse.ArgumentParser(description="Test MOFA-Video generation pipeline")
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default="./ckpts/stable-video-diffusion-img2vid-xt-1-1",
        help="Path to pretrained SVD model",
    )
    parser.add_argument(
        "--controlnet_path",
        type=str,
        default="./ckpts/controlnet",
        help="Path to trained ControlNet checkpoint",
    )
    parser.add_argument(
        "--unimatch_path",
        type=str,
        default="./train_utils/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth",
        help="Path to UniMatch checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./test_outputs",
        help="Output directory for generated videos",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of videos to generate",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=25,
        help="Number of frames to generate",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=320,
        help="Video height",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Video width",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for generation",
    )
    parser.add_argument(
        "--controlnet_scale",
        type=float,
        default=1.0,
        help="ControlNet conditioning scale (0-1)",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/projects_vol/gp_slab/minhthan001/data_webvid_reduce/reduced_clean_results_2M_train.csv",
        help="Path to dataset CSV file",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/projects_vol/gp_slab/minhthan001/data_webvid_reduce/reduced_WebVid",
        help="Root directory containing the video files",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="euler",
        choices=["euler", "ddim"],
        help="Scheduler to use (euler or ddim)",
    )
    parser.add_argument(
        "--return_log_prob",
        action="store_true",
        help="Whether to return log probabilities from DDIM scheduler",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device and dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float16
    
    print("=" * 60)
    print("Loading models...")
    print("=" * 60)
    
    # Load UniMatch for optical flow extraction
    print("Loading UniMatch...")
    unimatch = UniMatch(
        feature_channels=128,
        num_scales=2,
        upsample_factor=4,
        num_head=1,
        ffn_dim_expansion=4,
        num_transformer_layers=6,
        reg_refine=True,
        task='flow'
    ).to(device)
    
    checkpoint = torch.load(args.unimatch_path)
    unimatch.load_state_dict(checkpoint['model'])
    unimatch.eval()
    unimatch.requires_grad_(False)
    print("✓ UniMatch loaded")
    
    # Load SVD components
    print("Loading SVD components...")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model_path, 
        subfolder="image_encoder", 
        variant="fp16"
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.pretrained_model_path, 
        subfolder="vae", 
        variant="fp16"
    )
    unet = UNetSpatioTemporalConditionControlNetModel.from_pretrained(
        args.pretrained_model_path,
        subfolder="unet",
        variant="fp16",
    )
    print("✓ SVD components loaded")
    
    # Load ControlNet
    print(f"Loading ControlNet from {args.controlnet_path}...")
    controlnet = FlowControlNet.from_pretrained(args.controlnet_path)
    print("✓ ControlNet loaded")
    
    # Move models to device
    image_encoder.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)
    controlnet.to(device, dtype=weight_dtype)
    
    # Set to eval mode
    image_encoder.eval()
    vae.eval()
    unet.eval()
    controlnet.eval()
    
    # Create pipeline
    print("Creating pipeline...")
    pipeline_args = {
        "unet": unet,
        "controlnet": controlnet,
        "image_encoder": image_encoder,
        "vae": vae,
        "torch_dtype": weight_dtype,
    }

    if args.scheduler == "ddim":
        print("Using DDIM Scheduler")
        if args.return_log_prob:
            print("  - Returning log probabilities from DDIM Scheduler")
            pipeline_args["scheduler"] = DDIMSchedulerWithLogProb.from_pretrained(
                args.pretrained_model_path,
                subfolder="scheduler"
            )
        else:
            pipeline_args["scheduler"] = DDIMScheduler.from_pretrained(
                args.pretrained_model_path,
                subfolder="scheduler"
            )
    else:
        print("Using Euler Discrete Scheduler")

    pipeline = FlowControlNetPipelineWithLogProb.from_pretrained(
        args.pretrained_model_path,
        **pipeline_args
    )
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=False)
    print("✓ Pipeline created")
    
    # Load test dataset
    print("\nLoading test dataset...")
    test_dataset = WebVid10M(
        meta_path=args.dataset_path,
        data_dir=args.data_dir,
        sample_size=[args.height, args.width],
        sample_n_frames=args.num_frames,
        sample_stride=1
    )
    print(f"✓ Test dataset loaded with {len(test_dataset)} samples")
    
    # Set seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    print("\n" + "=" * 60)
    print(f"Generating {args.num_samples} videos...")
    print("=" * 60)
    
    # Generate videos
    for idx in tqdm(range(args.num_samples), desc="Generating videos"):
        # Get sample from dataset
        sample = test_dataset[idx]
        
        # Prepare inputs
        pixel_values = sample['pixel_values'].unsqueeze(0).to(device, dtype=weight_dtype)  # [1, T, C, H, W]
        video_name = sample['video_name'].replace('/', '_').split('.')[0]
        
        # Extract optical flows
        print(f"\n[{idx+1}/{args.num_samples}] Processing: {video_name}")
        print("  Extracting optical flows...")
        print(pixel_values.shape) # torch.Size([1, 25, 3, 320, 512])
        print(pixel_values.dtype) # torch.float16
        print(weight_dtype) # torch.float16
        print(pixel_values.float().dtype) 
        flows = get_optical_flows(unimatch, pixel_values)  # [1, T-1, 2, H, W]
        
        # Prepare first frame
        first_frame_tensor = pixel_values[0, 0]  # [C, H, W]
        first_frame_pil = Image.fromarray(
            (first_frame_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        )
        
        # Generate video
        print("  Generating video...")
        with torch.autocast("cuda", dtype=weight_dtype):
            output = pipeline(
                first_frame_pil,
                first_frame_pil,
                flows,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                decode_chunk_size=8,
                motion_bucket_id=127,
                fps=7,
                noise_aug_strength=0.02,
                controlnet_cond_scale=args.controlnet_scale,
                generator=torch.Generator(device).manual_seed(42),
            )
            print("Return log prob:")
            print(output.all_log_probs)  # [num_steps, 1, 1]
        
        video_frames = output.frames[0]
        
        # Convert to numpy
        video_frames_np = np.stack([np.array(frame) for frame in video_frames])
        
        # Visualize flows
        print("  Visualizing flows...")
        viz_flows = []
        for i in range(flows.shape[1]):
            temp_flow = flows[0, i].permute(1, 2, 0).cpu()
            viz_flows.append(flow_to_image(temp_flow))
        viz_flows = [np.uint8(np.ones_like(viz_flows[0]) * 255)] + viz_flows
        viz_flows = np.stack(viz_flows)
        
        # Get ground truth
        gt_frames = (pixel_values[0].permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        
        # Create separate folders for each output type
        gt_dir = os.path.join(args.output_dir, "gt")
        flow_dir = os.path.join(args.output_dir, "flow")
        generated_dir = os.path.join(args.output_dir, "generated")
        os.makedirs(gt_dir, exist_ok=True)
        os.makedirs(flow_dir, exist_ok=True)
        os.makedirs(generated_dir, exist_ok=True)
        
        # Save ground truth
        gt_path = os.path.join(gt_dir, f"{idx:03d}_{video_name}.mp4")
        torchvision.io.write_video(
            gt_path,
            gt_frames,
            fps=8,
            video_codec='h264',
            options={'crf': '10'}
        )
        
        # Save flow visualizations
        flow_path = os.path.join(flow_dir, f"{idx:03d}_{video_name}.mp4")
        torchvision.io.write_video(
            flow_path,
            viz_flows,
            fps=8,
            video_codec='h264',
            options={'crf': '10'}
        )
        
        # Save generated video
        generated_path = os.path.join(generated_dir, f"{idx:03d}_{video_name}.mp4")
        torchvision.io.write_video(
            generated_path,
            video_frames_np,
            fps=8,
            video_codec='h264',
            options={'crf': '10'}
        )
        
        print(f"  Saving to: {args.output_dir}/{{gt,flow,generated}}")
        
        print(f"  ✓ Saved: {video_name}")
    
    print("\n" + "=" * 60)
    print(f"✓ Generation complete! Videos saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
   

    # with log prob
    # python test_generation_with_logprob.py --controlnet_path ./ckpts/controlnet --num_samples 5 --output_dir ./output_author_checkpoints --scheduler ddim --return_log_prob