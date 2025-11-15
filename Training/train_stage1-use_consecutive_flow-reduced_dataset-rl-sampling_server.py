import argparse
import logging
import os
import pickle
import cv2
import socket
import struct
import threading
from queue import Queue, Empty
import numpy as np
import shutil
import lpips
import torch
import time
import zlib
import torchvision
from PIL import Image
from tqdm import tqdm, trange

from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler
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
        logger.info(f"Progress: sent {offset}/{total_size}")
    
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


class EpipolarRewardFunction:
    def __init__(
        self,
        device,
        use_sift = True,
        min_matches = 8,
        ransac_threshold = 1.0
    ):  
        # This only works iff the video is a static scene. If the video has moving object, epipolar reward function get noise
        self.device = device
        self.use_sift = use_sift
        self.min_matches = min_matches
        self.ransac_threshold = ransac_threshold

        if use_sift:
            self.detector = cv2.SIFT_create()
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            self.detector = cv2.ORB_create(nfeatures=2000)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        logger.info(f"Epipolar reward initialized (SIFT={use_sift})")

    
    def detect_and_match(
        self,
        img1,
        img2
    ):
        kp1, des1 = self.detector.detectAndCompute(img1, None)
        kp2, des2 = self.detector.detectAndCompute(img2, None)

        if des1 is None or des2 is None or len(kp1) < self.min_matches:
            return None, None
        
        matches = self.matcher.knnMatch(des1, des2, k=2)
        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < self.min_matches:
            return None, None
        
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
        return pts1, pts2

    
    def compute_sampson_error(
        self,
        pts1,
        pts2,
        F
    ):
        """
        Compute Sampson epipolar error (Eq. 4 from paper)
        S_E = (x'^T F x)^2 / ((Fx)_1^2 + (Fx)_2^2 + (F^T x')_1^2 + (F^T x')_2^2)
        """
        # homogenous coordinates
        pts1_h = np.concatenate([pts1, np.ones((pts1.shape[0], 1))], axis=1)
        pts2_h = np.concatenate([pts2, np.ones((pts2.shape[0], 1))], axis=1)

        # epipolar constraint
        Fx = (F @ pts1_h.T).T
        Ftx = (F.T @ pts2_h.T).T
        x_Fx = np.sum(pts2_h * Fx, axis=1)

        # denominator
        Fx_sq = Fx[:, 0]**2 + Fx[:, 1]**2
        Ftx_sq = Ftx[:, 0]**2 + Ftx[:, 1]**2
        denominator = np.maximum(Fx_sq + Ftx_sq, 1e-8)
        sampson_errors = (x_Fx ** 2)/denominator
        return np.mean(sampson_errors)


    def compute_frame_pair_error(self, frame1, frame2):
        """
        Compute epipolar error between two frames
        """
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)

        pts1, pts2 = self.detect_and_match(gray1, gray2)

        if pts1 is None or len(pts1) < self.min_matches:
            return None
        
        # estimate fundamental matrix
        F, mask = cv2.findFundamentalMat(
            pts1, pts2, cv2.FM_RANSAC,
            ransacReprojThreshold=self.ransac_threshold,
            confidence=0.999, maxIters=2000
        )

        if F is None or F.shape != (3, 3):
            return None
        
        # use only inliers
        inlier_mask = mask.ravel() == 1
        pts1_inliers = pts1[inlier_mask]
        pts2_inliers = pts2[inlier_mask]

        if len(pts1_inliers) < self.min_matches:
            return None
        
        return self.compute_sampson_error(pts1_inliers, pts2_inliers, F)



    def compute_video_consistency(
        self,
        frames,
        sample_stride = 4
    ):
        """Compute 3D consistency across video"""
        errors = []
        num_frames = len(frames)

        # sample frame pairs with stride
        for i in range(0, num_frames - sample_stride, sample_stride):
            j = min(i + sample_stride, num_frames - 1)
            error = self.compute_frame_pair_error(frames[i], frames[j])

            if error is not None:
                errors.append(error)
        
        if len(errors) == 0:
            logger.warning("No valid epipolar errors")
            return 1.0
        
        return np.mean(errors)


    
    def __call__(
        self,
        frames,
        gt_frames = None,
        prompts = None
    ):
        """
        compute reward from 3D geometric consistency
        lower sampson error = higher reward
        """
        sampson_error = self.compute_video_consistency(frames, sample_stride=4)
        sampson_error = np.clip(sampson_error, 0.0, 2.0)
        reward = -sampson_error
        logger.info(f"Sampson: {sampson_error:.4f}, Reward: {reward:.4f}")
        return reward




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


class SamplingServer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda')
        self.setup_models()
        self.setup_dataloader()
        if self.args.reward_fn == 'lpips':
            self.reward_fn = LpipsRewardFunction(self.device)
        elif self.args.reward_fn == 'epipolar':
            self.reward_fn = EpipolarRewardFunction(self.device)
        else:
            raise ValueError(f"Not support reward function {self.args.reward_fn}")

        # queue for storing generated samples
        self.sample_queue = Queue(maxsize=args.queue_size)
        self.stop_sampling = threading.Event()
        self.weights_lock = threading.Lock()

        # statistics
        self.total_samples_generated = 0
        self.epoch_counter = 0


    def setup_models(self):
        logger.info("Loading models")
        weight_dtype = torch.float16

        self.feature_extractor = CLIPImageProcessor.from_pretrained(
            self.args.pretrained_model_name_or_path, subfolder='feature_extractor'
        )
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            self.args.pretrained_model_name_or_path, subfolder='image_encoder', variant='fp16'
        )
        self.vae = AutoencoderKLTemporalDecoder.from_pretrained(
            self.args.pretrained_model_name_or_path, subfolder='vae', variant='fp16'
        )
        self.unet = UNetSpatioTemporalConditionControlNetModel.from_pretrained(
            self.args.pretrained_model_name_or_path, subfolder='unet', variant='fp16'
        )
        self.controlnet = FlowControlNet.from_pretrained(self.args.controlnet_model_name_or_path)

        self.vae.requires_grad_(False)
        self.image_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.controlnet.requires_grad_(False)

        self.unimatch = UniMatch(feature_channels=128,
            num_scales=2,
            upsample_factor=4,
            num_head=1,
            ffn_dim_expansion=4,
            num_transformer_layers=6,
            reg_refine=True,
            task='flow').to('cuda')
        
        self.checkpoint = torch.load('./train_utils/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth')
        self.unimatch.load_state_dict(self.checkpoint['model'])
        self.unimatch.eval()
        self.unimatch.requires_grad_(False)

        # Keep Unimatch in float32 for precision
        # self.unimatch.to(weight_dtype)
        self.vae.to(weight_dtype)
        self.image_encoder.to(weight_dtype)
        self.unet.to(weight_dtype)
        self.controlnet.to(weight_dtype)

        # Create pipeline
        self.pipeline = FlowControlNetPipeline.from_pretrained(
            self.args.pretrained_model_name_or_path,
            unet=self.unet,
            controlnet=self.controlnet,
            image_encoder=self.image_encoder,
            vae=self.vae,
            torch_dtype=weight_dtype
        )
        self.pipeline = self.pipeline.to(self.device)

    def setup_dataloader(self):
        train_dataset = WebVid10M(
            meta_path='/projects_vol/gp_slab/minhthan001/data_webvid_reduce/reduced_clean_results_2M_train.csv',
            data_dir="/projects_vol/gp_slab/minhthan001/data_webvid_reduce/reduced_WebVid",
            sample_stride=self.args.sample_stride,
            sample_n_frames=self.args.num_frames,
            sample_size=[self.args.height, self.args.width]
        )
        generator = torch.Generator()
        generator.manual_seed(self.args.seed)
        self.dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.args.sample_batch_size,
            shuffle=True,
            num_workers=4,
            generator=generator
        )
        self.dataloader_iter = iter(self.dataloader)


    def update_controlnet_weights(self, state_dict):
        logger.info("Updating controlnet weights")
        self.controlnet.load_state_dict(state_dict)
        self.controlnet.eval()

    
    def sample_batch(self):
        # Loads ground truth video from WebVid10M dataset
        # Extracts optical flow using Unimatch
        # Generates video using the pipeline with current ControlNet weights
        # Computes reward using selected reward function
        # Returns sample dict with pixel_vales, flows, reward
        try:
            batch = next(self.dataloader_iter)
        except StopIteration:
            self.dataloader_iter = iter(self.dataloader)
            batch = next(self.dataloader_iter)
        
        weight_dtype = torch.float16
        pixel_values = batch['pixel_values'].to(self.device)

        flows = get_optical_flows(self.unimatch, pixel_values)
        flows = flows.to(weight_dtype).to(self.device)
        pixel_values = pixel_values.to(weight_dtype)
        
        cond_image = pixel_values[:, 0, :, :, :]
        pil_cond_images = [Image.fromarray((cond_image[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)) for i in range(cond_image.shape[0])]

        with torch.no_grad():
            # Use autocast for fp16 to avoid type mismatches and improve performance
            with torch.autocast("cuda", dtype=weight_dtype):
                video_outputs = self.pipeline(
                pil_cond_images[0],
                pil_cond_images[0],
                flows,
                height=self.args.height,
                width=self.args.width,
                num_frames=self.args.num_frames,
                decode_chunk_size=8,
                motion_bucket_id=127,
                fps=7,
                noise_aug_strength=0.02,
                num_inference_steps=self.args.num_inference_steps,
                )
        
        generated_frames = video_outputs.frames[0]
        gt_frames = (pixel_values[0].permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        for i in range(self.args.num_frames):
            img = generated_frames[i]
            generated_frames[i] = np.array(img)
        
        reward = self.reward_fn(generated_frames, gt_frames=gt_frames)

        ################################################################################################################################################################################################################################################################
        # for debug 
        os.makedirs(f"rl_videos-use_consecutive_flow-reduced_dataset/groundtruth_videos", exist_ok=True)
        os.makedirs(f"rl_videos-use_consecutive_flow-reduced_dataset/generated_videos", exist_ok=True)
        torchvision.io.write_video(f"rl_videos-use_consecutive_flow-reduced_dataset/generated_videos/video_{self.total_samples_generated:02d}-{reward:.4f}.mp4", generated_frames, fps=8, video_codec='h264', options={'crf': '10'})
        torchvision.io.write_video(f"rl_videos-use_consecutive_flow-reduced_dataset/groundtruth_videos/video_{self.total_samples_generated:02d}.mp4", gt_frames, fps=8, video_codec='h264', options={'crf': '10'})
        ################################################################################################################################################################################################################################################################

        sample = {
            'pixel_values': pixel_values.cpu(),
            'flows': flows.cpu(),
            'reward': reward,
            'video_name': batch['video_name'][0],
        }
        return sample

    def continuous_sampling_worker(self):
        # Continuously generates samples
        # Adds them to a queue (default size is 64)
        # Runs independently of training requests
        # Run_server() socket server, listens on port 9999 and handles client request in sperate threads
        logger.info("Starting continuous sampling worker")

        while not self.stop_sampling.is_set():
            try:
                if self.sample_queue.qsize() >= self.args.queue_size:
                    logger.info(f"Queue full ({self.sample_queue.qsize()}/{self.args.queue_size})")
                    time.sleep(1)
                    continue
                
                # generate sample
                logger.info(f"Generating sample {self.total_samples_generated + 1} (queue: {self.sample_queue.qsize()}/{self.args.queue_size})")
                sample = self.sample_batch()

                # add to queue
                self.sample_queue.put(sample, timeout=5)
                self.total_samples_generated += 1
                logger.info(f"Sample added to queue. Total generated: {self.total_samples_generated}")

            except Exception as e:
                logger.error(f"Error in sampling worker: {e}")
                time.sleep(1)
        
        logger.info("Sampling worker stopped")


    def run_server(self, host='0.0.0.0', port=9999):
        """Run the sampling server"""
        sampling_thread = threading.Thread(target=self.continuous_sampling_worker, daemon=True)
        sampling_thread.start()
        logger.info("Background sampling thread started")

        # start server socket
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        server_socket.bind((host, port))
        server_socket.listen(5)
        logger.info(f"Sampling server listening on {host}:{port}")

        try:
            while True:
                client_socket, addr = server_socket.accept()
                logger.info(f"Connection from {addr}")
                client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 2 * 1024 * 1024)
                thread = threading.Thread(target=self.handle_client, args=(client_socket,))
                thread.start()
        except KeyboardInterrupt:
            logger.info("Shutting down server")
            self.stop_sampling.set()
            sampling_thread.join(timeout=5)

    
    def handle_client(self, client_socket):
        try:
            while True:
                # receive command
                cmd_length_data = client_socket.recv(4)
                if not cmd_length_data:
                    break
                cmd_length = struct.unpack("!I", cmd_length_data)[0]
                cmd = client_socket.recv(cmd_length).decode('utf-8')
                # if cmd == 'SAMPLE':
                #     # receive epoch number
                #     epoch = struct.unpack("!I", client_socket.recv(4))[0]
                #     num_batches = struct.unpack("!I", client_socket.recv(4))[0]

                #     logger.info(f"Sampling {num_batches} batches for epoch {epoch}")
                #     samples = []
                #     for i in trange(num_batches):
                #         logger.info(f"Generating sample {i+1}/{num_batches}")
                #         sample = self.sample_batch(epoch)
                #         samples.append(sample)
                    
                #     logger.info(f"Sending {len(samples)} samples with compression")
                #     send_large_data(client_socket, samples, compress=True)
                #     logger.info(f"Sent {len(samples)} samples successfully")
                if cmd == 'GET_SAMPLES':
                    num_samples = struct.unpack("!I", client_socket.recv(4))[0]
                    logger.info(f"Client requested {num_samples} (queue size: {self.sample_queue.qsize()})")
                    samples = []
                    timeout_per_sample = 60
                    for i in trange(num_samples):
                        try:
                            sample = self.sample_queue.get(timeout=timeout_per_sample)
                            samples.append(sample)
                            logger.info(f"Retrieved sample {i+1}/{num_samples} from queue")
                        except Empty:
                            logger.warning(f"Queue empty after waiting {timeout_per_sample}s, sending {len(samples)} samples")
                            break

                    if samples:
                        logger.info(f'Sending {len(samples)} samples to client')
                        send_large_data(client_socket, samples, compress=True)
                        logger.info(f"Samples sent successfully")
                    else:
                        logger.warning("No samples available, sending empty list")
                        send_large_data(client_socket, [], compress=True)

                elif cmd == 'UPDATE_WEIGHTS':
                    # receive new weights
                    logger.info("Receiving weight update")
                    state_dict = recv_large_data(client_socket)
                    self.update_controlnet_weights(state_dict)
                    client_socket.sendall(b'OK')
                    logger.info("Weights updated successfully")

                elif cmd == 'QUEUE_SIZE':
                    size = self.sample_queue.qsize()
                    client_socket.sendall(struct.pack('!I', size))
                    logger.info(f"Sent queue size: {size}")

                elif cmd == 'CLOSE':
                    break

        except Exception as e:
            logger.error(f"Error handling client: {e}")
        finally:
            client_socket.close()
    


def parse_args():
    parser = argparse.ArgumentParser(description="Sampling server for DDPO training")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--controlnet_model_name_or_path", type=str, required=True)
    parser.add_argument("--num_frames", type=int, default=25)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--sample_stride", type=int, default=1)
    parser.add_argument("--sample_batch_size", type=int, default=2, help="Batch size for sampling")
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--host", type=str, default='0.0.0.0')
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--reward_fn", type=str, default="lpips", 
                       choices=["lpips", "epipolar"])
    parser.add_argument("--queue_size", type=int, default=64)
    return parser.parse_args()
    


def main():
    if os.path.exists("rl_videos-use_consecutive_flow-reduced_dataset"):
        shutil.rmtree("rl_videos-use_consecutive_flow-reduced_dataset")
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )

    args = parse_args()
    server = SamplingServer(args)
    server.run_server(host=args.host, port=args.port)


if __name__ == '__main__':
    main()