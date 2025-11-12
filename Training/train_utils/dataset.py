import os, io, csv, math, random
import numpy as np
from einops import rearrange
import pandas as pd

import torch
from decord import VideoReader, cpu
import torch.distributed as dist

import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from PIL import Image


def pil_image_to_numpy(image):
    """Convert a PIL image to a NumPy array."""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return np.array(image)


def numpy_to_pt(images: np.ndarray) -> torch.FloatTensor:
    """Convert a NumPy image to a PyTorch tensor."""
    if images.ndim == 3:
        images = images[..., None]
    images = torch.from_numpy(images.transpose(0, 3, 1, 2))
    return images.float() / 255


def zero_rank_print(s):
    if (not dist.is_initialized()) and (dist.is_initialized() and dist.get_rank() == 0): print("### " + s)


class WebVid10M(Dataset):
    def __init__(
            self,
            meta_path='/apdcephfs/share_1290939/0_public_datasets/WebVid/metadata/results_2M_train.csv',
            data_dir='/apdcephfs/share_1290939/0_public_datasets/WebVid',
            sample_size=[256, 256], # target frame resloution
            sample_stride=1, # frame sampling interval (=1 means consecutive frames)
            sample_n_frames=14, # number of frames to extract per video clip
        ):
        zero_rank_print(f"loading annotations from {meta_path} ...")

        metadata = pd.read_csv(meta_path)
        metadata['caption'] = metadata['name']
        del metadata['name']
        self.metadata = metadata
        self.metadata.dropna(inplace=True)
        self.data_dir = data_dir

        self.length = len(self.metadata)
        print(f"data scale: {self.length}")

        self.sample_stride   = sample_stride
        print(f"sample stride: {self.sample_stride}")
        self.sample_n_frames = sample_n_frames
        
        # sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        print("sample size", sample_size)
        self.sample_size = sample_size

        self.pixel_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            # transforms.Resize(sample_size),
            # transforms.CenterCrop(sample_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
    
    def _get_video_path(self, sample):
        rel_video_fp = os.path.join(sample['page_dir'], str(sample['videoid']) + '.mp4')
        # Check if data_dir already contains 'videos' subdirectory structure
        # For original WebVid: data_dir = /path/to/WebVid, videos in /path/to/WebVid/videos/
        # For reduced WebVid: data_dir = /path/to/reduced_WebVid, videos in /path/to/reduced_WebVid/
        if os.path.exists(os.path.join(self.data_dir, 'videos')):
            full_video_fp = os.path.join(self.data_dir, 'videos', rel_video_fp)
        else:
            full_video_fp = os.path.join(self.data_dir, rel_video_fp)
        return full_video_fp, rel_video_fp
    
    def get_batch(self, index):

        while True: # this is just for skip failed video

            index = index % len(self.metadata)
            sample = self.metadata.iloc[index]
            video_path, rel_path = self._get_video_path(sample)

            required_frame_num = self.sample_stride * self.sample_n_frames

            try:
                video_reader = VideoReader(video_path, ctx=cpu(0))
                if len(video_reader) < required_frame_num:
                    index += 1
                    continue
                else:
                    pass
            except:
                index += 1
                print(f"Load video failed! path = {video_path}")
                continue
            
            frame_num = len(video_reader)

            ## select a random clip
            random_range = frame_num - required_frame_num
            start_idx = random.randint(0, random_range) if random_range > 0 else 0
            frame_indices = [start_idx + self.sample_stride*i for i in range(self.sample_n_frames)]

            try:
                frames = video_reader.get_batch(frame_indices)
                break
            except:
                print(f"Get frames failed! path = {video_path}; [max_ind vs frame_total:{max(frame_indices)} / {frame_num}]")
                index += 1
                continue

        # validation if video has enough frames (required_frame_num = stride × n_frames)
        assert(frames.shape[0] == self.sample_n_frames),f'{len(frames)}, self.video_length={self.sample_n_frames}'

        frames = frames.asnumpy()
        # resize to 256 x 256
        resized_frames = []
        for i in range(frames.shape[0]):
            frame = np.array(Image.fromarray(frames[i]).convert('RGB').resize([self.sample_size[1], self.sample_size[0]]))
            resized_frames.append(frame)
        resized_frames = np.array(resized_frames)

        resized_frames = torch.tensor(resized_frames).permute(0, 3, 1, 2).float()  # [t,h,w,c] -> [t,c,h,w]

        return resized_frames, rel_path
        
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        pixel_values, video_name = self.get_batch(idx)

        # pixel_values = self.pixel_transforms(pixel_values)
        pixel_values = pixel_values / 255. # to [0,1]
        
        sample = dict(pixel_values=pixel_values, video_name=video_name)
        return sample

