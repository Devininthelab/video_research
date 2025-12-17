import lpips
import torch
import numpy as np
import logging


class LpipsRewardFunction:
    def __init__(self, device):
        self.device = device
        self.lpips_model = lpips.LPIPS(net='alex').to(device)
        self.lpips_model.eval()

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