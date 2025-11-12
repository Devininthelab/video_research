from scipy.stats import wasserstein_distance
import cv2
import numpy as np
import torch
import os
from skimage.metrics import structural_similarity as ssim
import multiprocessing as mp
from scipy.spatial.distance import euclidean
import structlog
import argparse
import json
from tqdm import tqdm
logger = structlog.get_logger()


def analyze_visual_quality(
    video_path: str,
    sample_rate: int = 5
):
    """
    Analyze the visual quality of a video by calculating sharpness.

    This function uses the variance of the Laplacian operator as a proxy for
    sharpness. A higher score generally indicates a sharper, more in-focus frame.
    Blurry or artifact-heavy frames (like that "ghost faces") will have a low score.

    Args:
        video_path (str): the path to the video file
        sample_rate (int): process every N-th frame to speed up analysis.
    
    Returns:
        float: the average sharpness score across the sampled frames, or -1.0 if
                video cannot be opened.
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Error: could not open video file {video_path}")
        return -1.0
    
    sharpness_scores = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % sample_rate == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_scores.append(lap_var)
        
        frame_count += 1
    
    cap.release()
    if not sharpness_scores:
        return 0.0
    
    return np.mean(sharpness_scores)


def analyze_motion_quality(
    video_path: str,
    sample_rate: int = 2
):
    """
    Analyze motion quality using dense optical flow (Farneback method)

    This function calculates the average magnitude of motion vectors between
    frames. Smooth, natural motion will have a moderate magnitude flow.
    Jerky, erratic, or artifact-filed motion will result in unusually high
    average flow.

    Args:
        video_path (str): the path to the video file.
        sample_rate (int): process every N-th frame to speed up analysis.

    Returns:
        float: the average magnitude of optical flow, or -1.0 on error.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Error: could not open video file {video_path}")
        return -1.0

    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return 0.0
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    flow_magnitudes = []
    frame_count = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % sample_rate == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray,
                gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )
            
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            flow_magnitudes.append(np.mean(magnitude))
            prev_gray = gray
        
        frame_count += 1
    
    cap.release()
    if not flow_magnitudes:
        return 0.0
    
    return np.mean(flow_magnitudes)


def analyze_temporal_consistency(
    video_path: str,
    sample_rate: int = 2
):
    """
    Analyzes temporal consistency using frame-to-frame SSIM.

    Structural similarity index (SSIM) measures the perceptual similarity
    between two images. A score close to 1.0 means they are very similar.
    This function finds the minimum SSIM between consecutive frames, which
    pinpoints the most significant glitch, flicker, or scene jump.

    Args:
        video_path (str): the path to the video file
        sample_rate (int): process every N-th frame
    
    Returns:
        float: the minimum SSIM score found, or -1.0 on error
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Error: could not open video file {video_path}")
        return -1.0

    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return 1.0 # no frames to compare, so no inconsistency
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    min_ssim_score = 1.0
    frame_count = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break
    
        if frame_count % sample_rate == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            score = ssim(prev_gray, gray, data_range=gray.max() - gray.min())
            if score < min_ssim_score:
                min_ssim_score = score
            prev_gray = gray
        
        frame_count += 1
    
    cap.release()
    return min_ssim_score



def analyze_video(
    video_path
):
    """
    Main function to run all video analyses and return results

    Args:
        video_path (str): the path to the video file to analyze

    Returns:
        dict: a dictionary containing the computed scores
    """

    visual_quality_score = analyze_visual_quality(video_path)
    motion_quality_score = analyze_motion_quality(video_path)
    temporal_consistency_score = analyze_temporal_consistency(video_path)
    results = {
        'visual_quality': visual_quality_score,
        'motion_quality': motion_quality_score,
        'temporal_consistency': temporal_consistency_score
    }
    return results


def evaluate_motion_comprehensive(
    index,
    real_video,
    generated_video
):
    """
    Comprehensive motion evaluation between two videos.

    Returns:
        Dictionary with multiple motion metrics
    """
    logger.info(f"Start evaluating sample {index}")
    analyzed_results = analyze_video(generated_video)

    results = {
        'index': index,
        'generated_video': generated_video,
        'real_video': real_video,
        'visual_quality': analyzed_results['visual_quality'],
        'motion_quality': analyzed_results['motion_quality'].item(),
        'temporal_consistency': analyzed_results['temporal_consistency'].item(),
    }

    if use_consecutive_flow:
        with open(f"evals_stage1-use_consecutive_flow/{output_id}/sample_{index:04d}.json", 'w') as f:
            json.dump(results, f, indent=4)
    else:
        with open(f"evals_stage1/{output_id}/sample_{index:04d}.json", 'w') as f:
            json.dump(results, f, indent=4)

    logger.info(f"Finish evaluating sample {index}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, default='/home/thong/weride_project/MOFA-Video/Training/outputs_stage1/step_12500_val_img')
    args = parser.parse_args()
    output_dir = args.output_dir
    output_id = output_dir.split('/')[-1]
    use_consecutive_flow = 'use_consecutive_flow' in output_dir

    generated_video_folder_path = os.path.join(output_dir, 'generated_videos')
    real_video_folder_path = os.path.join(output_dir, 'real_videos')

    video_file_list = sorted(os.listdir(generated_video_folder_path), key=lambda x: int(x.split('-')[0]))[:1000]
    motion_mag_diff_list = []
    flow_emd_list = []
    direction_similarity_list = []
    trajectory_distance_list = []
    if use_consecutive_flow:
        eval_dir = f"evals_stage1-use_consecutive_flow/{output_id}"
    else: 
        eval_dir = f"evals_stage1/{output_id}"
    os.makedirs(eval_dir, exist_ok=True)
    evaled_file_list = os.listdir(eval_dir)
    evaled_index_set = set()
    real_video_file_path_list = [os.path.join(real_video_folder_path, file) for file in video_file_list]
    generated_video_file_path_list = [os.path.join(generated_video_folder_path, file) for file in video_file_list]
    for evaled_file in tqdm(evaled_file_list):
        evaled_file_path = os.path.join(eval_dir, evaled_file)
        with open(evaled_file_path, 'r') as f:
            evals = json.load(f)
        evaled_index_set.add(evals['index'])
    to_eval_index_set = set([i for i in range(len(real_video_file_path_list)) if i not in evaled_index_set])

    evaluate_pool = mp.Pool(processes=16)
    evaluate_pool.starmap(evaluate_motion_comprehensive, [(i, real_video_file_path_list[i], generated_video_file_path_list[i]) for i in to_eval_index_set])
    evaluate_pool.close()
    evaluate_pool.join()