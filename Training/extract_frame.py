import cv2
from PIL import Image
import numpy as np

def video_to_horizontal_strip(video_path, output_path, num_frames=6):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Could not open video")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Chọn index của 6 frames, lấy đều từ đầu tới cuối
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))

    cap.release()

    if len(frames) == 0:
        raise ValueError("No frames extracted")

    # --- Stack theo chiều ngang ---
    widths = [f.width for f in frames]
    heights = [f.height for f in frames]

    strip_width = sum(widths)
    strip_height = max(heights)

    strip = Image.new("RGB", (strip_width, strip_height))

    x_offset = 0
    for f in frames:
        strip.paste(f, (x_offset, 0))
        x_offset += f.width

    strip.save(output_path)
    print(f"Saved horizontal strip to: {output_path}")

video_to_horizontal_strip("/home/minhthan001/Video/video_research/Training/2days_rl_videos-use_consecutive_flow-reduced_dataset/generated_videos/video_00-0.6023.mp4", "output_horizontal_strip.png", num_frames=6)