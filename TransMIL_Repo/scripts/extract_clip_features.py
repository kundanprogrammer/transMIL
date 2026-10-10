"""
Extract CLIP ViT-B/16 features from video frames for VAD datasets.

Usage:
  python scripts/extract_clip_features.py \
      --video_dir /path/to/videos \
      --output_dir /path/to/features \
      --fps 1

Processes each video at the specified FPS, extracts the CLIP [CLS] token
per frame, and saves as .npy files of shape (num_frames, 512).
Use 10-crop augmentation (--ten_crop) for better features.
"""

import argparse
import os
import glob
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

try:
    import clip
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def extract_frames(video_path, fps=1):
    """Extract frames from video at given FPS."""
    if not HAS_CV2:
        raise ImportError("OpenCV required: pip install opencv-python")

    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30.0
    interval = max(1, int(video_fps / fps))

    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
        idx += 1
    cap.release()
    return frames


def ten_crop_transform(image, size=224):
    """Generate 10 crops: 5 spatial + 5 flipped (standard VAD augmentation)."""
    w, h = image.size
    crops = []

    positions = [
        (0, 0), (w - size, 0), (0, h - size), (w - size, h - size),
        ((w - size) // 2, (h - size) // 2),
    ]

    resized = image.resize((max(size, w), max(size, h)))
    w, h = resized.size

    positions = [
        (0, 0), (w - size, 0), (0, h - size), (w - size, h - size),
        ((w - size) // 2, (h - size) // 2),
    ]

    for x, y in positions:
        crop = resized.crop((x, y, x + size, y + size))
        crops.append(crop)
        crops.append(crop.transpose(Image.FLIP_LEFT_RIGHT))

    return crops


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", required=True, help="Directory of videos")
    parser.add_argument("--output_dir", required=True, help="Output directory for features")
    parser.add_argument("--fps", type=int, default=1, help="Frames per second to extract")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--ten_crop", action="store_true",
                        help="Use 10-crop augmentation (5 spatial + 5 flipped)")
    parser.add_argument("--extensions", type=str, default="mp4,avi,mkv,mov",
                        help="Video extensions to process")
    args = parser.parse_args()

    if not HAS_CLIP:
        print("Install CLIP: pip install git+https://github.com/openai/CLIP.git")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/16", device=device)
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    exts = args.extensions.split(",")
    videos = []
    for ext in exts:
        videos.extend(glob.glob(os.path.join(args.video_dir, f"**/*.{ext}"), recursive=True))
    videos.sort()
    print(f"Found {len(videos)} videos")

    for i, vpath in enumerate(videos):
        rel = os.path.relpath(vpath, args.video_dir)
        out_path = os.path.join(args.output_dir, os.path.splitext(rel)[0] + ".npy")

        if os.path.exists(out_path):
            continue

        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        frames = extract_frames(vpath, args.fps)
        if not frames:
            print(f"  [{i+1}/{len(videos)}] {rel}: no frames extracted, skipping")
            continue

        all_features = []
        for start in range(0, len(frames), args.batch_size):
            batch_frames = frames[start:start + args.batch_size]

            if args.ten_crop:
                batch_imgs = []
                for frame in batch_frames:
                    crops = ten_crop_transform(frame)
                    batch_imgs.extend([preprocess(c) for c in crops])
            else:
                batch_imgs = [preprocess(f) for f in batch_frames]

            batch_tensor = torch.stack(batch_imgs).to(device)
            with torch.no_grad():
                features = model.encode_image(batch_tensor).float()

            if args.ten_crop:
                features = features.reshape(len(batch_frames), 10, -1).mean(dim=1)

            features = features / features.norm(dim=-1, keepdim=True)
            all_features.append(features.cpu().numpy())

        features = np.concatenate(all_features, axis=0)
        np.save(out_path, features)

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{len(videos)}] {rel}: {features.shape}")

    print(f"Done. Features saved to {args.output_dir}")


if __name__ == "__main__":
    main()
