"""
EfficientVAD Training Pipeline
Lightweight CLIP-adapted video anomaly detection.

Usage:
  python train_efficientvad.py --dataset ucf --feat_dir /path/to/clip_features
  python train_efficientvad.py --dataset sht --feat_dir /path/to/clip_features
  python train_efficientvad.py --dataset xd  --feat_dir /path/to/clip_features

Features should be pre-extracted CLIP ViT-B/16 features stored as .npy files,
each of shape (variable_T, 512). They are resampled to 32 segments at load time.

Expected directory structure per dataset:
  feat_dir/
    train_list.txt        — lines: "relative/path.npy label"
    test_list.txt
    features/             — .npy feature files
    test_frame_mask/      — (optional) per-video frame-level GT masks
"""

import argparse
import os
import time
import json
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

from src.models.efficientvad import EfficientVAD
from src.utils.losses import EfficientVADLoss
from src.datasets.dataset import GenericFeatureDataset, MILPairDataset


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def estimate_flops(model, input_shape=(1, 32, 512)):
    """Rough FLOPs estimate for a forward pass."""
    flops = 0
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            flops += 2 * m.in_features * m.out_features
            if m.bias is not None:
                flops += m.out_features
        elif isinstance(m, torch.nn.Conv1d):
            k = m.kernel_size[0]
            c_in = m.in_channels // m.groups
            flops += 2 * k * c_in * m.out_channels * input_shape[1]
    return flops


@torch.no_grad()
def evaluate(model, test_dataset, device, dataset_name="ucf"):
    """Frame-level evaluation: AUC for UCF/SHT, AP for XD-Violence."""
    model.eval()
    loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    all_frame_scores = []
    all_frame_labels = []
    video_results = {}

    for feats, label, vid_id in loader:
        feats = feats.to(device)
        scores, _ = model(feats)
        scores = scores[0].cpu().numpy()
        vid_name = vid_id[0] if isinstance(vid_id, (list, tuple)) else vid_id

        video_results[vid_name] = {
            'scores': scores,
            'label': label.item(),
        }

        mask_path = None
        if hasattr(test_dataset, 'root'):
            mask_path = os.path.join(test_dataset.root, 'test_frame_mask', f'{vid_name}.npy')

        if mask_path and os.path.exists(mask_path):
            frame_mask = np.load(mask_path)
            n_frames = len(frame_mask)
            frame_scores = np.repeat(scores, max(1, n_frames // len(scores) + 1))[:n_frames]
            if len(frame_scores) < n_frames:
                pad = np.full(n_frames - len(frame_scores), scores[-1])
                frame_scores = np.concatenate([frame_scores, pad])
            all_frame_scores.extend(frame_scores)
            all_frame_labels.extend(frame_mask)

    results = {}
    if all_frame_scores:
        fs = np.array(all_frame_scores)
        fl = np.array(all_frame_labels).astype(int)
        if len(np.unique(fl)) > 1:
            results['frame_auc'] = roc_auc_score(fl, fs) * 100
            results['frame_ap'] = average_precision_score(fl, fs) * 100

    # Video-level metrics
    vid_scores = np.array([v['scores'].max() for v in video_results.values()])
    vid_labels = np.array([v['label'] for v in video_results.values()])
    if len(np.unique(vid_labels)) > 1:
        results['video_auc'] = roc_auc_score(vid_labels, vid_scores) * 100
        results['video_ap'] = average_precision_score(vid_labels, vid_scores) * 100

    return results


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    loss_details = {}

    for anom_feat, norm_feat in loader:
        anom_feat = anom_feat.to(device)
        norm_feat = norm_feat.to(device)

        optimizer.zero_grad()
        anom_scores, anom_mags = model(anom_feat)
        norm_scores, norm_mags = model(norm_feat)

        loss, details = criterion(anom_scores, norm_scores, anom_mags, norm_mags)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        for k, v in details.items():
            loss_details[k] = loss_details.get(k, 0) + v

    n = max(1, len(loader))
    avg_loss = total_loss / n
    avg_details = {k: v / n for k, v in loss_details.items()}
    return avg_loss, avg_details


def main():
    parser = argparse.ArgumentParser(description="Train EfficientVAD")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["ucf", "sht", "xd"],
                        help="Dataset: ucf (UCF-Crime), sht (ShanghaiTech), xd (XD-Violence)")
    parser.add_argument("--feat_dir", type=str, required=True,
                        help="Root directory containing features and list files")
    parser.add_argument("--train_list", type=str, default=None,
                        help="Path to train list file (default: feat_dir/train_list.txt)")
    parser.add_argument("--test_list", type=str, default=None,
                        help="Path to test list file (default: feat_dir/test_list.txt)")

    parser.add_argument("--feat_dim", type=int, default=512)
    parser.add_argument("--bottleneck_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--se_reduction", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.6)
    parser.add_argument("--num_segments", type=int, default=32)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--mag_margin", type=float, default=100.0)
    parser.add_argument("--mag_weight", type=float, default=1e-4)
    parser.add_argument("--smooth_weight", type=float, default=8e-5)
    parser.add_argument("--sparse_weight", type=float, default=8e-5)

    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_list = args.train_list or os.path.join(args.feat_dir, "train_list.txt")
    test_list = args.test_list or os.path.join(args.feat_dir, "test_list.txt")

    train_ds = GenericFeatureDataset(train_list, args.num_segments, args.feat_dir)
    test_ds = GenericFeatureDataset(test_list, args.num_segments, args.feat_dir)
    train_pairs = MILPairDataset(train_ds)
    train_loader = DataLoader(train_pairs, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)

    model = EfficientVAD(
        feat_dim=args.feat_dim,
        bottleneck_dim=args.bottleneck_dim,
        se_reduction=args.se_reduction,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    total_params, trainable_params = count_parameters(model)
    flops = estimate_flops(model, (1, args.num_segments, args.feat_dim))
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")
    print(f"Estimated FLOPs per sample: {flops:,}")
    print(f"Model size: {total_params * 4 / 1024:.1f} KB (float32)")

    criterion = EfficientVADLoss(
        mag_margin=args.mag_margin,
        mag_weight=args.mag_weight,
        smooth_weight=args.smooth_weight,
        sparse_weight=args.sparse_weight,
        top_k=args.top_k,
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(args.epochs * 0.6), int(args.epochs * 0.8)],
        gamma=0.1,
    )

    os.makedirs(args.save_dir, exist_ok=True)
    best_metric = 0
    best_epoch = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        avg_loss, details = train_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        log = f"Epoch {epoch:3d}/{args.epochs} | Loss: {avg_loss:.4f} | " \
              f"rank: {details['rank']:.4f} mag: {details['mag']:.1f} | {elapsed:.1f}s"
        print(log)

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            results = evaluate(model, test_ds, device, args.dataset)
            primary = results.get('frame_auc', results.get('video_auc', 0))
            if args.dataset == 'xd':
                primary = results.get('frame_ap', results.get('video_ap', 0))

            res_str = " | ".join(f"{k}: {v:.2f}" for k, v in results.items())
            print(f"  Eval: {res_str}")

            if primary > best_metric:
                best_metric = primary
                best_epoch = epoch
                ckpt_path = os.path.join(args.save_dir, f"efficientvad_{args.dataset}_best.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'results': results,
                    'args': vars(args),
                }, ckpt_path)
                print(f"  Best model saved (metric={primary:.2f})")

            history.append({'epoch': epoch, 'loss': avg_loss, **results})

    # Final summary
    print(f"\nBest epoch: {best_epoch} | Best metric: {best_metric:.2f}")
    print(f"Total params: {total_params:,} ({total_params/1e6:.3f}M)")
    print(f"Estimated FLOPs: {flops:,} ({flops/1e6:.2f}M)")

    # Inference speed benchmark
    model.eval()
    dummy = torch.randn(1, args.num_segments, args.feat_dim).to(device)
    with torch.no_grad():
        for _ in range(10):
            model(dummy)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(1000):
            model(dummy)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latency_ms = (time.time() - t0) / 1000 * 1000
    print(f"Inference latency: {latency_ms:.2f} ms/sample")

    summary = {
        'dataset': args.dataset,
        'best_epoch': best_epoch,
        'best_metric': best_metric,
        'total_params': total_params,
        'flops': flops,
        'latency_ms': latency_ms,
        'history': history,
        'args': vars(args),
    }
    with open(os.path.join(args.save_dir, f"efficientvad_{args.dataset}_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
