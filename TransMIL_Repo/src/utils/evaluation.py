import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
import os

def evaluate_mil(model, dataset, gt_data, device, dataset_name="SHT"):
    """Evaluate video-level AUC using segment-level scores."""
    model.eval()
    
    all_scores = []
    all_labels = []
    per_video = {}
    
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    
    all_frame_scores = []
    all_frame_labels = []
    
    for feats, labels, vid_ids in loader:
        feats = feats.to(device)
        scores = model(feats)  # (B, 32)
        
        for i in range(len(vid_ids)):
            vid_id = vid_ids[i]
            vid_score = scores[i].cpu().numpy()    # (32,)
            vid_label = labels[i].item()
            
            # Video-level score = max segment score
            max_score = vid_score.max()
            
            per_video[vid_id] = {
                "scores": vid_score,
                "label": vid_label,
                "max_score": max_score,
            }
            
            all_scores.append(max_score)
            all_labels.append(vid_label)
            
            # Frame-level extraction
            f_mask = None
            
            if hasattr(dataset, "root"):
                mask_path = os.path.join(dataset.root, "testing", "test_frame_mask", f"{vid_id}.npy")
                if os.path.exists(mask_path):
                    f_mask = np.load(mask_path)
            
            if f_mask is None and gt_data is not None:
                match_k = None
                if vid_id in gt_data:
                    match_k = vid_id
                else:
                    base = os.path.basename(vid_id).split('.')[0]
                    for k in gt_data.keys():
                        if os.path.basename(str(k)).split('.')[0] == base:
                            match_k = k
                            break
                
                if match_k:
                    potential_mask = np.array(gt_data[match_k])
                    if potential_mask.ndim > 0 and len(potential_mask) > 1:
                        f_mask = potential_mask

            if f_mask is not None:
                n_frames = len(f_mask)
                if n_frames > 0:
                    # Interpolate 32 segment scores up to n_frames
                    f_scores = np.repeat(vid_score, max(1, n_frames // 32 + 1))[:n_frames]
                    if len(f_scores) < n_frames:
                        pad = np.ones(n_frames - len(f_scores)) * vid_score[-1]
                        f_scores = np.concatenate([f_scores, pad])
                    all_frame_scores.extend(f_scores)
                    all_frame_labels.extend(f_mask)
    
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    if len(np.unique(all_labels)) > 1:
        auc_roc = roc_auc_score(all_labels, all_scores)
        auc_pr = average_precision_score(all_labels, all_scores)
    else:
        auc_roc, auc_pr = 0.0, 0.0
        
    if len(all_frame_labels) > 0 and len(np.unique(all_frame_labels)) > 1:
        f_roc = roc_auc_score(all_frame_labels, all_frame_scores)
        f_pr = average_precision_score(all_frame_labels, all_frame_scores)
    else:
        f_roc, f_pr = 0.0, 0.0
    
    return auc_roc, auc_pr, f_roc, f_pr, all_scores, all_labels, per_video


# ── 11. Main Experiment Runner ────────────────────────────────────────────

def temporal_localization_eval(model, dataset, root, device):
    """
    Evaluate segment-level scores against per-frame GT masks.
    Maps 32-segment scores back to frame-level for comparison.
    """
    mask_dir = os.path.join(root, "testing", "test_frame_mask")
    if not os.path.isdir(mask_dir):
        print("⚠️  test_frame_mask not found, skipping temporal localization")
        return None
    
    model.eval()
    all_frame_scores = []
    all_frame_labels = []
    
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    for feats, labels, vid_ids in loader:
        vid_id = vid_ids[0]
        mask_path = os.path.join(mask_dir, f"{vid_id}.npy")
        if not os.path.exists(mask_path):
            continue
        
        # Get segment scores
        feats = feats.to(device)
        with torch.no_grad():
            scores = model(feats)[0].cpu().numpy()  # (32,)
        
        # Load frame-level GT mask
        frame_mask = np.load(mask_path)  # variable length binary array
        n_frames = len(frame_mask)
        
        # Upsample 32 segment scores → n_frames
        frame_scores = np.repeat(scores, max(1, n_frames // 32 + 1))[:n_frames]
        
        all_frame_scores.extend(frame_scores)
        all_frame_labels.extend(frame_mask)
    
    if not all_frame_scores:
        return None
    
    all_frame_scores = np.array(all_frame_scores)
    all_frame_labels = np.array(all_frame_labels).astype(int)
    
    if len(np.unique(all_frame_labels)) > 1:
        frame_auc = roc_auc_score(all_frame_labels, all_frame_scores)
        frame_ap = average_precision_score(all_frame_labels, all_frame_scores)
    else:
        frame_auc, frame_ap = 0.0, 0.0
    
    print(f"  Frame-level AUC-ROC: {frame_auc*100:.2f}%")
    print(f"  Frame-level AUC-PR:  {frame_ap*100:.2f}%")
    return frame_auc, frame_ap


# ── 21. Run Novel Experiments ─────────────────────────────────────────────