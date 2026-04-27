import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from collections import defaultdict
from sklearn.metrics import roc_auc_score, average_precision_score

def per_category_analysis(per_video_results):
    """Compute per-crime-category detection accuracy."""
    cat_scores = defaultdict(list)
    cat_labels = defaultdict(list)
    
    for vid_id, info in per_video_results.items():
        if "/" in vid_id:
            category = vid_id.split("/")[0]
        else:
            category = "Normal" if info["label"] == 0 else "Abnormal"
        cat_scores[category].append(info["max_score"])
        cat_labels[category].append(info["label"])
    
    results = {}
    for cat in sorted(cat_scores.keys()):
        scores = np.array(cat_scores[cat])
        labels = np.array(cat_labels[cat])
        mean_s = scores.mean()
        det_rate = (scores > 0.5).mean() if labels.mean() > 0 else (scores <= 0.5).mean()
        results[cat] = {"mean_score": mean_s, "detect_rate": det_rate, "count": len(scores)}
    return results

def temporal_localization_eval(model, dataset, root, device):
    """Evaluate segment-level scores against per-frame GT masks."""
    mask_dir = os.path.join(root, "testing", "test_frame_mask")
    if not os.path.isdir(mask_dir):
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
        
        feats = feats.to(device)
        with torch.no_grad():
            scores = model(feats)[0].cpu().numpy()
        
        frame_mask = np.load(mask_path)
        n_frames = len(frame_mask)
        frame_scores = np.repeat(scores, max(1, n_frames // 32 + 1))[:n_frames]
        
        all_frame_scores.extend(frame_scores)
        all_frame_labels.extend(frame_mask)
    
    if not all_frame_scores: return None
    
    frame_auc = roc_auc_score(all_frame_labels, all_frame_scores)
    frame_ap = average_precision_score(all_frame_labels, all_frame_scores)
    return {"auc": frame_auc, "ap": frame_ap}
