import torch
import time
import os
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

def train_mil(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for anom_feat, norm_feat in train_loader:
        anom_feat = anom_feat.to(device)
        norm_feat = norm_feat.to(device)
        
        anom_scores = model(anom_feat)
        norm_scores = model(norm_feat)
        
        loss, _, _, _ = criterion(anom_scores, norm_scores)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / max(1, len(train_loader))

@torch.no_grad()
def evaluate_mil(model, dataset, gt_data, device):
    model.eval()
    all_scores = []
    all_labels = []
    all_frame_scores = []
    all_frame_labels = []
    
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    for feats, labels, vid_ids in loader:
        feats = feats.to(device)
        scores = model(feats)
        
        for i in range(len(vid_ids)):
            vid_id = vid_ids[i]
            vid_score = scores[i].cpu().numpy()
            vid_label = labels[i].item()
            
            max_score = vid_score.max()
            all_scores.append(max_score)
            all_labels.append(vid_label)
            
            # Simplified frame-level evaluation if mask is available
            mask_path = os.path.join(dataset.root, "testing", "test_frame_mask", f"{vid_id}.npy")
            if os.path.exists(mask_path):
                f_mask = np.load(mask_path)
                n_frames = len(f_mask)
                # Upsample 32 segment scores → n_frames (Exactly as in transmil_kaggle.py)
                f_scores = np.repeat(vid_score, max(1, n_frames // 32 + 1))[:n_frames]
                all_frame_scores.extend(f_scores)
                all_frame_labels.extend(f_mask)
    
    auc_roc = roc_auc_score(all_labels, all_scores) if len(np.unique(all_labels)) > 1 else 0.0
    f_roc = roc_auc_score(all_frame_labels, all_frame_scores) if len(all_frame_labels) > 0 and len(np.unique(all_frame_labels)) > 1 else 0.0
    
    return auc_roc, f_roc
