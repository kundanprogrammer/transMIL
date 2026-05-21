import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE
import numpy as np

def plot_results(results_dict, save_path="/kaggle/working/mil_vad_results.png"):
    """Generate publication-quality 2×2 results figure."""
    n_experiments = len(results_dict)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#e67e22", "#1abc9c"]
    
    # Plot 1: ROC curves
    ax = axes[0, 0]
    for i, (name, res) in enumerate(results_dict.items()):
        fpr, tpr, _ = roc_curve(res["labels"], res["scores"])
        ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                label=f"{name} (AUC={res['auc_roc']:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves"); ax.legend(fontsize=8)
    
    # Plot 2: Score distributions
    ax = axes[0, 1]
    for i, (name, res) in enumerate(results_dict.items()):
        norm_scores = res["scores"][res["labels"] == 0]
        anom_scores = res["scores"][res["labels"] == 1]
        ax.hist(norm_scores, bins=30, alpha=0.4, color=colors[0], density=True,
                label="Normal" if i == 0 else None)
        ax.hist(anom_scores, bins=30, alpha=0.4, color=colors[1], density=True,
                label="Anomaly" if i == 0 else None)
    ax.set_xlabel("Anomaly Score"); ax.set_ylabel("Density")
    ax.set_title("Score Distribution"); ax.legend()
    
    # Plot 3: Training loss
    ax = axes[1, 0]
    for i, (name, res) in enumerate(results_dict.items()):
        ax.plot(res["history"]["loss"], color=colors[i % len(colors)], lw=1.5, label=name)
    ax.set_xlabel("Epoch"); ax.set_ylabel("MIL Ranking Loss")
    ax.set_title("Training Loss"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    
    # Plot 4: AUC over training
    ax = axes[1, 1]
    for i, (name, res) in enumerate(results_dict.items()):
        if res["history"]["auc_roc"]:
            ax.plot(res["history"]["epoch"],
                    [a * 100 for a in res["history"]["auc_roc"]],
                    color=colors[i % len(colors)], lw=2, marker="o",
                    markersize=4, label=name)
    ax.set_xlabel("Epoch"); ax.set_ylabel("AUC-ROC (%)")
    ax.set_title("AUC-ROC Over Training"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    
    plt.suptitle("MIL-VAD: Weakly Supervised Video Anomaly Detection", 
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"📊 Results saved to {save_path}")


# ── 13. Summary Table ─────────────────────────────────────────────────────

def plot_temporal_scores(results_dict, model_name, dataset, gt_data, save_path):
    if model_name not in results_dict: return
    
    per_video = results_dict[model_name]["per_video"]
    # Find one normal and one abnormal video
    anom_vids = [v for v, data in per_video.items() if data["label"] == 1]
    norm_vids = [v for v, data in per_video.items() if data["label"] == 0]
    
    if not anom_vids or not norm_vids: return
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    colors = {"scores": "#e74c3c", "gt": "#2ecc71", "norm": "#3498db"}
    
    # Abnormal Video Plot
    vid1 = anom_vids[0]
    scores1 = per_video[vid1]["scores"]
    axes[0].plot(np.linspace(0, 1, 32), scores1, lw=2.5, color=colors["scores"], label="Anomaly Score")
    # Try to find GT
    if gt_data and vid1 in gt_data:
        gt_mask = np.array(gt_data[vid1])
        axes[0].fill_between(np.linspace(0, 1, len(gt_mask)), 0, gt_mask, color=colors["gt"], alpha=0.3, label="Ground Truth Anomaly")
    axes[0].set_title(f"Abnormal Video ({vid1})", fontweight="bold")
    axes[0].set_ylabel("Score")
    axes[0].legend(loc="upper right")
    axes[0].set_ylim(-0.05, 1.05)
    
    # Normal Video Plot
    vid2 = norm_vids[0]
    scores2 = per_video[vid2]["scores"]
    axes[1].plot(np.linspace(0, 1, 32), scores2, lw=2.5, color=colors["norm"], label="Anomaly Score (Normal)")
    axes[1].set_title(f"Normal Video ({vid2})", fontweight="bold")
    axes[1].set_xlabel("Normalized Video Duration")
    axes[1].set_ylabel("Score")
    axes[1].legend(loc="upper right")
    axes[1].set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"   Saved temporal sequence to {save_path}")


def generate_tsne(model, dataset, device, save_path):
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    raw_feats = []
    labels_list = []
    
    # Setup a hook to extract Transformer latent representation
    latents = []
    def hook(m, i, o):
        latents.append(o.detach().cpu().numpy())
    
    # Attach hook to the final normalization layer of TransMIL
    handle = model.norm.register_forward_hook(hook)
    
    model.eval()
    with torch.no_grad():
        for feats, labels, _ in loader:
            feats = feats.to(device)
            raw_feats.append(feats.cpu().numpy())
            labels_list.extend(labels.numpy())
            _ = model(feats)
    
    handle.remove()
    
    # Flatten: (B, 32, D) -> (B, D) via temporal mean for video-level embedding
    raw_feats = np.concatenate(raw_feats, axis=0).mean(axis=1)  # (N, 1024)
    latents = np.concatenate(latents, axis=0).mean(axis=1)      # (N, hidden_dim)
    
    labels_array = np.array(labels_list)
    
    print("   Running multi-core t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    
    raw_2d = tsne.fit_transform(raw_feats)
    latent_2d = tsne.fit_transform(latents)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    color_map = {0: "#3498db", 1: "#e74c3c"} # Blue normal, Red anomaly
    label_map = {0: "Normal", 1: "Abnormal"}
    
    for l in [0, 1]:
        idx = labels_array == l
        ax1.scatter(raw_2d[idx, 0], raw_2d[idx, 1], c=color_map[l], label=label_map[l], alpha=0.7, edgecolors='w', s=60)
        ax2.scatter(latent_2d[idx, 0], latent_2d[idx, 1], c=color_map[l], label=label_map[l], alpha=0.7, edgecolors='w', s=60)
        
    ax1.set_title("Raw I3D Features", fontweight="bold")
    ax1.set_xticks([]); ax1.set_yticks([])
    ax1.legend()
    
    ax2.set_title("TransMIL Latent Space", fontweight="bold")
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"   Saved t-SNE plot to {save_path}")
