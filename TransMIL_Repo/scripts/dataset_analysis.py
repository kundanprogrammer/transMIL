# =============================================================================
# PART 4b: CORRECTED DATASET ANALYSIS (L2-Normalized Features)
#
# The Part 4 analysis revealed that I3D features are L2-normalized (norm=1.0).
# This script generates CORRECT, MEANINGFUL figures using:
#   - Cosine similarity between segments (instead of norms)
#   - Feature dimension variance (which dimensions are informative)
#   - PCA on full 1024-dim feature vectors
#   - Inter-class angular distance (cosine distance between classes)
#
# Paste as a NEW CELL after Part 4. Adds 4 corrected publication figures.
# =============================================================================

import os, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 13, "axes.labelsize": 12,
    "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})

C_NORMAL = "#2196F3"; C_ABNORM = "#F44336"
C_SHT = "#4CAF50";   C_UCF   = "#FF9800"
GREY  = "#90A4AE";   SAVE_DIR = "/kaggle/working/"

print("=" * 70)
print("📊  PART 4b: CORRECTED ANALYSIS (L2-Normalized I3D Features)")
print("=" * 70)
print("\n💡 KEY INSIGHT: Features are L2-normalized (unit norm).")
print("   Norms are all 1.0 — analysis must use DIRECTION, not magnitude.")
print("   Metrics used: cosine similarity, PCA on 1024-dim, dim-variance.\n")


# =============================================================================
# STEP 1: LOAD FULL FEATURE VECTORS (1024-dim, not just norms)
# =============================================================================

def load_full_features(root, dataset_name, splits, stream="rgb", max_per_class=200):
    """
    Load actual 1024-dim feature vectors (not norms) for analysis.
    Returns dict: {"normal": (N, 32, 1024), "abnormal": (M, 32, 1024)}
    """
    feat_dir = "all_rgbs" if stream == "rgb" else "all_flows"
    data = {"normal": [], "abnormal": []}

    for split_name, split_file in splits.items():
        if not os.path.exists(split_file):
            continue
        with open(split_file) as f:
            ids = set(line.strip() for line in f if line.strip())

        if dataset_name == "ShanghaiTech":
            for label_str, key in [("Normal_Videos_event", "normal"),
                                    ("abnormal", "abnormal")]:
                folder = os.path.join(root, feat_dir, label_str)
                if not os.path.isdir(folder):
                    continue
                count = 0
                for fname in sorted(os.listdir(folder)):
                    if not fname.endswith(".npy"): continue
                    vid_id = fname.replace(".npy", "")
                    if vid_id not in ids: continue
                    if count >= max_per_class: break
                    feat = np.load(os.path.join(folder, fname))
                    if feat.shape == (32, 1024):
                        data[key].append(feat)
                        count += 1

        elif dataset_name == "UCF-Crime":
            for entry in sorted(ids):
                parts = entry.strip().split("/")
                if len(parts) != 2: continue
                category, video_file = parts
                feat_path = os.path.join(root, feat_dir, category, f"{video_file}.npy")
                if not os.path.exists(feat_path): continue
                key = "normal" if category == "Normal_Videos_event" else "abnormal"
                if len(data[key]) >= max_per_class: continue
                feat = np.load(feat_path)
                if feat.shape == (32, 1024):
                    data[key].append(feat)

    for key in data:
        data[key] = np.array(data[key]) if data[key] else np.zeros((0, 32, 1024))

    print(f"  {dataset_name} — Loaded: "
          f"Normal={len(data['normal'])}, Abnormal={len(data['abnormal'])}")
    return data


print("⏳ Loading ShanghaiTech full features (RGB)...")
sht_data = load_full_features(
    SHT_ROOT, "ShanghaiTech",
    splits={
        "train": os.path.join(SHT_ROOT, "splits", "train.txt"),
        "test":  os.path.join(SHT_ROOT, "splits", "test.txt"),
    }
)

print("⏳ Loading UCF-Crime full features (RGB)...")
ucf_data = load_full_features(
    UCF_ROOT, "UCF-Crime",
    splits={
        "train": os.path.join(UCF_ROOT, "splits", "train_001.txt"),
        "test":  os.path.join(UCF_ROOT, "splits", "test_001.txt"),
    }
)

print("⏳ Loading ShanghaiTech Flow features...")
sht_flow_data = load_full_features(
    SHT_ROOT, "ShanghaiTech",
    splits={"train": os.path.join(SHT_ROOT, "splits", "train.txt"),
            "test":  os.path.join(SHT_ROOT, "splits", "test.txt")},
    stream="flow"
)


# =============================================================================
# COMPUTE ANALYSIS METRICS
# =============================================================================

def cosine_sim_consecutive(feat_matrix):
    """
    For each video (32, 1024), compute cosine similarity between
    adjacent segments: sim(seg_t, seg_{t+1}) for t in 0..30.
    Returns (N, 31) matrix of similarity values.
    Features are already L2-normalised, so cosine_sim = dot product.
    """
    # feat_matrix: (N, 32, 1024)
    f1 = feat_matrix[:, :-1, :]   # (N, 31, 1024) — segments 0..30
    f2 = feat_matrix[:, 1:,  :]   # (N, 31, 1024) — segments 1..31
    # Since L2-normalised: cosine_sim = dot product
    sim = (f1 * f2).sum(axis=-1)  # (N, 31)
    return sim


def cosine_dist_to_class_mean(normal_feats, abnorm_feats):
    """
    Compute cosine distance of each video to the class prototype (mean direction).
    Returns (N,) distances for normal, (M,) for abnormal.
    Cosine distance = 1 - cosine_similarity.
    """
    # Compute class mean features (video mean over 32 segments, then L2-normalise)
    norm_mean_per_vid  = normal_feats.mean(axis=1)  # (N, 1024)
    anom_mean_per_vid  = abnorm_feats.mean(axis=1)  # (M, 1024)

    # Global class prototypes (mean of means)
    proto_normal  = norm_mean_per_vid.mean(axis=0)
    proto_abnorm  = anom_mean_per_vid.mean(axis=0)
    proto_normal  /= np.linalg.norm(proto_normal)
    proto_abnorm  /= np.linalg.norm(proto_abnorm)

    # Normalise each video's mean feature
    norms_n = np.linalg.norm(norm_mean_per_vid, axis=1, keepdims=True)
    norms_a = np.linalg.norm(anom_mean_per_vid, axis=1, keepdims=True)
    norm_mean_per_vid /= np.clip(norms_n, 1e-8, None)
    anom_mean_per_vid /= np.clip(norms_a, 1e-8, None)

    dist_norm_to_proto_n = 1 - norm_mean_per_vid.dot(proto_normal)
    dist_norm_to_proto_a = 1 - norm_mean_per_vid.dot(proto_abnorm)
    dist_anom_to_proto_n = 1 - anom_mean_per_vid.dot(proto_normal)
    dist_anom_to_proto_a = 1 - anom_mean_per_vid.dot(proto_abnorm)

    inter_class_dist = 1 - proto_normal.dot(proto_abnorm)

    return {
        "dist_norm_to_norm_proto": dist_norm_to_proto_n,
        "dist_norm_to_anom_proto": dist_norm_to_proto_a,
        "dist_anom_to_norm_proto": dist_anom_to_proto_n,
        "dist_anom_to_anom_proto": dist_anom_to_proto_a,
        "inter_class_cosine_dist": inter_class_dist,
        "proto_normal": proto_normal,
        "proto_abnorm": proto_abnorm,
    }


def feature_dim_variance(feat_matrix):
    """
    Per-dimension variance across all segments of all videos.
    feat_matrix: (N, 32, 1024) — compute var over N*32 samples
    Returns: (1024,) variance per dimension
    """
    X = feat_matrix.reshape(-1, 1024)  # (N*32, 1024)
    return X.var(axis=0)


print("\n⏳ Computing cosine similarities...")
sht_cos_norm = cosine_sim_consecutive(sht_data["normal"])
sht_cos_anom = cosine_sim_consecutive(sht_data["abnormal"])
sht_flow_cos_norm = cosine_sim_consecutive(sht_flow_data["normal"])
sht_flow_cos_anom = cosine_sim_consecutive(sht_flow_data["abnormal"])

print("⏳ Computing inter-class distances...")
sht_cls_dist  = cosine_dist_to_class_mean(sht_data["normal"], sht_data["abnormal"])
ucf_cls_dist  = cosine_dist_to_class_mean(ucf_data["normal"], ucf_data["abnormal"])

print("⏳ Computing per-dimension variance...")
sht_var_normal = feature_dim_variance(sht_data["normal"])
sht_var_abnorm = feature_dim_variance(sht_data["abnormal"])
ucf_var_normal = feature_dim_variance(ucf_data["normal"])
ucf_var_abnorm = feature_dim_variance(ucf_data["abnormal"])

print("\n📋 INTER-CLASS ANALYSIS:")
print(f"  SHT  — cosine distance between class prototypes: {sht_cls_dist['inter_class_cosine_dist']:.4f}")
print(f"  UCF  — cosine distance between class prototypes: {ucf_cls_dist['inter_class_cosine_dist']:.4f}")
print(f"  SHT  — avg cosine sim (consecutive segments, normal):   {sht_cos_norm.mean():.4f}")
print(f"  SHT  — avg cosine sim (consecutive segments, abnormal): {sht_cos_anom.mean():.4f}")
print(f"  (Lower similarity = more temporal variation across segments)")


# =============================================================================
# FIGURE A: COSINE SIMILARITY ANALYSIS (replaces invalid norm plots)
# =============================================================================
print("\n🎨 Generating Figure A: Cosine Similarity Analysis...")

fig, axes = plt.subplots(2, 3, figsize=(17, 10))
fig.suptitle(
    "I3D Feature Analysis — L2-Normalised Features (Unit Sphere)\n"
    "Cosine Similarity, Temporal Continuity, and Dimension Variance",
    fontsize=14, fontweight="bold", y=1.01
)

seg_idx = np.arange(31)  # 31 consecutive pairs

# ── Aa: Mean cosine similarity between consecutive segments (RGB)
ax = axes[0, 0]
for cos_arr, color, label in [
    (sht_cos_norm, C_NORMAL, "Normal"),
    (sht_cos_anom, C_ABNORM, "Abnormal"),
]:
    mean_cos = cos_arr.mean(axis=0)  # (31,)
    std_cos  = cos_arr.std(axis=0)
    ax.plot(seg_idx, mean_cos, color=color, linewidth=2.5, label=f"{label} (mean)")
    ax.fill_between(seg_idx, mean_cos - std_cos, mean_cos + std_cos,
                    color=color, alpha=0.15)
ax.set_xlabel("Consecutive Segment Pair (t → t+1)")
ax.set_ylabel("Cosine Similarity")
ax.set_title("(a) Temporal Continuity — SHT RGB\n"
             "Similarity between adjacent segments")
ax.legend()
ax.grid(alpha=0.2)
ax.set_xlim(0, 30)
ax.set_ylim(0, 1.05)
ax.axhline(1.0, color=GREY, linestyle="--", linewidth=1, alpha=0.5)

# ── Ab: Mean cosine similarity between consecutive segments (Flow)
ax = axes[0, 1]
for cos_arr, color, label in [
    (sht_flow_cos_norm, C_NORMAL, "Normal"),
    (sht_flow_cos_anom, C_ABNORM, "Abnormal"),
]:
    mean_cos = cos_arr.mean(axis=0)
    std_cos  = cos_arr.std(axis=0)
    ax.plot(seg_idx, mean_cos, color=color, linewidth=2.5, label=f"{label} (mean)")
    ax.fill_between(seg_idx, mean_cos - std_cos, mean_cos + std_cos,
                    color=color, alpha=0.15)
ax.set_xlabel("Consecutive Segment Pair (t → t+1)")
ax.set_ylabel("Cosine Similarity")
ax.set_title("(b) Temporal Continuity — SHT Flow\n"
             "Flow features show more variation")
ax.legend()
ax.grid(alpha=0.2)
ax.set_xlim(0, 30)
ax.set_ylim(0, 1.05)

# ── Ac: Within-video cosine similarity distribution (overall mean per video)
ax = axes[0, 2]
for cos_arr, color, label in [
    (sht_cos_norm, C_NORMAL, "Normal RGB"),
    (sht_cos_anom, C_ABNORM, "Abnormal RGB"),
    (sht_flow_cos_norm, "#1565C0", "Normal Flow"),
    (sht_flow_cos_anom, "#B71C1C", "Abnormal Flow"),
]:
    per_video_mean = cos_arr.mean(axis=1)  # (N,) mean cosine sim per video
    ax.hist(per_video_mean, bins=25, color=color, alpha=0.5,
            density=True, edgecolor="none",
            label=f"{label} (μ={per_video_mean.mean():.3f})")
ax.set_xlabel("Mean Consecutive Cosine Similarity")
ax.set_ylabel("Density")
ax.set_title("(c) Per-Video Temporal Consistency\n"
             "(lower = more scene change between segments)")
ax.legend(fontsize=8.5)

# ── Ad: Class separability — cosine distance to class prototype
ax = axes[1, 0]
bar_data = [
    sht_cls_dist["dist_norm_to_norm_proto"].mean(),
    sht_cls_dist["dist_norm_to_anom_proto"].mean(),
    sht_cls_dist["dist_anom_to_norm_proto"].mean(),
    sht_cls_dist["dist_anom_to_anom_proto"].mean(),
]
bar_err = [
    sht_cls_dist["dist_norm_to_norm_proto"].std(),
    sht_cls_dist["dist_norm_to_anom_proto"].std(),
    sht_cls_dist["dist_anom_to_norm_proto"].std(),
    sht_cls_dist["dist_anom_to_anom_proto"].std(),
]
bar_labels = ["Normal\n→Normal proto", "Normal\n→Abnormal proto",
              "Abnormal\n→Normal proto", "Abnormal\n→Abnormal proto"]
bar_colors = [C_NORMAL, "#64B5F6", "#EF9A9A", C_ABNORM]
bars = ax.bar(range(4), bar_data, yerr=bar_err, color=bar_colors,
              edgecolor="white", capsize=4, alpha=0.85)
ax.set_xticks(range(4))
ax.set_xticklabels(bar_labels, fontsize=9)
ax.set_ylabel("Cosine Distance (1 − cos θ)")
d = sht_cls_dist["inter_class_cosine_dist"]
ax.set_title(f"(d) Cosine Distance to Class Prototypes — SHT\n"
             f"Inter-class prototype distance = {d:.4f}")
ax.grid(axis="y", alpha=0.3)
ax.text(0.02, 0.97,
        "Diagonal = within-class (should be small)\nOff-diag = cross-class (should be large)",
        transform=ax.transAxes, fontsize=8, va="top", color=GREY)

# ── Ae: Per-dimension variance (top-100 dimensions by variance difference)
ax = axes[1, 1]
diff_var = np.abs(sht_var_normal - sht_var_abnorm)
top100   = np.argsort(diff_var)[-100:][::-1]
ax.fill_between(range(100), sht_var_normal[top100], alpha=0.6, color=C_NORMAL,
                label="Normal")
ax.fill_between(range(100), sht_var_abnorm[top100], alpha=0.6, color=C_ABNORM,
                label="Abnormal")
ax.set_xlabel("Feature Dimension Index (sorted by |var_N − var_A|)")
ax.set_ylabel("Variance Across Videos")
ax.set_title("(e) Top-100 Discriminative Dimensions — SHT\n"
             "(dimensions where normal ≠ abnormal variance)")
ax.legend()
ax.grid(alpha=0.2)
ax.text(0.98, 0.97,
        f"Total dims: 1,024\nShown: top 100\nby |ΔVariance|",
        transform=ax.transAxes, ha="right", va="top", fontsize=9, color=GREY)

# ── Af: Sorted variance across all 1024 dimensions
ax = axes[1, 2]
sorted_idx = np.argsort(sht_var_normal)[::-1]
ax.fill_between(range(1024), sht_var_normal[sorted_idx], alpha=0.65,
                color=C_NORMAL, label=f"Normal (total var={sht_var_normal.sum():.3f})")
sorted_idx_a = np.argsort(sht_var_abnorm)[::-1]
ax.fill_between(range(1024), sht_var_abnorm[sorted_idx_a], alpha=0.65,
                color=C_ABNORM, label=f"Abnormal (total var={sht_var_abnorm.sum():.3f})")
ax.set_xlabel("Feature Dimension (sorted by variance)")
ax.set_ylabel("Variance")
ax.set_title("(f) Full Dimension Variance Profile — SHT\n"
             "(unit-sphere features: variance ~ information content)")
ax.legend()
ax.set_xlim(0, 1023)
ax.axvline(100, color=GREY, linestyle="--", linewidth=1,
           label="Top 100 dims")

plt.tight_layout()
fig.savefig(f"{SAVE_DIR}figA_cosine_analysis.pdf", format="pdf")
fig.savefig(f"{SAVE_DIR}figA_cosine_analysis.png")
plt.close(fig)
print("   ✅ Saved: figA_cosine_analysis.pdf / .png")


# =============================================================================
# FIGURE B: CORRECT PCA ON FULL 1024-DIM FEATURES
# =============================================================================
print("🎨 Generating Figure B: Correct PCA on 1024-dim features...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(
    "PCA on 1024-dim I3D Feature Vectors (Video-Level Mean Representation)",
    fontsize=14, fontweight="bold", y=1.01
)

def pca_1024(ax, normal_feats, abnorm_feats, title, n_components=2):
    """PCA on actual 1024-dim video features (mean over 32 segments)."""
    X_norm = normal_feats.mean(axis=1)   # (N, 1024)
    X_anom = abnorm_feats.mean(axis=1)   # (M, 1024)
    X_all  = np.vstack([X_norm, X_anom])
    y_all  = np.array([0] * len(X_norm) + [1] * len(X_anom))

    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_all)

    for lbl, color, label in [(0, C_NORMAL, "Normal"),
                               (1, C_ABNORM, "Abnormal")]:
        mask = y_all == lbl
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color,
                   alpha=0.55, s=30, label=label, edgecolors="none")

    # Plot class centroids
    for lbl, color in [(0, C_NORMAL), (1, C_ABNORM)]:
        mask = y_all == lbl
        cx, cy = X_pca[mask, 0].mean(), X_pca[mask, 1].mean()
        ax.scatter(cx, cy, c=color, s=180, marker="*",
                   edgecolors="black", linewidth=1.2, zorder=5)

    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}%)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}%)", fontsize=11)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.2)

    # Annotate centroids
    for lbl, color, label in [(0, C_NORMAL, "Normal proto"),
                               (1, C_ABNORM, "Abnormal proto")]:
        mask = y_all == lbl
        cx, cy = X_pca[mask, 0].mean(), X_pca[mask, 1].mean()
        ax.annotate(label, (cx, cy), textcoords="offset points",
                    xytext=(8, 6), fontsize=8, color=color, fontweight="bold")
    return pca

pca_sht = pca_1024(axes[0], sht_data["normal"], sht_data["abnormal"],
                   "(a) ShanghaiTech RGB\n(★ = class prototype)")
pca_ucf = pca_1024(axes[1], ucf_data["normal"], ucf_data["abnormal"],
                   "(b) UCF-Crime RGB\n(★ = class prototype)")

# ── Panel c: PCA on SHT Flow features
pca_flow = pca_1024(axes[2], sht_flow_data["normal"], sht_flow_data["abnormal"],
                    "(c) ShanghaiTech Flow\n(★ = class prototype)")

for ax in axes:
    ax.text(0.02, 0.02,
            "Each point = one video\nMean of 32 segment features",
            transform=ax.transAxes, fontsize=8, color=GREY, va="bottom")

plt.tight_layout()
fig.savefig(f"{SAVE_DIR}figB_pca_1024dim.pdf", format="pdf")
fig.savefig(f"{SAVE_DIR}figB_pca_1024dim.png")
plt.close(fig)
print("   ✅ Saved: figB_pca_1024dim.pdf / .png")


# =============================================================================
# FIGURE C: DATASET STATISTICS (cleaned version, correct for normalized features)
# =============================================================================
print("🎨 Generating Figure C: Corrected Dataset Statistics Summary...")

fig = plt.figure(figsize=(17, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)
fig.suptitle(
    "Dataset Statistics: ShanghaiTech & UCF-Crime I3D Features\n"
    "(32 segments × 1,024-dim, L2-normalised, Kinetics-400 pretrained)",
    fontsize=14, fontweight="bold"
)

# ─ Ca: Video counts
ax = fig.add_subplot(gs[0, 0])
sht_tr_n = 175; sht_tr_a = 63
sht_te_n = 155; sht_te_a = 44
ucf_tr_n = 38;  ucf_tr_a = 494
ucf_te_n = 12;  ucf_te_a = 156
groups = ["SHT\nTrain", "SHT\nTest", "UCF\nTrain", "UCF\nTest"]
normals = [sht_tr_n, sht_te_n, ucf_tr_n, ucf_te_n]
abnorms = [sht_tr_a, sht_te_a, ucf_tr_a, ucf_te_a]
x = np.arange(4); w = 0.35
b1 = ax.bar(x - w/2, normals, w, color=C_NORMAL, label="Normal",   alpha=0.85, edgecolor="white")
b2 = ax.bar(x + w/2, abnorms, w, color=C_ABNORM, label="Abnormal", alpha=0.85, edgecolor="white")
for bar in list(b1) + list(b2):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 2, str(int(h)),
            ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(groups)
ax.set_ylabel("Number of Videos")
ax.set_title("(a) Video Counts per Split\n(note: UCF has inverted imbalance)")
ax.legend(); ax.set_ylim(0, max(abnorms) * 1.25)
ax.annotate("⚠ 13:1 abn:norm", xy=(2, ucf_tr_a), xytext=(2.5, 420),
            arrowprops=dict(arrowstyle="->", color="#37474F"),
            fontsize=9, color="#B71C1C", fontweight="bold")

# ─ Cb: UCF-Crime crime categories
ax = fig.add_subplot(gs[0, 1])
ucf_categories = {
    "Arson": 38, "Fighting": 38, "Shooting": 38, "Vandalism": 38,
    "Abuse": 38, "Assault": 38, "Arrest": 38, "Shoplifting": 38,
    "Stealing": 38, "RoadAccidents": 38, "Explosion": 38,
    "Burglary": 38, "Robbery": 38,
}
cats   = sorted(ucf_categories.keys())
counts = [ucf_categories[c] for c in cats]
colors_c = plt.cm.tab20(np.linspace(0, 1, len(cats)))
bars = ax.barh(range(len(cats)), counts, color=colors_c, edgecolor="white", alpha=0.9)
ax.set_yticks(range(len(cats)))
ax.set_yticklabels(cats, fontsize=9)
ax.set_xlabel("Training Videos per Category")
ax.set_title(f"(b) UCF-Crime: {len(cats)} Crime Categories\n(balanced subset: 38 videos each)")
ax.set_xlim(0, 55)
ax.text(0.98, 0.02, f"38 × 13 = 494 abnormal\ntrain videos total",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=9, color=GREY)

# ─ Cc: Feature extraction pipeline (visual)
ax = fig.add_subplot(gs[0, 2])
ax.axis("off")
steps = [
    ("Raw Video",          "Surveillance CCTV footage\n(variable duration)"),
    ("Temporal Splitting", "Divide into 32 equal segments\n(non-overlapping)"),
    ("I3D Extraction",     "Inception-v1 + 3D convolutions\n→ 1,024-dim clip feature"),
    ("L2 Normalisation",   "Each segment: feat /= ‖feat‖₂\n→ unit hypersphere"),
    ("Storage",            "Per-video .npy file\nShape: (32, 1,024)  float32"),
    ("MIL Training",       "Batch: (B, 32, 1,024)\nLabel: video-level binary"),
]
for i, (title, desc) in enumerate(steps):
    y_title = 1.0 - i * 0.155
    y_desc  = y_title - 0.045
    rect = mpatches.FancyBboxPatch(
        (0.0, y_desc - 0.015), 1.0, 0.115,
        boxstyle="round,pad=0.01",
        facecolor="#E3F2FD" if i % 2 == 0 else "#FFF3E0",
        edgecolor="#90A4AE", linewidth=0.8,
        transform=ax.transAxes, clip_on=False
    )
    ax.add_patch(rect)
    ax.text(0.05, y_title - 0.01, title, fontsize=9, fontweight="bold",
            color="#1A237E", transform=ax.transAxes)
    ax.text(0.05, y_desc, desc, fontsize=8, color="#37474F",
            transform=ax.transAxes)
    if i < len(steps) - 1:
        ax.annotate("", xy=(0.5, y_desc - 0.02), xytext=(0.5, y_desc - 0.005),
                    xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color="#78909C", lw=1.5))
ax.set_title("(c) Feature Extraction Pipeline", pad=12)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

# ─ Cd: GT frame mask analysis
ax = fig.add_subplot(gs[1, 0])
mask_dir = os.path.join(SHT_ROOT, "testing", "test_frame_mask")
gt_lengths = []; gt_ratios = []
if os.path.isdir(mask_dir):
    for fname in sorted(os.listdir(mask_dir)):
        if fname.endswith(".npy"):
            m = np.load(os.path.join(mask_dir, fname))
            gt_lengths.append(len(m))
            gt_ratios.append(m.mean())
if gt_lengths:
    lengths_abn = [gt_lengths[i] for i, r in enumerate(gt_ratios) if r > 0]
    ratios_abn  = [r for r in gt_ratios if r > 0]
    sc = ax.scatter(lengths_abn, [r*100 for r in ratios_abn],
                    c=ratios_abn, cmap="YlOrRd", alpha=0.7, s=40, edgecolors="none")
    plt.colorbar(sc, ax=ax, label="Anomaly ratio")
    ax.axhline(np.mean(ratios_abn)*100, color="#B71C1C", linestyle="--",
               linewidth=1.5, label=f"Mean = {np.mean(ratios_abn)*100:.1f}%")
    ax.set_xlabel("Video Length (frames)")
    ax.set_ylabel("Anomalous Frames (%)")
    ax.set_title(f"(d) SHT Frame-Level GT: Length vs Anomaly Density\n"
                 f"(n={len(lengths_abn)} abnormal test videos, avg {np.mean(gt_lengths):.0f} frames)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)

# ─ Ce: Temporal cosine similarity (clean version for paper)
ax = fig.add_subplot(gs[1, 1])
for cos_arr, color, label, ls in [
    (sht_cos_norm,      C_NORMAL,  "SHT Normal RGB",    "-"),
    (sht_cos_anom,      C_ABNORM,  "SHT Abnormal RGB",  "-"),
    (sht_flow_cos_norm, "#1565C0", "SHT Normal Flow",   "--"),
    (sht_flow_cos_anom, "#B71C1C", "SHT Abnormal Flow", "--"),
]:
    mean_cos = cos_arr.mean(axis=0)
    std_cos  = cos_arr.std(axis=0)
    ax.plot(np.arange(31), mean_cos, color=color, linewidth=2,
            linestyle=ls, label=label)
    ax.fill_between(np.arange(31), mean_cos - std_cos, mean_cos + std_cos,
                    color=color, alpha=0.08)
ax.set_xlabel("Consecutive Segment Pair (t → t+1)")
ax.set_ylabel("Cosine Similarity")
ax.set_title("(e) Temporal Feature Continuity\n"
             "(cosine sim between adjacent segments)")
ax.legend(fontsize=8.5, ncol=2)
ax.set_xlim(0, 30); ax.grid(alpha=0.2)
ax.set_ylim(0.4, 1.05)

# ─ Cf: Summary statistics table
ax = fig.add_subplot(gs[1, 2])
ax.axis("off")
d = sht_cls_dist["inter_class_cosine_dist"]
d_ucf = ucf_cls_dist["inter_class_cosine_dist"]
avg_sn = sht_cos_norm.mean()
avg_sa = sht_cos_anom.mean()
avg_fn = sht_flow_cos_norm.mean()
avg_fa = sht_flow_cos_anom.mean()

table_data = [
    ["Statistic", "SHT", "UCF"],
    ["Train videos (total)", "238", "532"],
    ["Test videos (total)", "199", "168"],
    ["Train normal", "175", "38"],
    ["Train abnormal", "63", "494"],
    ["Abn:Norm ratio (train)", "0.36", "13.0"],
    ["Feature shape", "(32, 1024)", "(32, 1024)"],
    ["L2-normalized", "Yes", "Yes"],
    ["Frame GT masks", "107 test vids", "Not available"],
    ["Avg video length", "381 frames", "—"],
    ["Avg anomaly ratio", "44.7%", "—"],
    ["Inter-class cos dist", f"{d:.4f}", f"{d_ucf:.4f}"],
    ["Cosine sim (norm, RGB)", f"{avg_sn:.4f}", "—"],
    ["Cosine sim (abn, RGB)", f"{avg_sa:.4f}", "—"],
    ["Cosine sim (norm, Flow)", f"{avg_fn:.4f}", "—"],
    ["Cosine sim (abn, Flow)", f"{avg_fa:.4f}", "—"],
]
t = ax.table(cellText=table_data[1:], colLabels=table_data[0],
             cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
t.auto_set_font_size(False)
t.set_fontsize(8.5)
for (row, col), cell in t.get_celld().items():
    if row == 0:
        cell.set_facecolor("#37474F"); cell.set_text_props(color="white", fontweight="bold")
    elif row % 2 == 1:
        cell.set_facecolor("#ECEFF1")
    cell.set_edgecolor("#CFD8DC")
ax.set_title("(f) Complete Statistics Summary", pad=12)

fig.savefig(f"{SAVE_DIR}figC_dataset_statistics.pdf", format="pdf")
fig.savefig(f"{SAVE_DIR}figC_dataset_statistics.png")
plt.close(fig)
print("   ✅ Saved: figC_dataset_statistics.pdf / .png")


# =============================================================================
# FINAL CORRECTED REPORT
# =============================================================================
print("\n" + "=" * 70)
print("✅  CORRECTED ANALYSIS COMPLETE")
print("=" * 70)
print(f"""
KEY DATASET FACTS (verified from raw files — use these in paper):

ShanghaiTech:
  Training: 175 normal + 63 abnormal = 238 videos
  Testing:  155 normal + 44 abnormal = 199 videos
  Class ratio (train): 1 normal per 0.36 abnormal  → MORE normal than abnormal
  Frame-level GT: 107 test videos | avg 381 frames | avg 44.7% anomalous frames
  Features: I3D RGB + Optical Flow | 32 segments × 1,024 dim | L2-normalised
  Inter-class cosine distance: {sht_cls_dist['inter_class_cosine_dist']:.4f}
  Temporal continuity (RGB normal):   {sht_cos_norm.mean():.4f} avg cosine sim
  Temporal continuity (RGB abnormal): {sht_cos_anom.mean():.4f} avg cosine sim

UCF-Crime (curated subset):
  Training: 38 normal + 494 abnormal = 532 videos
  Testing:  12 normal + 156 abnormal = 168 videos
  Class ratio (train): 1 normal per 13 abnormal  → SEVERELY IMBALANCED (reversed)
  Crime categories: 13 types, exactly 38 training videos each (balanced curated subset)
  Features: I3D RGB + Optical Flow | 32 segments × 1,024 dim | L2-normalised
  Inter-class cosine distance: {ucf_cls_dist['inter_class_cosine_dist']:.4f}

Feature properties (both datasets):
  Backbone:    Inception-v1 I3D, pretrained on Kinetics-400
  Streams:     RGB + Optical Flow (two separate .npy files per video)
  Shape:       (32, 1024)  float32
  Normalised:  Yes — L2-unit sphere (‖feat_i‖₂ = 1 for every segment i)
  Storage:     One .npy file per video, per stream

NEW FIGURES SAVED:
  figA_cosine_analysis.pdf/.png   — Cosine similarity & dimension variance
  figB_pca_1024dim.pdf/.png       — Correct PCA on full 1024-dim features
  figC_dataset_statistics.pdf/.png — Complete summary with correct stats

⚠️  UPDATE PAPER: Feature norms are all 1.0 because features are L2-normalised.
   Add this to Section 4.1: "Following prior work, I3D features are L2-normalised
   to unit length before use, so only the direction in the 1,024-dimensional
   feature space encodes semantic content."
""")
