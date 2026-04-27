import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

def plot_results(results, save_path="results.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # ROC Curves
    ax = axes[0]
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(res["labels"], res["scores"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={res['auc_roc']:.4f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_title("ROC Curves")
    ax.legend()
    
def plot_temporal_scores(per_video_results, vid_id, gt_data=None, save_path="temporal.png"):
    if vid_id not in per_video_results: return
    info = per_video_results[vid_id]
    scores = info["scores"]
    
    plt.figure(figsize=(10, 4))
    plt.plot(np.linspace(0, 1, 32), scores, label="Anomaly Score", color='red')
    if gt_data and vid_id in gt_data:
        gt = np.array(gt_data[vid_id])
        plt.fill_between(np.linspace(0, 1, len(gt)), 0, gt, color='green', alpha=0.2, label="Ground Truth")
    plt.title(f"Temporal Anomaly Scores: {vid_id}")
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def generate_latex_table(results, dataset_name):
    print(f"\\begin{{table}}[h]\n\\centering\n\\begin{{tabular}}{{lcc}}\n\\toprule")
    print(f"\\textbf{{Method}} & \\textbf{{AUC-ROC (%)}} & \\textbf{{AUC-PR (%)}} \\\\\n\\midrule")
    for name, res in results.items():
        if dataset_name in name:
            print(f"{name} & {res['auc_roc']*100:.2f} & {res['auc_pr']*100:.2f} \\\\")
    print(f"\\bottomrule\n\\end{{tabular}}\n\\caption{{Results on {dataset_name}}}\n\\end{{table}}")
