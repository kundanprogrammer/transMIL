import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def get_latents(model, dataset, device):
    """Extract latent representations using a forward hook."""
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    
    latents = []
    def hook(m, i, o):
        latents.append(o.detach().cpu().numpy())
    
    # Attempt to attach to a normalization layer (likely model.norm)
    handle = None
    if hasattr(model, 'norm'):
        handle = model.norm.register_forward_hook(hook)
    
    model.eval()
    labels_list = []
    with torch.no_grad():
        for feats, labels, _ in loader:
            feats = feats.to(device)
            labels_list.extend(labels.numpy())
            _ = model(feats)
    
    if handle: handle.remove()
    
    if latents:
        latents = np.concatenate(latents, axis=0).mean(axis=1)
        return latents, np.array(labels_list)
    return None, None

def plot_tsne(latents, labels, save_path="tsne.png"):
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    latent_2d = tsne.fit_transform(latents)
    
    plt.figure(figsize=(8, 6))
    for l in np.unique(labels):
        idx = labels == l
        label = "Normal" if l == 0 else "Anomaly"
        plt.scatter(latent_2d[idx, 0], latent_2d[idx, 1], label=label, alpha=0.6)
    plt.title("t-SNE of TransMIL Latent Space")
    plt.legend()
    plt.savefig(save_path)
    plt.close()
