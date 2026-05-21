import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os

from src.datasets.dataset import ShanghaiTechMIL, UCFCrimeMIL, MILPairDataset
from src.models.transmil import TransMILClassifier
from src.models.sultani import MILClassifier
from src.models.rtfm import RTFMClassifier
from src.models.umil import UMILClassifier, UMILLoss
from src.models.pemil import PEMILClassifier
from src.utils.losses import MILRankingLoss
from src.utils.evaluation import evaluate_mil, temporal_localization_eval

def get_model(model_name, feat_dim=1024, hidden_dim=512, dropout=0.6):
    if model_name.lower() == "transmil":
        return TransMILClassifier(feat_dim=feat_dim, hidden_dim=hidden_dim, n_layers=1, n_heads=4)
    elif model_name.lower() == "sultani":
        return MILClassifier(feat_dim=feat_dim, hidden_dim=hidden_dim, dropout=dropout)
    elif model_name.lower() == "rtfm":
        return RTFMClassifier(feat_dim=feat_dim, hidden_dim=hidden_dim, dropout=dropout)
    elif model_name.lower() == "umil":
        return UMILClassifier(feat_dim=feat_dim, hidden_dim=hidden_dim, dropout=dropout)
    elif model_name.lower() == "pemil":
        return PEMILClassifier(feat_dim=feat_dim, hidden_dim=hidden_dim, dropout=dropout, n_prototypes=13)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def train_epoch(model, dataloader, optimizer, criterion, device, model_name):
    model.train()
    total_loss = 0
    for anom_feat, norm_feat in dataloader:
        anom_feat, norm_feat = anom_feat.to(device), norm_feat.to(device)
        optimizer.zero_grad()
        
        if model_name.lower() == "umil":
            anom_scores, anom_proj = model(anom_feat, return_proj=True)
            norm_scores, norm_proj = model(norm_feat, return_proj=True)
            loss, *_ = criterion(anom_scores, norm_scores, anom_proj, norm_proj)
        else:
            anom_scores = model(anom_feat)
            norm_scores = model(norm_feat)
            loss, *_ = criterion(anom_scores, norm_scores)
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(dataloader))

def main():
    parser = argparse.ArgumentParser(description="Train VAD models")
    parser.add_argument("--dataset", type=str, choices=["sht", "ucf"], default="sht", help="Dataset to use")
    parser.add_argument("--model", type=str, choices=["transmil", "sultani", "rtfm", "umil", "pemil"], default="transmil", help="Model to train")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset placeholders (Update paths!)
    SHT_ROOT = "/path/to/ShanghaiTech"
    UCF_ROOT = "/path/to/UCF-Crime"

    print(f"Initializing {args.model.upper()} on {args.dataset.upper()}")
    model = get_model(args.model).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    
    if args.model.lower() == "umil":
        criterion = UMILLoss()
    else:
        criterion = MILRankingLoss()

    print("Setup complete. Please integrate standard dataloaders to run loop.")

if __name__ == "__main__":
    main()
