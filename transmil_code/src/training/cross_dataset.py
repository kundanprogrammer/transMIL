import torch
from torch.utils.data import DataLoader
from training.engine import train_mil, evaluate_mil
from data.ucf_crime import MILPairDataset

def run_cross_dataset_transfer(model, train_dataset, test_dataset, gt_data, device, config):
    """Train on one dataset (e.g., UCF) and test on another (e.g., SHT)."""
    pair_ds = MILPairDataset(train_dataset)
    train_loader = DataLoader(pair_ds, batch_size=config["batch_size"], shuffle=True, drop_last=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    criterion = config["criterion"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    
    best_auc = 0
    for epoch in range(1, config["epochs"] + 1):
        loss = train_mil(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        
        if epoch % config["eval_every"] == 0:
            auc, f_auc = evaluate_mil(model, test_dataset, gt_data, device)
            if auc > best_auc:
                best_auc = auc
            print(f"Cross-Dataset Epoch {epoch} | Loss: {loss:.4f} | Test AUC: {auc:.4f}")
    
    return best_auc
