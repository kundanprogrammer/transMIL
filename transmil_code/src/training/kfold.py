import torch
import numpy as np
from torch.utils.data import DataLoader
from training.engine import train_mil, evaluate_mil
from data.ucf_crime import UCFCrimeMIL, MILPairDataset

def run_kfold_cv(root, model_class, device, config, n_folds=4):
    """Run K-Fold cross-validation on UCF-Crime."""
    fold_aucs = []
    for fold in range(1, n_folds + 1):
        print(f"\n--- Starting Fold {fold}/{n_folds} ---")
        train_ds = UCFCrimeMIL(root, split="train", fold=fold)
        test_ds = UCFCrimeMIL(root, split="test", fold=fold)
        
        pair_ds = MILPairDataset(train_ds)
        train_loader = DataLoader(pair_ds, batch_size=config["batch_size"], shuffle=True)
        
        model = model_class(**config["model_params"]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
        criterion = config["criterion"]
        
        best_fold_auc = 0
        for epoch in range(1, config["epochs"] + 1):
            train_mil(model, train_loader, optimizer, criterion, device)
            if epoch % config["eval_every"] == 0:
                auc, _ = evaluate_mil(model, test_ds, None, device)
                if auc > best_fold_auc:
                    best_fold_auc = auc
        
        fold_aucs.append(best_fold_auc)
        print(f"Fold {fold} Best AUC: {best_fold_auc:.4f}")
    
    print(f"\nMean K-Fold AUC: {np.mean(fold_aucs):.4f} +/- {np.std(fold_aucs):.4f}")
    return fold_aucs
