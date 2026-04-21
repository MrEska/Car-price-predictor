import numpy as np
import torch


def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    is_train = optimizer is not None

    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    preds_all = []
    targets_all = []

    for x_num, x_cat, y in loader:
        x_num = x_num.to(device)
        x_cat = x_cat.to(device)
        y = y.to(device)

        with torch.set_grad_enabled(is_train):
            preds = model(x_num, x_cat)
            loss = criterion(preds, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * len(y)

        preds_all.append(preds.detach().cpu().numpy())
        targets_all.append(y.detach().cpu().numpy())

    preds_all = np.concatenate(preds_all)
    targets_all = np.concatenate(targets_all)

    preds_real = np.expm1(preds_all)
    targets_real = np.expm1(targets_all)

    mae = np.mean(np.abs(preds_real - targets_real))
    rmse = np.sqrt(np.mean((preds_real - targets_real) ** 2))

    avg_loss = total_loss / len(loader.dataset)

    return avg_loss, mae, rmse