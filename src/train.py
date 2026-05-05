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

    # Bezpieczny powrót z log(price) do price.
    # Chroni przed sytuacją, gdzie niewytrenowany model zwróci ekstremalnie dużą log-cenę.
    max_target_log = np.max(targets_all)

    preds_all_safe = np.clip(preds_all, 0, max_target_log + 1.0)
    targets_all_safe = np.clip(targets_all, 0, max_target_log + 1.0)

    preds_real = np.expm1(preds_all_safe)
    targets_real = np.expm1(targets_all_safe)

    errors = preds_real - targets_real

    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))

    avg_loss = total_loss / len(loader.dataset)

    return avg_loss, mae, rmse


def predict(model, loader, device="cpu"):
    """
    Zwraca predykcje i prawdziwe wartości w normalnej skali ceny.
    """
    model.eval()

    preds_all = []
    targets_all = []

    with torch.no_grad():
        for x_num, x_cat, y in loader:
            x_num = x_num.to(device)
            x_cat = x_cat.to(device)

            preds = model(x_num, x_cat)

            preds_all.append(preds.cpu().numpy())
            targets_all.append(y.numpy())

    preds_all = np.concatenate(preds_all)
    targets_all = np.concatenate(targets_all)

    preds_real = np.expm1(preds_all)
    targets_real = np.expm1(targets_all)

    return preds_real, targets_real


def compute_baseline(y_train_log, y_val_log):
    """
    Baseline: zawsze przewidujemy średnią cenę z train.
    """
    train_prices = np.expm1(y_train_log)
    val_prices = np.expm1(y_val_log)

    mean_price = train_prices.mean()
    baseline_pred = np.full_like(val_prices, mean_price)

    mae = np.mean(np.abs(baseline_pred - val_prices))
    rmse = np.sqrt(np.mean((baseline_pred - val_prices) ** 2))

    return mae, rmse