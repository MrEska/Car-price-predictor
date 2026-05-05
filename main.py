import os
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import (
    DATA_PATH,
    MODEL_DIR,
    HISTORY_DIR,
    PREDICTION_DIR,
    SEED,
    EXPERIMENTS,
)
from src.data_prep import prepare_data
from src.dataset import CarPriceDataset
from src.model import CarPriceModel
from src.train import run_epoch, predict, compute_baseline


def seed_everything(seed):
    """
    Ustawia seed, żeby wyniki były bardziej powtarzalne.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_dirs():
    """
    Tworzy foldery na modele, historie i predykcje.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(HISTORY_DIR, exist_ok=True)
    os.makedirs(PREDICTION_DIR, exist_ok=True)


def create_dataloaders(data, batch_size_train, batch_size_eval):
    """
    Tworzy Dataset i DataLoader dla train/val/test.
    """
    train_ds = CarPriceDataset(
        data["x_train_num"],
        data["x_train_cat"],
        data["y_train"],
    )

    val_ds = CarPriceDataset(
        data["x_val_num"],
        data["x_val_cat"],
        data["y_val"],
    )

    test_ds = CarPriceDataset(
        data["x_test_num"],
        data["x_test_cat"],
        data["y_test"],
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size_train,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size_eval,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size_eval,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader


def create_criterion(config):
    """
    Tworzy funkcję straty.
    """
    if config["loss"] == "huber":
        return nn.HuberLoss(delta=config["huber_delta"])

    if config["loss"] == "mse":
        return nn.MSELoss()

    raise ValueError(f"Nieznana funkcja straty: {config['loss']}")


def train_one_experiment(data, config, device):
    """
    Trenuje jeden wariant modelu.
    Zwraca wyniki najlepszego modelu.
    """
    print("\n" + "=" * 80)
    print(f"START EXPERIMENT: {config['name']}")
    print("=" * 80)

    train_loader, val_loader, test_loader = create_dataloaders(
        data=data,
        batch_size_train=config["batch_size_train"],
        batch_size_eval=config["batch_size_eval"],
    )

    cat_cardinalities = [
        len(data["cat_maps"][col])
        for col in data["cat_maps"]
    ]

    model = CarPriceModel(
        num_numeric=data["x_train_num"].shape[1],
        cat_cardinalities=cat_cardinalities,
        hidden_dims=config["hidden_dims"],
        dropout=config["dropout"],
    ).to(device)

    criterion = create_criterion(config)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    best_val_mae = float("inf")
    best_val_rmse = float("inf")
    best_val_loss = float("inf")
    best_epoch = 0

    patience_counter = 0
    history = []

    checkpoint_path = os.path.join(MODEL_DIR, f"{config['name']}.pth")
    history_path = os.path.join(HISTORY_DIR, f"{config['name']}_history.csv")
    predictions_path = os.path.join(PREDICTION_DIR, f"{config['name']}_test_predictions.csv")

    for epoch in range(config["max_epochs"]):
        train_loss, train_mae, train_rmse = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_mae, val_rmse = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
        )

        print(
            f"Epoch {epoch + 1:02d} | "
            f"train_loss={train_loss:.4f} train_mae={train_mae:.0f} train_rmse={train_rmse:.0f} | "
            f"val_loss={val_loss:.4f} val_mae={val_mae:.0f} val_rmse={val_rmse:.0f}"
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_mae": train_mae,
            "train_rmse": train_rmse,
            "val_loss": val_loss,
            "val_mae": val_mae,
            "val_rmse": val_rmse,
        })

        # Najlepszy model wybieramy po val_mae, bo MAE jest w złotówkach/dolarach
        # i najłatwiej go interpretować.
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_val_rmse = val_rmse
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "config": config,
                "cat_cardinalities": cat_cardinalities,
                "num_numeric": data["x_train_num"].shape[1],
                "num_stats": data["num_stats"],
                "cat_maps": data["cat_maps"],
                "best_val_mae": best_val_mae,
                "best_val_rmse": best_val_rmse,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
            }

            torch.save(checkpoint, checkpoint_path)
            print("   -> Zapisano nowy najlepszy model!")
        else:
            patience_counter += 1

            if patience_counter >= config["patience"]:
                print(
                    f"\n[Early Stopping] Brak poprawy przez "
                    f"{config['patience']} epok. Przerywam trening."
                )
                break

    history_df = pd.DataFrame(history)
    history_df.to_csv(history_path, index=False)

    print("\nŁadowanie najlepszego modelu do testów...")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_mae, test_rmse = run_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        optimizer=None,
        device=device,
    )

    preds_real, targets_real = predict(
        model=model,
        loader=test_loader,
        device=device,
    )

    predictions_df = pd.DataFrame({
        "true_price": targets_real,
        "predicted_price": preds_real,
        "error": preds_real - targets_real,
        "abs_error": np.abs(preds_real - targets_real),
    })

    predictions_df.to_csv(predictions_path, index=False)

    print(
        f"TEST | loss={test_loss:.4f} "
        f"mae={test_mae:.0f} "
        f"rmse={test_rmse:.0f}"
    )

    return {
        "experiment": config["name"],
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_mae": best_val_mae,
        "best_val_rmse": best_val_rmse,
        "test_loss": test_loss,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "checkpoint_path": checkpoint_path,
        "history_path": history_path,
        "predictions_path": predictions_path,
    }


def main():
    seed_everything(SEED)
    create_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    print("\nPrzygotowywanie danych...")
    data = prepare_data(DATA_PATH)

    print("\nDane gotowe:")
    print("x_train_num:", data["x_train_num"].shape)
    print("x_train_cat:", data["x_train_cat"].shape)
    print("y_train:", data["y_train"].shape)

    baseline_mae, baseline_rmse = compute_baseline(
        y_train_log=data["y_train"],
        y_val_log=data["y_val"],
    )

    print("\nBASELINE:")
    print(f"val_mae={baseline_mae:.0f}")
    print(f"val_rmse={baseline_rmse:.0f}")

    results = []

    for config in EXPERIMENTS:
        result = train_one_experiment(
            data=data,
            config=config,
            device=device,
        )

        results.append(result)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("best_val_mae")

    results_path = "outputs/experiment_results.csv"
    results_df.to_csv(results_path, index=False)

    print("\n" + "=" * 80)
    print("PODSUMOWANIE EKSPERYMENTÓW")
    print("=" * 80)
    print(results_df[[
        "experiment",
        "best_epoch",
        "best_val_mae",
        "best_val_rmse",
        "test_mae",
        "test_rmse",
    ]])

    best = results_df.iloc[0]

    print("\nNajlepszy eksperyment:")
    print(f"name: {best['experiment']}")
    print(f"best_val_mae: {best['best_val_mae']:.0f}")
    print(f"test_mae: {best['test_mae']:.0f}")
    print(f"checkpoint: {best['checkpoint_path']}")


if __name__ == "__main__":
    main()