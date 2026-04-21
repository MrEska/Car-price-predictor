import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data_prep import prepare_data
from src.dataset import CarPriceDataset
from src.model import CarPriceModel
from src.train import run_epoch

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    data = prepare_data("Database/vehicle_price_prediction.csv")

    train_ds = CarPriceDataset(data["x_train_num"], data["x_train_cat"], data["y_train"])
    val_ds = CarPriceDataset(data["x_val_num"], data["x_val_cat"], data["y_val"])
    test_ds = CarPriceDataset(data["x_test_num"], data["x_test_cat"], data["y_test"])

    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=2048, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=2048, shuffle=False)

    cat_cardinalities = [len(data["cat_maps"][col]) for col in data["cat_maps"]]

    model = CarPriceModel(
        num_numeric=data["x_train_num"].shape[1],
        cat_cardinalities=cat_cardinalities
    ).to(device)

    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    max_epochs = 50

    for epoch in range(max_epochs):
        train_loss, train_mae, train_rmse = run_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_mae, val_rmse = run_epoch(
            model, val_loader, criterion, None, device
        )

        print(
            f"Epoch {epoch + 1:02d} | "
            f"train_loss={train_loss:.4f} train_mae={train_mae:.0f} train_rmse={train_rmse:.0f} | "
            f"val_loss={val_loss:.4f} val_mae={val_mae:.0f} val_rmse={val_rmse:.0f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
            print("   -> Zapisano nowy najlepszy model!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[Early Stopping] Brak poprawy przez {patience} epok. Przerywam trening.")
                break

    print("\nŁadowanie najlepszego modelu do testów...")
    model.load_state_dict(torch.load("best_model.pth"))

    test_loss, test_mae, test_rmse = run_epoch(
        model, test_loader, criterion, None, device
    )

    print(
        f"TEST | loss={test_loss:.4f} mae={test_mae:.0f} rmse={test_rmse:.0f}"
    )

if __name__ == "__main__":
    main()