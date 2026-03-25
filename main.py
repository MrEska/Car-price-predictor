import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data_prep import prepare_data
from src.dataset import CarPriceDataset
from src.model import CarPriceModel


def check_array(name, arr):
    print(f"\n--- {name} ---")
    print("shape:", arr.shape)
    print("dtype:", arr.dtype)
    print("nan count:", np.isnan(arr).sum() if np.issubdtype(arr.dtype, np.floating) else "not-float")
    if np.issubdtype(arr.dtype, np.floating):
        print("min:", np.min(arr))
        print("max:", np.max(arr))
        print("mean:", np.mean(arr))


def main():
    print("Loading and preparing data...")
    data = prepare_data("Database/vehicle_price_prediction.csv")

    # 1. TEST PREPROCESSINGU
    check_array("x_train_num", data["x_train_num"])
    check_array("x_val_num", data["x_val_num"])
    check_array("x_test_num", data["x_test_num"])

    check_array("x_train_cat", data["x_train_cat"])
    check_array("y_train", data["y_train"])

    print("\nChecking category ranges...")
    for i, col in enumerate(data["cat_maps"].keys()):
        max_index_in_data = data["x_train_cat"][:, i].max()
        vocab_size = len(data["cat_maps"][col])
        print(f"{col}: max index in data = {max_index_in_data}, vocab size = {vocab_size}")
        assert max_index_in_data < vocab_size, f"Category index out of range in column {col}"

    # 2. TEST DATASET
    print("\nCreating dataset...")
    train_ds = CarPriceDataset(
        data["x_train_num"],
        data["x_train_cat"],
        data["y_train"]
    )

    print("Dataset length:", len(train_ds))

    sample_x_num, sample_x_cat, sample_y = train_ds[0]
    print("\nSingle sample:")
    print("x_num shape:", sample_x_num.shape)
    print("x_cat shape:", sample_x_cat.shape)
    print("y shape:", sample_y.shape if hasattr(sample_y, "shape") else "scalar")
    print("x_num dtype:", sample_x_num.dtype)
    print("x_cat dtype:", sample_x_cat.dtype)
    print("y dtype:", sample_y.dtype)

    # 3. TEST DATALOADER
    print("\nCreating dataloader...")
    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)

    x_num_batch, x_cat_batch, y_batch = next(iter(train_loader))
    print("\nBatch shapes:")
    print("x_num_batch:", x_num_batch.shape)
    print("x_cat_batch:", x_cat_batch.shape)
    print("y_batch:", y_batch.shape)

    print("\nBatch dtypes:")
    print("x_num_batch:", x_num_batch.dtype)
    print("x_cat_batch:", x_cat_batch.dtype)
    print("y_batch:", y_batch.dtype)

    # 4. TEST MODELU
    cat_cardinalities = [len(data["cat_maps"][col]) for col in data["cat_maps"]]

    print("\nCreating model...")
    model = CarPriceModel(
        num_numeric=data["x_train_num"].shape[1],
        cat_cardinalities=cat_cardinalities
    )

    print(model)

    print("\nRunning forward pass...")
    preds = model(x_num_batch, x_cat_batch)

    print("preds shape:", preds.shape)
    print("preds dtype:", preds.dtype)
    print("preds min:", preds.min().item())
    print("preds max:", preds.max().item())
    print("Any NaN in preds:", torch.isnan(preds).any().item())
    print("Any Inf in preds:", torch.isinf(preds).any().item())

    assert preds.shape[0] == x_num_batch.shape[0], "Prediction batch size mismatch"
    assert not torch.isnan(preds).any(), "Model returned NaN"
    assert not torch.isinf(preds).any(), "Model returned Inf"

    print("\nAll basic tests passed.")


if __name__ == "__main__":
    main()