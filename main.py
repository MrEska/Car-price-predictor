from torch.utils.data import DataLoader
from src.data_prep import prepare_data
from src.dataset import CarPriceDataset
from src.model import CarPriceModel


def main():
    data = prepare_data("Dataset/vehicle_price_prediction.csv")

    train_ds = CarPriceDataset(
        data["x_train_num"],
        data["x_train_cat"],
        data["y_train"]
    )

    val_ds = CarPriceDataset(
        data["x_val_num"],
        data["x_val_cat"],
        data["y_val"]
    )

    test_ds = CarPriceDataset(
        data["x_test_num"],
        data["x_test_cat"],
        data["y_test"]
    )

    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=2048, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=2048, shuffle=False)

    cat_cardinalities = [len(data["cat_maps"][col]) for col in data["cat_maps"]]

    model = CarPriceModel(
        num_numeric=data["x_train_num"].shape[1],
        cat_cardinalities=cat_cardinalities
    )

    x_num, x_cat, y = next(iter(train_loader))
    preds = model(x_num, x_cat)

    print("preds shape:", preds.shape)

if __name__ == "__main__":
    main()