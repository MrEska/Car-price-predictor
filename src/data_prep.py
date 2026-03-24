import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


NUM_COLS = [
    "year",
    "mileage",
    "engine_hp",
    "owner_count",
    "vehicle_age",
    "mileage_per_year",
    "brand_popularity",
]

CAT_COLS = [
    "make",
    "model",
    "transmission",
    "fuel_type",
    "drivetrain",
    "body_type",
    "exterior_color",
    "interior_color",
    "accident_history",
    "seller_type",
    "condition",
    "trim",
]

TARGET_COL = "price"


#load dataset from file
def load_data(path):
    df = pd.read_csv(path)
    df["accident_history"] = df["accident_history"].fillna("Unknown")
    return df

#split data to train, validation, test
def split_data(df):
    x = df[NUM_COLS + CAT_COLS].copy()
    y = df[TARGET_COL].copy()

    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.5, random_state=42
    )

    return x_train, x_val, x_test, y_train, y_val, y_test

#count mean and std for numeric stats
def fit_numeric_stats(x_train):
    num_stats = {}

    for col in NUM_COLS:
        train_col = x_train[col].astype(np.float32)
        mean = train_col.mean()
        std = train_col.std()

        if std == 0:
            std = 1.0

        num_stats[col] = {"mean": mean, "std": std}

    return num_stats

#standardize numeric stats
def transform_numeric(df_part, num_stats):
    arr = []

    for col in NUM_COLS:
        x = df_part[col].astype(np.float32)
        mean = num_stats[col]["mean"]
        std = num_stats[col]["std"]
        x = ((x - mean) / std).to_numpy()
        arr.append(x)

    return np.stack(arr, axis=1).astype(np.float32)

#create map names -> numbers for model
def fit_categorical_maps(x_train):
    cat_maps = {}

    for col in CAT_COLS:
        vocab = {"__UNK__": 0}
        unique_vals = x_train[col].astype(str).unique().tolist()

        for val in unique_vals:
            if val not in vocab:
                vocab[val] = len(vocab)

        cat_maps[col] = vocab

    return cat_maps

#change names for numbers using map
def transform_categorical(df_part, cat_maps):
    arr = []

    for col in CAT_COLS:
        vocab = cat_maps[col]
        x = (
            df_part[col]
            .astype(str)
            .map(lambda v: vocab.get(v, 0))
            .to_numpy(dtype=np.int64)
        )
        arr.append(x)

    return np.stack(arr, axis=1).astype(np.int64)

#combine all above methods
def prepare_data(path):
    df = load_data(path)
    x_train, x_val, x_test, y_train, y_val, y_test = split_data(df)

    num_stats = fit_numeric_stats(x_train)
    cat_maps = fit_categorical_maps(x_train)

    x_train_num = transform_numeric(x_train, num_stats)
    x_val_num = transform_numeric(x_val, num_stats)
    x_test_num = transform_numeric(x_test, num_stats)

    x_train_cat = transform_categorical(x_train, cat_maps)
    x_val_cat = transform_categorical(x_val, cat_maps)
    x_test_cat = transform_categorical(x_test, cat_maps)

    y_train = y_train.to_numpy(dtype=np.float32)
    y_val = y_val.to_numpy(dtype=np.float32)
    y_test = y_test.to_numpy(dtype=np.float32)

    return {
        "x_train_num": x_train_num,
        "x_val_num": x_val_num,
        "x_test_num": x_test_num,
        "x_train_cat": x_train_cat,
        "x_val_cat": x_val_cat,
        "x_test_cat": x_test_cat,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "num_stats": num_stats,
        "cat_maps": cat_maps,
    }