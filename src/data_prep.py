import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import NUM_COLS, CAT_COLS, TARGET_COL, SEED


def load_data(path):
    """
    Wczytuje dane z CSV i uzupełnia braki.
    Numeryczne braki uzupełniamy medianą.
    Kategoryczne braki uzupełniamy tekstem 'Unknown'.
    """
    df = pd.read_csv(path)

    for col in NUM_COLS:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    for col in CAT_COLS:
        if df[col].isnull().any():
            df[col] = df[col].fillna("Unknown")

    return df


def split_data(df):
    """
    Dzieli dane na:
    - train: 80%
    - validation: 10%
    - test: 10%
    """
    x = df[NUM_COLS + CAT_COLS].copy()
    y = df[TARGET_COL].copy()

    x_train, x_temp, y_train, y_temp = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=SEED,
    )

    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.5,
        random_state=SEED,
    )

    return x_train, x_val, x_test, y_train, y_val, y_test


def fit_numeric_stats(x_train):
    """
    Liczy średnią i odchylenie standardowe dla cech numerycznych.
    Liczymy tylko na train, żeby nie podglądać walidacji/testu.
    """
    num_stats = {}

    for col in NUM_COLS:
        train_col = x_train[col].astype(np.float32)
        mean = train_col.mean()
        std = train_col.std()

        if std == 0 or np.isnan(std):
            std = 1.0

        num_stats[col] = {
            "mean": float(mean),
            "std": float(std),
        }

    return num_stats


def transform_numeric(df_part, num_stats):
    """
    Standaryzuje cechy numeryczne:
    x = (x - mean) / std
    """
    arr = []

    for col in NUM_COLS:
        x = df_part[col].astype(np.float32)
        mean = num_stats[col]["mean"]
        std = num_stats[col]["std"]

        x = ((x - mean) / std).to_numpy()
        arr.append(x)

    return np.stack(arr, axis=1).astype(np.float32)


def fit_categorical_maps(x_train):
    """
    Tworzy mapowanie kategorii tekstowych na liczby.
    Np. Toyota -> 1, BMW -> 2 itd.
    __UNK__ = 0 oznacza kategorię nieznaną.
    """
    cat_maps = {}

    for col in CAT_COLS:
        vocab = {"__UNK__": 0}
        unique_vals = x_train[col].astype(str).unique().tolist()

        for val in unique_vals:
            if val not in vocab:
                vocab[val] = len(vocab)

        cat_maps[col] = vocab

    return cat_maps


def transform_categorical(df_part, cat_maps):
    """
    Zamienia kategorie tekstowe na indeksy liczbowe.
    Jeśli pojawi się nieznana kategoria, dostaje indeks 0.
    """
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


def prepare_data(path, save_transformers_path="data_transformers.pkl"):
    """
    Główna funkcja przygotowania danych.
    Zwraca dane gotowe do PyTorch.
    """
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

    # Uczymy model na log(price), bo ceny mają duży rozrzut.
    y_train_log = np.log1p(y_train.to_numpy(dtype=np.float32))
    y_val_log = np.log1p(y_val.to_numpy(dtype=np.float32))
    y_test_log = np.log1p(y_test.to_numpy(dtype=np.float32))

    # Zapisujemy transformery, żeby później móc przetwarzać nowe dane tak samo.
    with open(save_transformers_path, "wb") as f:
        pickle.dump(
            {
                "num_stats": num_stats,
                "cat_maps": cat_maps,
                "num_cols": NUM_COLS,
                "cat_cols": CAT_COLS,
            },
            f,
        )

    return {
        "x_train_num": x_train_num,
        "x_val_num": x_val_num,
        "x_test_num": x_test_num,
        "x_train_cat": x_train_cat,
        "x_val_cat": x_val_cat,
        "x_test_cat": x_test_cat,
        "y_train": y_train_log,
        "y_val": y_val_log,
        "y_test": y_test_log,
        "num_stats": num_stats,
        "cat_maps": cat_maps,
    }