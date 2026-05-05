import torch

DATA_PATH = "Database/vehicle_price_prediction.csv"

MODEL_DIR = "outputs/models"
HISTORY_DIR = "outputs/histories"
PREDICTION_DIR = "outputs/predictions"

SEED = 42

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

EXPERIMENTS = [
    {
        "name": "mlp_256_128_64_lr1e-3_dropout0.2",
        "hidden_dims": (256, 128, 64),
        "dropout": 0.2,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size_train": 1024,
        "batch_size_eval": 2048,
        "max_epochs": 10,
        "patience": 5,
        "loss": "huber",
        "huber_delta": 1.0,
    },
    {
        "name": "mlp_256_128_64_lr5e-4_dropout0.2",
        "hidden_dims": (256, 128, 64),
        "dropout": 0.2,
        "lr": 5e-4,
        "weight_decay": 1e-4,
        "batch_size_train": 1024,
        "batch_size_eval": 2048,
        "max_epochs": 10,
        "patience": 5,
        "loss": "huber",
        "huber_delta": 1.0,
    },
    {
        "name": "mlp_512_256_128_lr1e-3_dropout0.1",
        "hidden_dims": (512, 256, 128),
        "dropout": 0.1,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size_train": 1024,
        "batch_size_eval": 2048,
        "max_epochs": 10,
        "patience": 5,
        "loss": "huber",
        "huber_delta": 1.0,
    },
]