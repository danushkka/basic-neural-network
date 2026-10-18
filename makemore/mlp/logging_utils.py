import csv
import os
from datetime import datetime


def log_experiment(
    config: dict,
    train_loss: float,
    val_loss: float,
    csv_path: str = "experiments.csv",
):
    """
    Append one row to experiments.csv.
    Creates the file with headers on first run.
    """
    row = {
        "date": datetime.now().isoformat(timespec="seconds"),
        **config,
        "train": round(train_loss, 4),
        "val": round(val_loss, 4),
        "gap": round(val_loss - train_loss, 4),
    }

    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"Logged to {csv_path}")
