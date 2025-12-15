import os
import torch
import argparse
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader


class MyDataModule(pl.LightningDataModule):
    def __init__(self, data_file: str, batch_size: int = 32):
        super().__init__()
        self.data_file = data_file
        self.batch_size = batch_size

    def setup(self, stage=None):
        # Load preprocessed dataset (.pt file)
        dataset = torch.load(self.data_file)

        # Infer interest columns from filename
        filename = os.path.basename(self.data_file).replace(".pt", "")
        interest_str = filename.replace("dataset_", "").split("_rarity")[0]
        interests = interest_str.split("_")
        print(f"Interests inferred from filename: {interests}")

        # Extract features and targets
        X_train = dataset["train"]["embedding"]
        y_train = torch.stack([dataset["train"][col] for col in interests], dim=1)
        X_val = dataset["val"]["embedding"]
        y_val = torch.stack([dataset["val"][col] for col in interests], dim=1)
        X_test = dataset["test"]["embedding"]
        y_test = torch.stack([dataset["test"][col] for col in interests], dim=1)

        # Compute train mean/std for normalization
        X_mean, X_std = X_train.mean(0), X_train.std(0)
        y_mean, y_std = y_train.mean(0), y_train.std(0)

        # Normalize all sets using train stats
        X_train = (X_train - X_mean) / X_std
        y_train = (y_train - y_mean) / y_std
        X_val = (X_val - X_mean) / X_std
        y_val = (y_val - y_mean) / y_std
        X_test = (X_test - X_mean) / X_std
        y_test = (y_test - y_mean) / y_std

        # Create TensorDatasets
        self.train_dataset = TensorDataset(X_train, y_train)
        self.val_dataset = TensorDataset(X_val, y_val)
        self.test_dataset = TensorDataset(X_test, y_test)

        print("Train shapes:", X_train.shape, y_train.shape)
        print("Val shapes:", X_val.shape, y_val.shape)
        print("Test shapes:", X_test.shape, y_test.shape)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_file', type=str, required=True)
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    args = parser.parse_args()

    datamodule = MyDataModule(args.data_file, batch_size=args.batch_size)
    datamodule.setup()

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    for batch in train_loader:
        tokens, targets = batch
        print("Tokens shape:", tokens.shape)
        print("Targets shape:", targets.shape)
        break