# Standard library imports
import numpy as np
import pandas as pd
import argparse
import torch
import os


def main(usecols, interests, rarity=None, log_transform=False):
    print("Load CSV")
    # Load in CSV containing data
    df_combined = pd.read_csv('data/data_panama/Panama_gnps_dreams_plot_melted_info.csv', delimiter=',', usecols=usecols)
    # Remove rows with missing values in the interest columns
    df_combined = df_combined.dropna(subset=interests)
    # Convert embedding string into a NumPy array, then save back as JSON-style list
    df_combined['embedding'] = df_combined['embedding'].apply(lambda x: np.array([float(i) for i in str(x).split(',')]))
   
    # If rarity is set, filter df_combined to only include rare features
    if rarity is not None:
        print("Filter data by rarity")
        feature_counts = df_combined['FEATURE_ID'].value_counts()
        rare_species = feature_counts[feature_counts <= rarity].index
        df_combined = df_combined[df_combined['FEATURE_ID'].isin(rare_species)].reset_index(drop=True)

    # 80/10/10 grouped split by features
    print("Split data")
    all_features = np.random.permutation(df_combined['FEATURE_ID'].unique())
    n_total = len(all_features)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)

    train_features = all_features[:n_train]
    val_features = all_features[n_train:n_train + n_val]
    test_features = all_features[n_train + n_val:]

    train_df = df_combined[df_combined['FEATURE_ID'].isin(train_features)].reset_index(drop=True)
    val_df = df_combined[df_combined['FEATURE_ID'].isin(val_features)].reset_index(drop=True)
    test_df = df_combined[df_combined['FEATURE_ID'].isin(test_features)].reset_index(drop=True)

    # Apply log transform to interest columns if requested
    transform_params = {}
    if log_transform:
        print("Applying log transform to interest columns")
        for col in interests:
            # transform train/val/test using train-derived shift
            train_df[col] = np.log(train_df[col].astype(float).values)
            val_df[col] = np.log(val_df[col].astype(float).values)
            test_df[col] = np.log(test_df[col].astype(float).values)
            transform_params[col] = {"method": "log", "shift": 0.0}
        print("Transform params:", transform_params)

    print("Save as torch tensors")
    dataset = {
        "train": {
            "FEATURE_ID": torch.tensor(train_df['FEATURE_ID'].values, dtype=torch.long),
            "embedding": torch.tensor(np.stack(train_df['embedding'].values), dtype=torch.float),
            **{col: torch.tensor(train_df[col].values, dtype=torch.float) for col in interests}
        },
        "val": {
            "FEATURE_ID": torch.tensor(val_df['FEATURE_ID'].values, dtype=torch.long),
            "embedding": torch.tensor(np.stack(val_df['embedding'].values), dtype=torch.float),
            **{col: torch.tensor(val_df[col].values, dtype=torch.float) for col in interests}
        },
        "test": {
            "FEATURE_ID": torch.tensor(test_df['FEATURE_ID'].values, dtype=torch.long),
            "embedding": torch.tensor(np.stack(test_df['embedding'].values), dtype=torch.float),
            **{col: torch.tensor(test_df[col].values, dtype=torch.float) for col in interests}
        },
        "meta": {
            "interests": interests,
            "log_transform": bool(log_transform),
            "transform_params": transform_params
        }
    }
    interest_str = "_".join(interests)
    if rarity is not None:
        filename = f"data/processed_datasets/dataset_{interest_str}_rarity{rarity}.pt"
    else:
        filename = f"data/processed_datasets/dataset_{interest_str}.pt"
    if log_transform:
        filename = filename.replace(".pt", "_log.pt")
    os.makedirs('data/processed_datasets', exist_ok=True)
    torch.save(dataset, filename)
    print(f"Dataset saved to {filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i1', '--interest1', type=str, required=True, help='Primary target variable for analysis')
    parser.add_argument('-i2', '--interest2', type=str, default=None, help='Optional secondary target variable for analysis')
    parser.add_argument('-r', '--rarity', type=int, default=None, help='Threshold for feature rarity; if None, 80/20 split is used')
    parser.add_argument('--log_transform', action='store_true', help='Apply natural log transform to interest columns')

    args = parser.parse_args()
    interest1 = args.interest1
    interest2 = args.interest2
    rarity = args.rarity
    log_transform = args.log_transform

    # Create list of interests for downstream use
    interests = [interest1] if interest2 is None else [interest1, interest2]
    usecols = ['FEATURE_ID', 'embedding', *interests]

    main(usecols, interests, rarity, log_transform)