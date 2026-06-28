## Imports
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter
import sys
from pathlib import Path
from types import SimpleNamespace
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde
import math
from matplotlib.backends.backend_pdf import PdfPages

# Add the parent of mixture_density_network to sys.path
project_root = Path(__file__).resolve().parents[1]  # points to Madidi_forest_analysis
sys.path.append(str(project_root))
print("Added to sys.path:", project_root)
from mixture_density_network.src.blocks import MixtureDensityNetwork, NoiseType
from mixture_density_network.training.train import LitModel


## Device Conifguration
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


## Load Data and Model
interests = ["DrySeasonDeficit", "ResinPi", "TotalP"]
base_dir_name = "batch32_epochs100_lr0.0004_weightdecay1e-06_beta10.9_beta20.95_dropout0.1_numcomponents15_hidden64_rarity4"

# Split by underscores
parts = base_dir_name.split("_")

# Extract parameters
params = {}
for part in parts:
    if part.startswith("batch"): params["batch_size"] = int(part.replace("batch", ""))
    elif part.startswith("epochs"): params["epochs"] = int(part.replace("epochs", ""))
    elif part.startswith("lr"): params["learn_rate"] = float(part.replace("lr", ""))
    elif part.startswith("weightdecay"): params["weight_decay"] = float(part.replace("weightdecay", ""))
    elif part.startswith("beta1"): params["beta1"] = float(part.replace("beta1", ""))
    elif part.startswith("beta2"): params["beta2"] = float(part.replace("beta2", ""))
    elif part.startswith("dropout"): params["dropout"] = float(part.replace("dropout", ""))
    elif part.startswith("numcomponents"): params["num_components"] = int(part.replace("numcomponents", ""))
    elif part.startswith("hidden"): params["hidden_size"] = int(part.replace("hidden", ""))
    elif part.startswith("rarity"): params["rarity_threshold"] = int(part.replace("rarity", ""))
        
# Default rarity_threshold to None if not found
params["rarity_threshold"] = params.get("rarity_threshold", None)

print(params)


interests_data = {}
log_interests = {"ResinPi", "TotalP"}

for interest in interests:    

    use_log = interest in log_interests

    # Build directory name conditionally
    current_dir_name = base_dir_name + ("_log" if use_log else "")

    # Define directories
    data_dir = project_root / "training_output" / interest / current_dir_name
    output_dir = project_root / "analysis" / "figures"
    print(f"{interest}: using {'log' if use_log else 'raw'} data")

    # Pick the checkpoint with the lowest validation loss
    checkpoint_files = list((data_dir / "checkpoints").glob("checkpoint_epoch=*_val_loss=*.ckpt"))
    best_checkpoint = min(checkpoint_files, key=lambda p: float(p.name.split("val_loss=")[-1].replace(".ckpt", "").split("_")[0]))

    # Extract best epoch 
    best_epoch = int(best_checkpoint.name.split("checkpoint_epoch=")[-1].split("_")[0])
    print(f"Best epoch for {interest}: {best_epoch}")

    # Load dataset
    rarity_suffix = f"_rarity{params['rarity_threshold']}" if params["rarity_threshold"] else ""
    if use_log:
        dataset_path = project_root / "data" / "processed_datasets" / f"dataset_{interest}{rarity_suffix}_log.pt"
    else:
        dataset_path = project_root / "data" / "processed_datasets" / f"dataset_{interest}{rarity_suffix}.pt"
    dataset = torch.load(dataset_path)
    X_train = dataset["train"]["embedding"].to(device)
    y_train = dataset["train"][interest].to(device)
    X_val = dataset["val"]["embedding"].to(device)
    y_val = dataset["val"][interest].to(device)
    X_test = dataset["test"]["embedding"].to(device)
    y_test = dataset["test"][interest].to(device)

    # Build train/test DataFrames 
    train_df = pd.DataFrame({
        'FEATURE_ID': dataset['train']['FEATURE_ID'],
        interest: dataset['train'][interest].cpu().numpy(),
        'embedding': list(dataset['train']['embedding'].cpu().numpy())
    })

    test_df = pd.DataFrame({
        'FEATURE_ID': dataset['test']['FEATURE_ID'],
        interest: dataset['test'][interest].cpu().numpy(),
        'embedding': list(dataset['test']['embedding'].cpu().numpy())
    })

    # Load model
    args = SimpleNamespace(
        data_file=str(dataset_path),
        lr=params["learn_rate"],
        weight_decay=params["weight_decay"],
        batch_size=params["batch_size"],
        epochs=params["epochs"],
        beta1=params["beta1"],
        beta2=params["beta2"],
        dropout=params["dropout"],
        num_components=params["num_components"],
        hidden_dim=params["hidden_size"],
    )
    model_module = LitModel.load_from_checkpoint(checkpoint_path=str(best_checkpoint), args=args, map_location=device,)
    model = model_module.model.to(device)
    model.eval()

    # Generate predictions
    with torch.no_grad():
        y_hat = model.sample(X_test)

    y_mean = torch.tensor(train_df[interest].values, dtype=torch.float32).mean().item()
    y_std = torch.tensor(train_df[interest].values, dtype=torch.float32).std().item()
    y_hat_unnormalized = y_hat * y_std + y_mean

    y_test_unsq = y_test.unsqueeze(1)

    # Rarity distribution
    all_splits = []
    for split in ['train', 'val', 'test']:
        df_split = pd.DataFrame({'FEATURE_ID': dataset[split]['FEATURE_ID'].numpy()})
        all_splits.append(df_split)
    all_df = pd.concat(all_splits, ignore_index=True)
    feature_counts = all_df['FEATURE_ID'].value_counts().reset_index()
    feature_counts.columns = ['FEATURE_ID', 'num_plots']
    rarity_threshold = 4
    feature_counts['rarity'] = ['Rare' if n <= rarity_threshold else 'Majority' for n in feature_counts['num_plots']]

    # Store everything
    interests_data[interest] = {
        "dataset": dataset,
        "best_epoch": best_epoch,
        "train_df": train_df,
        "test_df": test_df,
        "X_test": X_test,
        "y_test": y_test_unsq,
        "y_hat": y_hat_unnormalized,
        "feature_counts": feature_counts,
        "output_dir": output_dir,
        "model": model,
    }


normal_names = {
    "DrySeasonDeficit": "Dry-Season Moisture",
    "ResinPi": "Log Resin Phosphorus",
    "TotalP": "Log Total Phosphorus",
}


def darken_color(color, amount=0.4):
    """
    Darken the given color by multiplying (1-luminosity) by amount
    """
    c = mcolors.to_rgb(color)
    return tuple(max(min(comp * amount, 1), 0)for comp in c)


## Plot Model Predictions
def get_predictions(embedding, num_samples, model, device):
    emb = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).repeat(num_samples, 1).to(device)
    with torch.no_grad():
        preds = model.sample(emb).cpu().numpy()
    return preds.flatten()


# Plotting for each feature
def plot_feature_actual_vs_predicted(actual_values, pred_vals, feature_id, interest, ax, xmin, xmax, ymax, kde_bw):
    """Plot actual values as vertical lines with horizontal jitter for duplicates,
    and KDE of predictions for a single FEATURE_ID."""

    # Rug for actual values
    jitter = (np.random.rand(len(actual_values)) - 0.5) * 0.01
    sns.rugplot(actual_values + jitter, ax=ax, color='tab:blue', linewidth=2.5, height=1, zorder=1)

    # KDE for predicted values
    x_vals = np.linspace(xmin, xmax, 500)
    kde = gaussian_kde(pred_vals, bw_method=kde_bw) 
    kde_vals = kde(x_vals)

    ax.fill_between(x_vals, kde_vals, color='tab:orange', alpha=0.4, zorder=2)
    ax.plot(x_vals, kde_vals, color=darken_color('tab:orange', amount=0.5), linewidth=1, zorder=2)

    # Set y-axis based on actual KDE values
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0, ymax)

    # Labels, title, and styling
    ax.set_xlabel(normal_names.get(interest, interest), fontsize=15, labelpad=8)
    ax.set_ylabel('Density', fontsize=15, labelpad=8)
    ax.set_title(f'Feature ID {feature_id}', fontsize=14)
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(True)

    # Only show legend in the top right plot
    if ax.get_subplotspec().rowspan.start == 0 and ax.get_subplotspec().colspan.start == 2:
        rug_proxy = mlines.Line2D([], [], color='tab:blue', linewidth=2.5, linestyle='-', label='Actual')
        kde_proxy = mpatches.Patch(facecolor=mcolors.to_rgba('tab:orange', 0.4), edgecolor=darken_color('tab:orange', amount=0.6), label='Predicted')
        handles, labels = ax.get_legend_handles_labels()
        handles.append(rug_proxy)
        handles.append(kde_proxy)
        ax.legend(handles=handles, fontsize=13)


rows = 20
cols = 9
plots_per_page = rows * cols  
sample = 100000

for interest in interests_data:
    data = interests_data[interest]
    model = data["model"]
    test_df = data["test_df"]
    train_df = data["train_df"]

    ## Compute number of pages

    # Keep only features with exactly 4 actual observations
    feature_counts = test_df.groupby("FEATURE_ID").size()
    all_features = feature_counts[feature_counts == 4].index.to_numpy()

    # Compute mean observed value per feature and sort
    feature_means = test_df.groupby("FEATURE_ID")[interest].mean()
    all_features = feature_means.loc[all_features].sort_values().index.to_numpy()

    num_features = len(all_features)
    num_pages = math.ceil(num_features / plots_per_page)
    print(f"{interest}: {num_features} features = {num_pages} pages")

    ## Compute global shared axes across all features for this interest
    all_actual = []
    all_pred = []

    y_mean = train_df[interest].mean()
    y_std = train_df[interest].std()

    for feature_id in all_features:
        fdata = test_df[test_df["FEATURE_ID"] == feature_id]

        # Actual
        actual_vals = fdata[interest].values
        all_actual.append(actual_vals)

        # Predicted
        embedding = fdata["embedding"].iloc[0]
        preds = get_predictions(embedding, num_samples=sample, model=model, device=device)

        pred_vals = preds * y_std + y_mean
        all_pred.append(pred_vals)

    all_actual_flat = np.concatenate(all_actual)
    all_pred_flat = np.concatenate(all_pred)

    xmin = min(all_actual_flat.min(), all_pred_flat.min())
    xmax = max(all_actual_flat.max(), all_pred_flat.max())
    print(f"{interest}: global xmin={xmin:.3f}, xmax={xmax:.3f}")

    # Compute shared global KDE bandwidth from total distribution matrix
    global_kde = gaussian_kde(all_pred_flat)
    global_bw = global_kde.factor * 0.8

    # Shared y-limit across all features enforcing the global bandwidth constraint
    x_vals = np.linspace(xmin, xmax, 500)
    all_kde_max = []
    
    # Iterate through individual pre-split lists safely instead of reshaping flat matrices
    for pred_vals in all_pred:
        downsampled_preds = np.random.choice(pred_vals, size=min(2000, len(pred_vals)), replace=False)
        kde = gaussian_kde(downsampled_preds, bw_method=global_bw)
        all_kde_max.append(kde(x_vals).max())
        
    # Average-based y-axis scaling instead of max-based
    mean_peak = np.mean(all_kde_max) if all_kde_max else 1.0
    # Add some padding
    ymax = mean_peak * 1.3
    print(f"{interest}: mean peak={mean_peak:.3f}, ymax={ymax:.3f}")

    # Create PDF for all features
    final_output_dir = project_root / "analysis" / "figures"
    final_output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = final_output_dir / f"rare_features_4_observations_{interest}_{sample}.pdf"

    with PdfPages(pdf_path) as pdf:
        # Loop over pages
        for page in range(num_pages):
            start = page * plots_per_page
            end = min(start + plots_per_page, num_features)
            page_features = all_features[start:end]

            fig, axes = plt.subplots(rows, cols, figsize=(24, 44), constrained_layout=True)

            # Share axes on page 
            for r in range(rows):
                for c in range(cols):
                    if r == 0 and c == 0:
                        continue
                    axes[r, c].sharex(axes[0, 0])
                    axes[r, c].sharey(axes[0, 0])

            axes = axes.flatten()

            # Plot features on this page
            for ax_idx, feature_id in enumerate(page_features):
                ax = axes[ax_idx]

                # Fetch indices relative to overall tracking arrays
                global_idx = start + ax_idx
                actual_vals = all_actual[global_idx]
                pred_vals = all_pred[global_idx]

                plot_feature_actual_vs_predicted(actual_vals, pred_vals, feature_id, interest, ax=ax, xmin=xmin, xmax=xmax, ymax=ymax, kde_bw=global_bw)

                ax.set_xlabel("")
                ax.set_ylabel("")
                if ax.get_legend() is not None:
                    ax.get_legend().remove()

            # Turn off empty axes
            for ax_idx in range(len(page_features), len(axes)):
                axes[ax_idx].axis("off")

            pdf.savefig(fig)
            plt.close(fig)