# Using Mixture Density Networks to Predict Plot-level Environmental Factors from Tree Secondary Metabolites

-----

## About the project:

### Abstract

Mixture density networks (MDNs) offer a flexible approach for predicting variables with multimodal distributions. Traditional regression models produce single-value predictions, which may fail to capture the full variability of widely distributed variables. We evaluated the performance of an MDN model on a Panama metabolomics dataset, in which each metabolite occurs across multiple forest plots, resulting in environmental variables with multiple associated values. The MDN recovered the overall range and broad shape of environmental distributions, although feature-specific predictions often converged toward shared distributional patterns. These results suggest that mixture density networks can capture coarse environmental structure from metabolomic representations, while also highlighting challenges associated with limited environmental variation and sparse observations. 


-----

### Significance



**Keywords:** Mixture density network | DreaMS embeddings | Plant secondary metabolites

-----

## Reproducibility

### 1. Environment

```bash
# clone this repository
git clone https://github.com/kellyktvt/Panama_forest_analysis.git
cd Panama_forest_analysis

# create a clean virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# upgrade pip
pip install --upgrade pip

# install dependencies
pip install -r requirements.txt
```

### 2. Jupyter kernel

```bash
python -m ipykernel install --user --name panama-mdn --display-name "Panama MDN"
```

Select the Panama MDN kernel when running the notebooks.

### 3. DreaMS embeddings
We have provided the processed datasets for the three environmental variables that we analyzed, so you don't need to generate the embeddings again. But if you would like to do so, the steps are listed below.

We suggest setting up a separate environment for DreaMS, as it is installed from source. The instructions for installing DreaMS can be found here: https://dreams-docs.readthedocs.io/en/latest/.

```bash
# Generate the embeddings using this script
data/generate_embeddings.py

python -u data/generate_embeddings.py PanamaProp_f100_20250105_iimn_gnps > embeddings.log 2>&1
```

### 3. Data wrangling and processing
This goes through how we created the processed datasets that go into the model. 

```bash
# We wrangled the data with this notebook to create 'Panama_gnps_dreams_plot_melted_info.csv'
data/metabolites_data_wrangling_panama.ipynb

# For each variable, we ran this script twice to create its processed datasets. One for all features and another for features found in 4 or less plots.
mixture_density_network/data_processing/preprocessing.py

# Example command creating the rare set for dry season deficit
python -u mixture_density_network/data_processing/preprocessing.py -i1 DrySeasonDeficit -r 4 > log_preprocess_DrySeasonDeficit_rarity4.log 2>&1
```

### 5. Training the mixture density network
In the training script, the argument defaults have been set to what we used for our analysis, but you can change the defaults or override them for your own exploration. There is also the option to tune the hyperparameters if you would like.

```bash
# For each processed dataset, we ran this script to train the model on the dataset.
mixture_density_network/training/train.py

# Example command training the rare dry season deficit dataset
python -u -m mixture_density_network.training.train -d data/processed_datasets/dataset_DrySeasonDeficit_rarity4.pt -o output/Panama/ --cuda_num 0 > log_train_DrySeasonDeficit_rarity4.log 2>&1
```

### 6. Analysis
These are the notebooks and scripts that were used to create the figures found in the paper.

```bash
# This creates the map of the Panama forest plots that the leaf samples were collected in.
analysis/map_actual.ipynb

# This creates the figures for the datasets of all features.
analysis/plots_Panama_1interest_lightning_log_all.ipynb

# This creates the figures comparing the effects of changing the number of components.
analysis/plots_Panama_1interest_lightning_log_rarity_components.ipynb

# This creates the figures for the datasets of rare features.
analysis/plots_Panama_1interest_lightning_log_rarity.ipynb

# This creates the PDFs displaying distributions for many random features for the datasets of all features.
analysis/plots_random_features.py

# This creates the PDFs displaying distributions for all rare features with 4 observations.
analysis/plots_rare_features.py
```

#### Footnote
If you encounter any issues or missing files or scripts, please open an issue and we will fix it.

