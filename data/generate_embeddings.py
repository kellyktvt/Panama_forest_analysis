"""
Generate DreaMS embeddings from an MGF file and store them in an HDF5 file.

Command:
python generate_dreams_embeddings.py <stem>
"""

import sys
from pathlib import Path
from dreams.api import dreams_embeddings
import numpy as np
import h5py

stem = sys.argv[1]

data_dir = Path("data/data_panama")
mgf_path = data_dir / f"{stem}.mgf"
hdf5_path = data_dir / f"{stem}.hdf5"
# csv_path = data_dir / f"{stem}_embeddings.csv"

# Generate embeddings 
embs = dreams_embeddings(str(mgf_path))

# Save embeddings to a csv file
# np.save txt(csv_path, embs, delimiter=",")

# Add embeddings to hdf5 file
with h5py.File(hdf5_path, "a") as f:
    # Overwrite if the dataset already exists
    if 'DreaMS_embedding' in f:
        del f['DreaMS_embedding']
    f.create_dataset('DreaMS_embedding', data=embs)