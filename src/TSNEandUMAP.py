import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap


############## Inputs ##############


#Folder path containing 3D depth profiles
folder_path = 'data/Baseline HC Anode'

#Check that folder path exists
assert os.path.exists(folder_path), (
    f"Directory does not exist!\n"
    f"You asked for: {folder_path}\n"
    f"Your current working directory is: {os.getcwd()}\n"
    f"Therefore the absolute path is: {os.path.abspath(folder_path)},"
    f"but this does not exist!\n"
    f"If your dataset exists, you probably need to change your working directory, or change the dataset path."
)

# Name of the cache file to save/load latent space results
# This saves time so that the latent space isn't calculated each time
cache_name = "latent_cache"

# Analysis method
method = "TSNE"   # <-- choose "TSNE", or "UMAP"

# Check that the chosen dimensionality reduction method is valid
if method not in ["TSNE", "UMAP"]:
    raise ValueError(
        f"Invalid method '{method}'! Please choose either 'TSNE' or 'UMAP'."
    )

# number of bins in x,y,z directions
bx, by, bz = 16, 16, 10


# Full path to cache file (saves computational time for repeated runs)
cache_path = os.path.join(folder_path, cache_name + "_" + method + ".npy")


############## Load fragment maps ##############


# This section reads all the fragment files in the folder and stores them in a dictionary
image_data = {}
# Iterate through all .txt files in teh specified folder, reconsutrct each fragment's
# 3D intensity volume, and cache it as a binary (.npy) file for faster future loading.
# Fragment labels are extracted from filenames and used as keys for storing volumes.
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"): # process only .txt files
        base_name = os.path.splitext(filename)[0]
        fragment_label = base_name.split(" - ")[-1] # extract fragment names

        file_path = os.path.join(folder_path, filename)
        np_file_path = file_path.replace(".txt", ".npy") #preprocessed cache file

        # It .npy exists (already processed), load it directly for speed
        if os.path.exists(np_file_path):
            data = np.load(np_file_path)
        # Otherwise, load raw text data
        else:
            data = np.loadtxt(file_path, comments="#")
            # Determine the 3D shape from the last coordinate in the file
            x, y, z = int(data[-1, 0] + 1), int(data[-1, 1] + 1), int(data[-1, 2] + 1)
            # Reshape intesntiy values into a 3D array
            data = data[:, 3].reshape((x, y, z), order="F")
            # Save as .npy for faster loading next time
            np.save(np_file_path, data)

        # Store the 3D fragment data in a dictionary
        image_data[fragment_label] = data


############## Stack & Bin Data ##############


# Convert dictionary to list of arrays for processing
intensity_names, intensity_values = [], []
for name, data in image_data.items():
    intensity_names.append(name)
    intensity_values.append(data)

# make sure all fragments have same shape by cropping to smallest dimensions
min_shape = np.min([a.shape for a in intensity_values], axis=0)
intensity_values = [a[tuple(slice(0, m) for m in min_shape)] for a in intensity_values]

# Stack into a single 4D array: (num_fragments, x, y, z)
intensity_values = np.stack(intensity_values)
x, y, z = intensity_values.shape[1:]

# Check that the bin sizes evenly divide the data
if (x % bx != 0) or (y % by != 0) or (z % bz != 0):
    raise ValueError(
        f"Invalid bin size!\n"
        f"Your data dimensions are: x={x}, y={y}, z={z}\n"
        f"Your chosen bin sizes are: bx={bx}, by={by}, bz={bz}\n"
        f"Each bin size must evenly divide the corresponding data dimension.\n"
        f"Please choose bin sizes that are factors of the data dimensions."
    )

# Spatial Binning/Down sampling for multivariate stability
# The 3D volumes are reshaped into blocks of size (bin_x, bin_y, bin_z),
# and the mean intensity is taken within each block
assert x % bx == 0 and y % by == 0 and z % bz == 0
intensity_values = intensity_values.reshape(
    intensity_values.shape[0],
    x // bx, bx,
    y // by, by,
    z // bz, bz
).mean(axis=(2, 4, 6))

# Store new shape
new_x, new_y, new_z = intensity_values.shape[1:]

# flatten for dimensionality reduction
# Convert 3D volumes into 1D vectors per fragment
flat_data = intensity_values.reshape(intensity_values.shape[0], -1)


############## Dimensionality Reduction ##############


# Compute latent space (2D representations) with TSNE or UMAP
# If cached result exists, load it to save time
if os.path.exists(cache_path):
    latent_space = np.load(cache_path)
else:

    if method == "TSNE":
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        # Transpose for same orientation as PCA
        latent_space = reducer.fit_transform(flat_data.T).T

    elif method == "UMAP":

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=15,      # controls how local/global structure is preserved
            min_dist=0.1,        # controls tightness of clusters
            metric='euclidean',
            random_state=42
        )
        # Transpose to match PCA/TSNE orientation
        latent_space = reducer.fit_transform(flat_data.T).T


    # Save latent space to cache for future use
    np.save(cache_path, latent_space)


############## Reshape & Color by Depth ##############


# Convert latent space back to 3D for visualization
latent_space = latent_space.reshape(2, new_x, new_y, new_z)
X, Y, Z = np.meshgrid(np.arange(new_x), np.arange(new_y), np.arange(new_z), indexing='ij')

tsne_x = latent_space[0].flatten() # x-coordinates for scatter
tsne_y = latent_space[1].flatten() # y-coordinates for scatter


# Color by Z-depth
Z_flat = Z.flatten().astype(float)
Z_norm = (Z_flat - Z_flat.min()) / (Z_flat.max() - Z_flat.min())
Z_norm = 1 - Z_norm  # invert for visualization
colors = plt.cm.coolwarm(Z_norm) #colormap choice


############## Plot ##############


fig, ax = plt.subplots()
sc = ax.scatter(tsne_x, tsne_y, c=colors, s=10)
ax.set_xlabel(f"{method} 1", fontdict={'fontsize': 18, 'fontweight': 'bold'}, fontname='arial')
ax.set_ylabel(f"{method} 2", fontdict={'fontsize': 18, 'fontweight': 'bold'}, fontname='arial')
ax.tick_params(axis="x", direction="in", labelsize=16)
ax.tick_params(axis="y", direction="in", labelsize=16)

for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('bold')
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(1.5)

# remove tick labels
ax.set_xticks([])
ax.set_yticks([])

plt.tight_layout()
plt.show()