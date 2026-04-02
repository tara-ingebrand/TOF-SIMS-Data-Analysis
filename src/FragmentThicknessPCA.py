import os
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#Folder path containing _nmvalues.txt files
folder_path = '../data/TOF-SIMS/Na LHCE Diluent Project/3D Fragment Map/Baseline Anode'

image_data = {}

# Loop through and read all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".txt") and "nmvalues" in filename:  # Only process .txt files
        # Extract the fragment label from the filename
        base_name = os.path.splitext(filename)[0]
        fragment_label = base_name.replace("_nmvalues", "")  # Extract text after " - "

        # Full path to the file
        file_path = os.path.join(folder_path, filename)
        # Load the file, ignoring header lines starting with '#'
        data = np.loadtxt(file_path, comments="#")
        # Store the 3D intensity array in the dictionary
        image_data[fragment_label] = data

# size = 237 x 237, from sliding bin size of 20 in FragmentThicknessMaps code

# convert to matrix to do PCA
#each x, y point now is associated with the thickness value from each fragment
labels = image_data.keys()
image_data = np.stack(list(image_data.values()), axis=-1)

# do PCA
pca = PCA(n_components=1, random_state=42)
latent_space = pca.fit_transform(image_data)  # note transpose to match your tsne version

#plot PCA scores from the origin (so that scores are all positive)
mean_in_pca_space = (pca.mean_).dot(pca.components_.T)
latent_space = latent_space + mean_in_pca_space

explained_variance = pca.explained_variance_ratio_

# Print the result
print("Explained variance ratio for each component:", explained_variance)
print(f"Total variance explained by PC1: {explained_variance[0]*100:.2f}%")


# Extract loadings for each component
loadings = pca.components_  # shape: (n_components, n_fragments)

# Convert to a labeled dictionary
fragment_labels = list(labels)
for i, component in enumerate(loadings):
    print(f"\nPCA Component {i+1} Loadings:")
    for frag, value in zip(fragment_labels, component):
        print(f"  {frag:15s} {value:.3f}")

#simple metric for each fragment's relative contribution to a component can be
#represented by the normalized squared loadings
loadings = pca.components_[0]
relative_contrib = (loadings**2) / np.sum(loadings**2)
for frag, contrib in zip(fragment_labels, relative_contrib):
    print(f"{frag:15s} contributes {contrib*100:.2f}% to PC1")


# reshape back to correct size
latent_space = latent_space.flatten()
latent_space = latent_space.reshape(237,237)


fig, ax = plt.subplots(figsize=(6, 6))

# Define physical axes in µm
size_um = 100  # total size in µm
num_pixels = latent_space.shape[0]  # 237
pixel_size = size_um / num_pixels   # µm per pixel

plt.imshow(np.flipud(np.fliplr(latent_space)), cmap='inferno', origin='lower',
           extent=[0, size_um, 0, size_um], vmin=5, vmax=43)  # set physical axes


cbar = plt.colorbar(fraction=0.046, pad=0.04)
cbar.set_label('PC Score', fontsize=20, fontname='arial', fontweight='bold')

cbar.ax.tick_params(labelsize=18)

for label in cbar.ax.get_yticklabels():
    label.set_fontname('arial')
    label.set_fontweight('bold')

plt.xlabel('X (µm)', fontdict={'fontsize': 24, 'fontname': 'arial', 'fontweight': 'bold'})
plt.ylabel('Y (µm)', fontdict={'fontsize': 24, 'fontname': 'arial', 'fontweight': 'bold'})
ax.tick_params(axis="x", direction="in", labelsize=18)
ax.tick_params(axis="y", direction="in", labelsize=18)
for label in ax.get_xticklabels():
    label.set_fontweight('bold')
for label in ax.get_yticklabels():
    label.set_fontweight('bold')
for label in ax.get_yticklabels():
    label.set_fontweight('bold')

plt.tight_layout()
plt.show()