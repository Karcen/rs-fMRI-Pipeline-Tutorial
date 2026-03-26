"""
3D Brain Network Visualization (AAL166 Atlas)
- T1 MNI152 Aligned
- Real Functional Connectivity Matrix
- Strongest Pathways Only
"""

import numpy as np
import matplotlib.pyplot as plt
from nilearn import plotting
import nibabel as nib
import os
import warnings
warnings.filterwarnings('ignore')

# ==============================================
# FILE PATHS (MATCH YOUR PIPELINE)
# ==============================================
ATLAS_FILE       = "/Users/karcenzheng/Downloads/Jiacheng_Zheng的3D大脑/AAL3v2_for_SPM12/AAL3v1.nii"
T1_TEMPLATE_FILE = "/Users/karcenzheng/Downloads/Jiacheng_Zheng的3D大脑/mni_icbm152_nlin_sym_09a_nifti/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a.nii.gz"
MATRIX_FILE      = "/Users/karcenzheng/Downloads/Jiacheng_Zheng的3D大脑/brain_pipeline_outputs/fc_matrix.npy"

# ==============================================
# PLOTTING SETTINGS
# ==============================================
OUTPUT_IMAGE     = "3D_brain_network_AAL166_strongest.png"
NODE_SIZE        = 100
BRAIN_OPACITY    = 0.18
NUM_TOP_CONN     = 8  # Show TOP 8 strongest connections

# ==============================================
# LOAD ATLAS & MATRIX
# ==============================================
atlas_img = nib.load(ATLAS_FILE)
coords    = plotting.find_parcellation_cut_coords(atlas_img)

conn_matrix = np.load(MATRIX_FILE)
conn_matrix = (conn_matrix + conn_matrix.T) / 2
np.fill_diagonal(conn_matrix, 0)

# ==============================================
# KEEP ONLY STRONG CONNECTIONS
# ==============================================
def keep_strongest_connections(mat, n_keep):
    mat = mat.copy()
    upper = np.triu(mat)
    vals = upper[upper > 0]
    threshold = np.percentile(vals, 100 - (n_keep * 100 / len(vals)))
    mat[mat < threshold] = 0
    return mat

conn_matrix = keep_strongest_connections(conn_matrix, NUM_TOP_CONN)

# ==============================================
# NODE COLORS
# ==============================================
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
node_colors = [colors[i % 4] for i in range(len(coords))]

# ==============================================
# PLOT 3D BRAIN NETWORK
# ==============================================
fig, ax = plt.subplots(figsize=(12, 10), facecolor='white')

# Plot T1 template background
plotting.plot_glass_brain(
    T1_TEMPLATE_FILE,
    display_mode='ortho',
    axes=ax,
    alpha=BRAIN_OPACITY
)

# Plot brain network
plotting.plot_connectome(
    conn_matrix,
    coords,
    node_color=node_colors,
    node_size=NODE_SIZE,
    axes=ax,
    colorbar=True,
    edge_threshold=None,
    edge_vmin=-1, edge_vmax=1
)

plt.savefig(OUTPUT_IMAGE, dpi=300, bbox_inches='tight')
plt.close()

print("✅ 3D Brain Network Saved Successfully!")
print(f"✅ Atlas: AAL3v1 (166 regions)")
print(f"✅ Matrix: fc_matrix.npy")
print(f"✅ Output: {OUTPUT_IMAGE}")