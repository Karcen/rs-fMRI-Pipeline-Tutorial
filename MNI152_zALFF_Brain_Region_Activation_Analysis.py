"""
Resting fMRI zALFF Analysis
- MNI152 aligned
- Output TOP 10 most active regions (AAL3)
- Output coordinates + table
"""

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import image, masking
from nilearn.maskers import NiftiLabelsMasker
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ==============================================
# 👉 请确认你的文件路径
# ==============================================
BOLD_FILE    = "你的地址/BOLD_WIP_MB2_BOLD_NEW_401.nii.gz"
AAL_ATLAS    = "你的地址/AAL3v2_for_SPM12/AAL3v1.nii"
AAL_LABELS   = "你的地址/AAL3v2_for_SPM12/ROI_MNI_V7_vol.txt"
MNI_T1       = "你的地址/mni_icbm152_nlin_sym_09a_nifti/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a.nii.gz"
MNI_MASK     = "你的地址/mni_icbm152_nlin_sym_09a_nifti/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a_mask.nii.gz"

TR = 1.0
OUTPUT = Path("active_brain_results")
OUTPUT.mkdir(exist_ok=True)

# ==============================================
# Load data & RESAMPLE TO MNI SPACE
# ==============================================
print("Loading BOLD...")
img = nib.load(BOLD_FILE)

# ✅ 关键修复：把 BOLD 重采样到 MNI 空间（解决 affine 不匹配）
print("Resampling to MNI152 space...")
mni_img = image.resample_to_img(
    img, MNI_MASK,
    interpolation='linear',
    force_resample=True
)

# ==============================================
# Compute zALFF
# ==============================================
print("Computing zALFF...")
clean = image.clean_img(
    mni_img, detrend=True, standardize=False,
    low_pass=0.1, high_pass=0.01, t_r=TR, mask_img=MNI_MASK
)

alff = image.math_img("np.std(img, axis=3)", img=clean)
mask_data = nib.load(MNI_MASK).get_fdata() > 0.5
alff_vals = alff.get_fdata()[mask_data]
mu, sd = np.mean(alff_vals), np.std(alff_vals)
zalff = image.math_img(f"(img - {mu})/{sd}", img=alff)

# ==============================================
# Extract AAL3 regions
# ==============================================
print("Extracting AAL3 brain regions...")
with open(AAL_LABELS, 'r', 'utf-8', errors='ignore') as f:
    labels = [line.strip() for line in f if line.strip()]

masker = NiftiLabelsMasker(labels_img=AAL_ATLAS, standardize=False)
roi_values = masker.fit_transform(zalff)[0]

# ==============================================
# TOP 10 ACTIVE REGIONS
# ==============================================
df = pd.DataFrame({
    "Region_ID": range(1, len(roi_values)+1),
    "Region_Name": labels[:len(roi_values)],
    "zALFF": roi_values
})
df = df.sort_values(by="zALFF", ascending=False).reset_index(drop=True)
top10 = df.head(10)

# ==============================================
# MNI Coordinates
# ==============================================
atlas_data = nib.load(AAL_ATLAS).get_fdata()
affine = nib.load(AAL_ATLAS).affine

def get_roi_center(roi_id):
    coords = np.argwhere(atlas_data == roi_id)
    if len(coords) == 0:
        return [0,0,0]
    center = np.median(coords, axis=0)
    xyz = nib.affines.apply_affine(affine, center)
    return np.round(xyz, 2).tolist()

top10["MNI_X"], top10["MNI_Y"], top10["MNI_Z"] = zip(*top10["Region_ID"].apply(get_roi_center))

# ==============================================
# Save results
# ==============================================
top10.to_csv(OUTPUT / "TOP10_active_regions.csv", index=False)
top10[["Region_Name", "MNI_X", "MNI_Y", "MNI_Z", "zALFF"]].to_csv(OUTPUT / "top10_coordinates.csv", index=False)
df["MNI_X"], df["MNI_Y"], df["MNI_Z"] = zip(*df["Region_ID"].apply(get_roi_center))
df.to_csv(OUTPUT / "all_brain_regions_activity.csv", index=False)

# ==============================================
# Print
# ==============================================
print("\n" + "="*80)
print("            TOP 10 MOST ACTIVE BRAIN REGIONS (RESTING STATE)")
print("="*80)
for i, row in top10.iterrows():
    print(f"{i+1:2d}. {row.Region_Name:<40} z={row.zALFF:6.3f}  MNI={[row.MNI_X, row.MNI_Y, row.MNI_Z]}")

print("\n✅ ALL FILES SAVED TO: active_brain_results/")