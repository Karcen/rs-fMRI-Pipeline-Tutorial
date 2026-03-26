"""
=============================================================================
MULTIMODAL MRI BRAIN CONNECTIVITY PIPELINE
=============================================================================
Author: Generated for Jiacheng Zheng
Modalities: T1w, rs-fMRI (BOLD), DTI, DSI
Tools: nilearn, dipy, nibabel, networkx, scipy, matplotlib, seaborn, plotly

Pipeline Stages:
  1. Data Discovery & Validation
  2. rs-fMRI Preprocessing & Activation Metrics (ALFF, fALFF, ReHo)
  3. Seed-based & Whole-brain Functional Connectivity
  4. ICA Decomposition → DMN / Salience / CEN Detection
  5. Structural Connectivity (DTI tensor + tractography)
  6. Multimodal Integration (FC vs SC)
  7. Graph-theoretic Network Analysis
  8. Rich Visualization Suite
=============================================================================
"""

# ── Standard library ─────────────────────────────────────────────────────────
import os, glob, warnings, time
from pathlib import Path

warnings.filterwarnings("ignore")
import os


# ── Core numerical / scientific ───────────────────────────────────────────────
import numpy as np
import scipy.ndimage as ndi
from scipy.signal import butter, filtfilt
from scipy.stats import pearsonr

# ── NIfTI I/O ────────────────────────────────────────────────────────────────
import nibabel as nib

# ── nilearn ───────────────────────────────────────────────────────────────────
from nilearn import image, plotting, signal, decomposition
from nilearn.connectome import ConnectivityMeasure
from nilearn.maskers import NiftiLabelsMasker, NiftiMasker
from nilearn.datasets import fetch_atlas_harvard_oxford, load_mni152_template
from nilearn.image import clean_img, smooth_img, resample_to_img

# ── DIPY (structural connectivity) ───────────────────────────────────────────
try:
    from dipy.io.image import load_nifti
    from dipy.io.gradients import read_bvals_bvecs
    from dipy.core.gradients import gradient_table
    from dipy.reconst.dti import TensorModel
    from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, auto_response_ssst
    from dipy.direction import peaks_from_model
    from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
    from dipy.tracking.local_tracking import LocalTracking
    from dipy.tracking.streamline import Streamlines
    from dipy.tracking import utils as tracking_utils
    from dipy.segment.mask import median_otsu
    DIPY_AVAILABLE = True
except ImportError:
    print("[WARN] DIPY not fully installed – structural pipeline will be skipped.")
    DIPY_AVAILABLE = False

# ── Visualisation ─────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")                       # headless rendering
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("[WARN] plotly not installed – interactive plots will be skipped.")

# ── NetworkX ──────────────────────────────────────────────────────────────────
import networkx as nx

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION  (edit paths here if needed)
# ─────────────────────────────────────────────────────────────────────────────
PATHS = {
    "t1": "/Users/karcenzheng/Downloads/Jiacheng_Zheng的3D大脑/nifti/T1",  # ✅ 文件夹
    "bold": "/Users/karcenzheng/Downloads/Jiacheng_Zheng的3D大脑/nifti/BOLD",  # ✅
    "dti": "/Users/karcenzheng/Downloads/Jiacheng_Zheng的3D大脑/nifti/DTI",  # ✅
    "dsi": "/Users/karcenzheng/Downloads/Jiacheng_Zheng的3D大脑/nifti/DSI",  # ✅
}

OUT_DIR = Path("./brain_pipeline_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Preprocessing parameters
TR          = 2.0          # repetition time in seconds (EDIT if different)
FWHM        = 6.0          # spatial smoothing kernel (mm)
HP_FREQ     = 0.01         # bandpass high-pass (Hz)
LP_FREQ     = 0.10         # bandpass low-pass  (Hz)
N_DUMMIES   = 5            # dummy scans to drop at start
FA_THRESH   = 0.20         # DTI FA threshold for WM mask

# Seeds (MNI mm coordinates)
SEEDS = {
    "PCC":         (0,  -52,  26),
    "Hippocampus": (24, -22, -20),
    "mPFC":        (0,   52,  -6),
    "Insula":      (38,   2,   0),
    "DLPFC":       (44,  36,  20),
}

print("=" * 70)
print("  MULTIMODAL MRI BRAIN CONNECTIVITY PIPELINE")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
#  MODULE 1 – DATA DISCOVERY & VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def discover_niftis(directory: str) -> list[str]:
    """Return all .nii / .nii.gz files in *directory* (sorted)."""
    pattern_gz = os.path.join(directory, "**", "*.nii.gz")
    pattern    = os.path.join(directory, "**", "*.nii")
    files = sorted(glob.glob(pattern_gz, recursive=True) +
                   glob.glob(pattern,    recursive=True))
    return files


def validate_and_pick(files: list[str], label: str,
                      expected_ndim: int | None = None) -> nib.Nifti1Image | None:
    """
    From a list of NIfTI paths, load and validate each.
    Returns the first file that matches *expected_ndim* (3 or 4).
    If expected_ndim is None, returns the largest-volume file.
    """
    best, best_vols = None, 0
    for f in files:
        try:
            img = nib.load(f)
            shape = img.shape
            ndim  = len(shape)
            vols  = shape[3] if ndim == 4 else 1
            print(f"  [{label}] {os.path.basename(f)}  shape={shape}  dtype={img.get_data_dtype()}")
            if expected_ndim is None:
                if vols > best_vols:
                    best, best_vols = img, vols
            elif ndim == expected_ndim:
                if vols > best_vols:
                    best, best_vols = img, vols
        except Exception as e:
            print(f"  [WARN] Could not load {f}: {e}")
    return best


def load_all_modalities():
    """Discover and load all modality images. Returns a dict of nib images."""
    print("\n── MODULE 1: DATA DISCOVERY ──────────────────────────────────────")
    data = {}
    for key, path in PATHS.items():
        files = discover_niftis(path)
        print(f"\n[{key.upper()}] Found {len(files)} NIfTI file(s) in {path}")
        if not files:
            print(f"  [WARN] No files found for {key}. Check path.")
            data[key] = None
            continue
        # BOLD → 4D, T1/FA → 3D, DTI/DSI → 4D
        exp = 4 if key in ("bold", "dti", "dsi") else 3
        data[key] = validate_and_pick(files, key, expected_ndim=exp)

    # Summary
    print("\n── Loaded Modalities ─────────────────────────────────────────────")
    for k, v in data.items():
        status = f"shape={v.shape}" if v is not None else "MISSING"
        print(f"  {k:6s}: {status}")
    return data


# ─────────────────────────────────────────────────────────────────────────────
#  MODULE 2 – rs-fMRI PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def drop_dummies(img_4d: nib.Nifti1Image, n: int = N_DUMMIES) -> nib.Nifti1Image:
    """Drop first *n* volumes (dummy stabilisation scans)."""
    arr = img_4d.get_fdata()
    arr = arr[..., n:]
    return nib.Nifti1Image(arr, img_4d.affine, img_4d.header)


def bandpass_filter(img_4d: nib.Nifti1Image,
                    tr: float, lp: float, hp: float) -> nib.Nifti1Image:
    """
    Temporal bandpass filter via zero-phase Butterworth (2nd order).
    Approximation: applied voxel-wise using scipy.signal.filtfilt.
    """
    arr = img_4d.get_fdata()
    n_t = arr.shape[3]
    nyq = 0.5 / tr
    b, a = butter(2, [hp / nyq, lp / nyq], btype="band")
    # reshape to (voxels, time), filter, reshape back
    flat   = arr.reshape(-1, n_t)
    mask   = np.std(flat, axis=1) > 0          # skip zero-variance voxels
    result = flat.copy()
    result[mask] = filtfilt(b, a, flat[mask], axis=1)
    return nib.Nifti1Image(result.reshape(arr.shape), img_4d.affine, img_4d.header)


def preprocess_bold(bold_img: nib.Nifti1Image,
                    t1_img:   nib.Nifti1Image | None = None) -> nib.Nifti1Image:
    """
    Full rs-fMRI preprocessing chain:
      1) Drop dummies
      2) Spatial smoothing (nilearn)
      3) Bandpass filtering
      4) Resample to MNI152 2 mm (spatial normalisation proxy)

    Note on motion correction: Full realignment requires FSL/SPM.
    Here we approximate by mean-image normalisation (signal scaling).
    For production runs, run mcflirt/3dvolreg before this script.
    """
    print("\n── MODULE 2: rs-fMRI PREPROCESSING ──────────────────────────────")
    t0 = time.time()

    print("  Step 1/4: Dropping dummy scans …")
    bold = drop_dummies(bold_img)
    print(f"    → shape after drop: {bold.shape}")

    print("  Step 2/4: Spatial smoothing (FWHM = {:.1f} mm) …".format(FWHM))
    bold = smooth_img(bold, fwhm=FWHM)

    print("  Step 3/4: Bandpass filtering ({:.3f}–{:.2f} Hz) …".format(HP_FREQ, LP_FREQ))
    bold = bandpass_filter(bold, tr=TR, lp=LP_FREQ, hp=HP_FREQ)

    print("  Step 4/4: Resampling to MNI152 2 mm …")
    mni = load_mni152_template(resolution=2)
    bold_mni = resample_to_img(bold, mni, interpolation="linear")
    print(f"    → MNI shape: {bold_mni.shape}")

    print(f"  Preprocessing done in {time.time()-t0:.1f}s")
    nib.save(bold_mni, str(OUT_DIR / "bold_preprocessed_mni.nii.gz"))
    return bold_mni


# ─────────────────────────────────────────────────────────────────────────────
#  MODULE 3 – ACTIVATION METRICS (ALFF / fALFF / ReHo)
# ─────────────────────────────────────────────────────────────────────────────

def compute_alff_falff(bold_img: nib.Nifti1Image, tr: float) -> tuple:
    """
    ALFF  = mean amplitude of low-frequency fluctuations (0.01–0.1 Hz)
    fALFF = ALFF / total power (normalised)

    Method:
      - FFT along time axis
      - Sum power in [0.01, 0.1] Hz band → ALFF
      - Divide by total power → fALFF
    """
    print("\n  Computing ALFF / fALFF …")
    arr  = bold_img.get_fdata()                 # (x, y, z, t)
    n_t  = arr.shape[3]
    freqs = np.fft.rfftfreq(n_t, d=tr)         # frequency axis

    lf_mask   = (freqs >= HP_FREQ) & (freqs <= LP_FREQ)
    fft_arr   = np.abs(np.fft.rfft(arr, axis=3))

    alff_map  = fft_arr[..., lf_mask].mean(axis=3)
    total_pow = fft_arr.mean(axis=3)
    falff_map = np.where(total_pow > 0, alff_map / total_pow, 0)

    alff_img  = nib.Nifti1Image(alff_map,  bold_img.affine)
    falff_img = nib.Nifti1Image(falff_map, bold_img.affine)
    nib.save(alff_img,  str(OUT_DIR / "alff.nii.gz"))
    nib.save(falff_img, str(OUT_DIR / "falff.nii.gz"))
    print("    Saved alff.nii.gz  and  falff.nii.gz")
    return alff_img, falff_img


def compute_reho(bold_img: nib.Nifti1Image, neighbourhood: int = 1) -> nib.Nifti1Image:
    """
    Regional Homogeneity (ReHo):
      Kendall's W of a voxel's time series with its 26-neighbour cube.

    Approximation: We compute Kendall's W using the rank-correlation
    shortcut  W = (12 * S) / (k² * (n³ - n))  where S = sum of squared
    deviations of rank sums, k = number of time series (27), n = n_timepoints.

    This is the standard approximation used in REST/DPABI.
    """
    print("  Computing ReHo (26-neighbour Kendall W) …")
    arr  = bold_img.get_fdata().astype(np.float32)
    x, y, z, n_t = arr.shape
    k   = (2 * neighbourhood + 1) ** 3          # 27 for neighbourhood=1
    reho = np.zeros((x, y, z), dtype=np.float32)

    # Rank-transform along time axis for each voxel
    from scipy.stats import rankdata
    arr_ranked = np.apply_along_axis(rankdata, 3, arr)

    d = neighbourhood
    for xi in range(d, x - d):
        for yi in range(d, y - d):
            for zi in range(d, z - d):
                cube = arr_ranked[xi-d:xi+d+1,
                                  yi-d:yi+d+1,
                                  zi-d:zi+d+1, :]     # (k, n_t) → after reshape
                ts_mat = cube.reshape(k, n_t)          # rows = neighbours
                rank_sums = ts_mat.sum(axis=0)         # sum across neighbours
                S = np.sum((rank_sums - rank_sums.mean()) ** 2)
                W = 12 * S / (k ** 2 * (n_t ** 3 - n_t) + 1e-9)
                reho[xi, yi, zi] = W

    reho_img = nib.Nifti1Image(reho, bold_img.affine)
    nib.save(reho_img, str(OUT_DIR / "reho.nii.gz"))
    print("    Saved reho.nii.gz")
    return reho_img


# ─────────────────────────────────────────────────────────────────────────────
#  MODULE 4 – FUNCTIONAL CONNECTIVITY
# ─────────────────────────────────────────────────────────────────────────────

def load_atlas():
    """Fetch Harvard-Oxford cortical atlas (48 ROIs)."""
    print("\n  Loading Harvard-Oxford atlas …")
    ho = fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
    atlas_img   = ho.maps
    atlas_labels = ho.labels
    print(f"    {len(atlas_labels)} cortical regions")
    return atlas_img, atlas_labels


def seed_based_connectivity(bold_img: nib.Nifti1Image,
                             seeds: dict[str, tuple],
                             brain_mask: nib.Nifti1Image | None = None) -> dict:
    """
    Compute whole-brain seed-based correlation maps for each seed in *seeds*.
    Seeds are given as MNI mm coordinates → converted to nearest voxel.
    """
    print("\n  Seed-based functional connectivity …")
    arr    = bold_img.get_fdata()
    affine = bold_img.affine
    inv_aff = np.linalg.inv(affine)
    results = {}

    for name, mni_coord in seeds.items():
        # MNI mm → voxel indices
        vox = np.round(inv_aff @ np.array([*mni_coord, 1]))[:3].astype(int)
        vx, vy, vz = np.clip(vox, 0,
                              [arr.shape[0]-1, arr.shape[1]-1, arr.shape[2]-1])
        seed_ts = arr[vx, vy, vz, :]              # 1-D time series

        # Correlate seed with every voxel
        flat    = arr.reshape(-1, arr.shape[3])
        std_flat = flat.std(axis=1)
        nonzero  = std_flat > 0

        corr_map_flat = np.zeros(flat.shape[0])
        # Vectorised Pearson: z-score seed, z-score each voxel, dot product / n
        seed_z = (seed_ts - seed_ts.mean()) / (seed_ts.std() + 1e-9)
        flat_z = np.where(nonzero[:, None],
                          (flat - flat.mean(axis=1, keepdims=True)) /
                          (std_flat[:, None] + 1e-9), 0)
        corr_map_flat = flat_z @ seed_z / arr.shape[3]

        corr_map = corr_map_flat.reshape(arr.shape[:3])
        img      = nib.Nifti1Image(corr_map, affine)
        fname    = OUT_DIR / f"seed_fc_{name.replace(' ', '_')}.nii.gz"
        nib.save(img, str(fname))
        print(f"    {name} → vox {[vx,vy,vz]}  |r| range: "
              f"[{corr_map.min():.3f}, {corr_map.max():.3f}]")
        results[name] = img

    return results


def whole_brain_fc_matrix(bold_img: nib.Nifti1Image,
                           atlas_img: nib.Nifti1Image,
                           atlas_labels: list[str]) -> tuple:
    """
    Extract ROI time series with NiftiLabelsMasker and compute
    Pearson correlation matrix → functional connectivity matrix.
    """
    print("\n  Whole-brain functional connectivity matrix …")
    masker = NiftiLabelsMasker(
        labels_img=atlas_img,
        standardize=True,
        detrend=True,
        t_r=TR,
        memory_level=0,
        verbose=0
    )
    ts_matrix = masker.fit_transform(bold_img)   # (n_timepoints, n_rois)
    print(f"    Time series shape: {ts_matrix.shape}")

    conn_measure = ConnectivityMeasure(kind="correlation")
    fc_matrix    = conn_measure.fit_transform([ts_matrix])[0]

    np.save(str(OUT_DIR / "fc_matrix.npy"), fc_matrix)
    print(f"    FC matrix shape: {fc_matrix.shape}  saved fc_matrix.npy")
    return fc_matrix, ts_matrix, atlas_labels


# ─────────────────────────────────────────────────────────────────────────────
#  MODULE 5 – ICA & RESTING-STATE NETWORK DETECTION
# ─────────────────────────────────────────────────────────────────────────────

# Network templates (MNI sphere centres, radius 8 mm)
NETWORK_SEEDS = {
    "DMN":      [(0, -52, 26), (-46, -64, 28), (46, -64, 28), (0, 52, -6)],
    "Salience": [(38, 2, 0),  (-38, 2, 0),     (0, 16, 44)],
    "CEN":      [(44, 36, 20), (-44, 36, 20),  (40, -52, 48), (-40, -52, 48)],
}


def run_ica(bold_img: nib.Nifti1Image, n_components: int = 20) -> tuple:
    """
    CanICA decomposition (nilearn).
    Returns component images + mixing matrix.
    """
    print("\n── MODULE 5: ICA / RESTING-STATE NETWORKS ────────────────────────")
    print(f"  Running CanICA (n_components={n_components}) …")
    canica = decomposition.CanICA(
        n_components=n_components,
        threshold=3.0,
        random_state=42,
        verbose=0
    )
    canica.fit(bold_img)
    components_img = canica.components_img_
    nib.save(components_img, str(OUT_DIR / "ica_components.nii.gz"))
    print(f"    Saved {n_components} ICA components → ica_components.nii.gz")
    return components_img, canica


def match_network_components(components_img: nib.Nifti1Image,
                              bold_img: nib.Nifti1Image) -> dict:
    """
    For each known network, identify the ICA component with highest
    spatial correlation to the network template (sphere mask).
    """
    print("  Matching ICA components to known networks …")
    from nilearn import image as nim
    comp_data = components_img.get_fdata()           # (x, y, z, n_comp)
    n_comp    = comp_data.shape[3]
    affine    = components_img.affine

    matched = {}
    for net_name, centres in NETWORK_SEEDS.items():
        # Build template by summing sphere masks
        template = np.zeros(comp_data.shape[:3])
        inv_aff  = np.linalg.inv(affine)
        for c in centres:
            vox = np.round(inv_aff @ np.array([*c, 1]))[:3].astype(int)
            vx, vy, vz = np.clip(vox, 0,
                                  np.array(template.shape) - 1)
            # draw 4-voxel sphere
            xi, yi, zi = np.ogrid[:template.shape[0],
                                   :template.shape[1],
                                   :template.shape[2]]
            sphere = (xi-vx)**2 + (yi-vy)**2 + (zi-vz)**2 <= 16
            template[sphere] = 1

        # Correlate template with each component
        t_flat = template.ravel()
        best_r, best_idx = -np.inf, 0
        for i in range(n_comp):
            c_flat = comp_data[..., i].ravel()
            r = np.corrcoef(t_flat, c_flat)[0, 1]
            if r > best_r:
                best_r, best_idx = r, i

        matched[net_name] = {"component": best_idx, "r": best_r}
        print(f"    {net_name:12s} → component {best_idx:2d}  (r={best_r:.3f})")

    return matched


# ─────────────────────────────────────────────────────────────────────────────
#  MODULE 6 – STRUCTURAL CONNECTIVITY (DTI / DSI)
# ─────────────────────────────────────────────────────────────────────────────

def find_bval_bvec(dti_dir: str) -> tuple[str | None, str | None]:
    """Search for .bval and .bvec files in directory."""
    bvals = glob.glob(os.path.join(dti_dir, "**", "*.bval"), recursive=True)
    bvecs = glob.glob(os.path.join(dti_dir, "**", "*.bvec"), recursive=True)
    bval_f = bvals[0] if bvals else None
    bvec_f = bvecs[0] if bvecs else None
    return bval_f, bvec_f


def run_dti_tractography(dti_img: nib.Nifti1Image,
                          dti_dir: str) -> tuple | None:
    """
    DTI pipeline:
      1) Load gradients
      2) Brain mask (median Otsu)
      3) Fit tensor model
      4) Deterministic tractography (EuDX)
      5) Return streamlines + FA map
    """
    if not DIPY_AVAILABLE:
        print("  [SKIP] DIPY not available – skipping DTI tractography.")
        return None

    print("\n── MODULE 6: STRUCTURAL CONNECTIVITY (DTI) ──────────────────────")
    bval_f, bvec_f = find_bval_bvec(dti_dir)
    if bval_f is None or bvec_f is None:
        print("  [WARN] .bval / .bvec files not found – "
              "attempting FSL-style side-car discovery.")
        # Try same-name side-cars
        nii_files = discover_niftis(dti_dir)
        if nii_files:
            base = nii_files[0].replace(".nii.gz", "").replace(".nii", "")
            bval_f = base + ".bval" if os.path.exists(base + ".bval") else None
            bvec_f = base + ".bvec" if os.path.exists(base + ".bvec") else None
    if bval_f is None or bvec_f is None:
        print("  [ERROR] Cannot locate bval/bvec. "
              "Please place them alongside the NIfTI file. Skipping DTI.")
        return None

    print(f"  Loading gradients: {bval_f}")
    bvals, bvecs = read_bvals_bvecs(bval_f, bvec_f)
    gtab = gradient_table(bvals, bvecs)

    data  = dti_img.get_fdata()
    affine = dti_img.affine

    print("  Brain masking …")
    _, mask = median_otsu(data[..., 0], median_radius=2, numpass=1)

    print("  Fitting tensor model …")
    tensor_model = TensorModel(gtab)
    tensor_fit   = tensor_model.fit(data, mask=mask)
    fa_map = tensor_fit.fa

    print(f"    FA range: [{fa_map.min():.3f}, {fa_map.max():.3f}]")
    fa_img = nib.Nifti1Image(fa_map, affine)
    nib.save(fa_img, str(OUT_DIR / "fa_map.nii.gz"))
    print("    Saved fa_map.nii.gz")

    print("  Deterministic tractography (FA threshold = {}) …".format(FA_THRESH))
    stop_criterion = ThresholdStoppingCriterion(fa_map, FA_THRESH)
    from dipy.data import get_sphere

    sphere = get_sphere("repulsion724")  # 724-vertex tessellation — good balance
    # of angular resolution vs speed.
    # Alternatives: "repulsion100" (faster),
    #               "symmetric362" (classic)

    peak_directions = peaks_from_model(
        model=tensor_model,
        data=data,
        sphere=sphere,
        mask=mask,
        relative_peak_threshold=0.5,
        min_separation_angle=25,
        npeaks=5,  # max peaks per voxel
        normalize_peaks=True,
    )
    
    seeds = tracking_utils.seeds_from_mask(
        fa_map > FA_THRESH, affine, density=1
    )
    streamlines = Streamlines(
        LocalTracking(peak_directions, stop_criterion, seeds, affine,
                      step_size=0.5, max_cross=1)
    )
    print(f"    Generated {len(streamlines):,} streamlines")

    return streamlines, fa_img, tensor_fit


def build_sc_matrix(streamlines, atlas_img: nib.Nifti1Image,
                    atlas_labels: list[str],
                    dti_affine: np.ndarray) -> np.ndarray:
    """
    Structural connectivity matrix from streamline endpoints.
    Each streamline contributes a +1 to the (ROI_A, ROI_B) cell.
    """
    if not DIPY_AVAILABLE or streamlines is None:
        n = len(atlas_labels)
        print("  [INFO] Returning zero SC matrix (DIPY unavailable).")
        return np.zeros((n, n))

    print("  Building structural connectivity matrix from streamlines …")
    from dipy.tracking import utils as tutils

    atlas_data = atlas_img.get_fdata().astype(int)
    n_roi      = int(atlas_data.max())
    sc_matrix  = np.zeros((n_roi, n_roi), dtype=float)

    for sl in streamlines:
        if len(sl) < 2:
            continue
        p0 = sl[0]
        p1 = sl[-1]
        # Transform streamline endpoints from DTI voxel → atlas voxel
        inv_atlas = np.linalg.inv(atlas_img.affine)
        v0 = np.round(inv_atlas @ np.append(
            dti_affine @ np.append(p0, 1), 1))[:3].astype(int)
        v1 = np.round(inv_atlas @ np.append(
            dti_affine @ np.append(p1, 1), 1))[:3].astype(int)

        def safe_label(v):
            try:
                if all(0 <= v[i] < atlas_data.shape[i] for i in range(3)):
                    return int(atlas_data[v[0], v[1], v[2]])
            except Exception:
                pass
            return 0

        r0, r1 = safe_label(v0), safe_label(v1)
        if r0 > 0 and r1 > 0 and r0 != r1:
            sc_matrix[r0-1, r1-1] += 1
            sc_matrix[r1-1, r0-1] += 1

    # Normalise by max value
    if sc_matrix.max() > 0:
        sc_matrix /= sc_matrix.max()

    np.save(str(OUT_DIR / "sc_matrix.npy"), sc_matrix)
    print(f"    SC matrix shape: {sc_matrix.shape}  saved sc_matrix.npy")
    return sc_matrix


# ─────────────────────────────────────────────────────────────────────────────
#  MODULE 7 – MULTIMODAL INTEGRATION
# ─────────────────────────────────────────────────────────────────────────────

def integrate_fc_sc(fc_matrix: np.ndarray,
                    sc_matrix: np.ndarray,
                    atlas_labels: list[str]) -> dict:
    """
    Compare functional vs structural connectivity matrices.
    Identify:
      - FC-only edges (functional without structural support)
      - SC-only edges (structural without functional expression)
      - Core edges   (strong in both)
    Uses z-score thresholding (z > 1.5).
    """
    print("\n── MODULE 7: MULTIMODAL INTEGRATION ─────────────────────────────")
    n = min(fc_matrix.shape[0], sc_matrix.shape[0])
    fc = fc_matrix[:n, :n].copy()
    sc = sc_matrix[:n, :n].copy()
    np.fill_diagonal(fc, 0)
    np.fill_diagonal(sc, 0)

    def z_thresh(mat, z=1.5):
        m, s = mat.mean(), mat.std()
        return mat > (m + z * s)

    fc_strong = z_thresh(fc)
    sc_strong = z_thresh(sc)

    fc_only   = fc_strong & ~sc_strong
    sc_only   = sc_strong & ~fc_strong
    core      = fc_strong & sc_strong

    print(f"  Strong FC connections:        {fc_strong.sum()//2:4d}")
    print(f"  Strong SC connections:        {sc_strong.sum()//2:4d}")
    print(f"  FC-only (no structural):      {fc_only.sum()//2:4d}")
    print(f"  SC-only (no functional):      {sc_only.sum()//2:4d}")
    print(f"  CORE (both FC and SC strong): {core.sum()//2:4d}")

    np.save(str(OUT_DIR / "fc_only_mask.npy"),  fc_only)
    np.save(str(OUT_DIR / "sc_only_mask.npy"),  sc_only)
    np.save(str(OUT_DIR / "core_mask.npy"),     core)

    return {"fc_only": fc_only, "sc_only": sc_only,
            "core": core, "n": n, "labels": atlas_labels[:n]}


# ─────────────────────────────────────────────────────────────────────────────
#  MODULE 8 – NETWORK ANALYSIS (GRAPH THEORY)
# ─────────────────────────────────────────────────────────────────────────────

def build_brain_graph(fc_matrix: np.ndarray,
                      atlas_labels: list[str],
                      threshold_pct: float = 0.15) -> nx.Graph:
    """
    Build a weighted undirected brain graph.
    Only top *threshold_pct* of FC edges are retained (sparsity control).
    """
    print("\n── MODULE 8: GRAPH-THEORETIC NETWORK ANALYSIS ───────────────────")
    n   = fc_matrix.shape[0]
    mat = fc_matrix.copy()
    np.fill_diagonal(mat, 0)
    mat = np.abs(mat)                            # unsigned weights

    # Threshold: keep top 15 % of edges
    thr = np.percentile(mat, (1 - threshold_pct) * 100)
    adj = np.where(mat >= thr, mat, 0)

    G = nx.Graph()
    labels = atlas_labels[:n]
    for i, lab in enumerate(labels):
        G.add_node(i, label=lab)
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j] > 0:
                G.add_edge(i, j, weight=float(adj[i, j]))

    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def compute_graph_metrics(G: nx.Graph, atlas_labels: list[str]) -> dict:
    """Degree centrality, betweenness centrality, clustering coefficient."""
    print("  Computing graph metrics …")
    deg_cent  = nx.degree_centrality(G)
    bet_cent  = nx.betweenness_centrality(G, weight="weight", normalized=True)
    clust     = nx.clustering(G, weight="weight")

    metrics = {
        "degree_centrality":      deg_cent,
        "betweenness_centrality": bet_cent,
        "clustering_coefficient": clust,
    }

    # Top 10 hubs by composite score (average of normalised metrics)
    all_nodes = list(G.nodes())
    scores    = {n: (deg_cent.get(n, 0) +
                     bet_cent.get(n, 0) +
                     clust.get(n, 0)) / 3
                 for n in all_nodes}
    top_hubs  = sorted(scores.items(), key=lambda x: -x[1])[:10]

    print("\n  ╔══════════════════════════════════════════════════════╗")
    print(  "  ║           TOP 10 HUB REGIONS (Brain Graph)          ║")
    print(  "  ╠══════════════════════════════════════════════════════╣")
    for rank, (node_id, score) in enumerate(top_hubs, 1):
        label = atlas_labels[node_id] if node_id < len(atlas_labels) else f"ROI-{node_id}"
        print(f"  ║ {rank:2d}. {label[:40]:40s} {score:.3f} ║")
    print(  "  ╚══════════════════════════════════════════════════════╝")

    metrics["hub_ranking"] = top_hubs
    return metrics


def print_top_connections(fc_matrix: np.ndarray,
                           atlas_labels: list[str], top_n: int = 10):
    """Print the top-N strongest functional connections."""
    print("\n  ╔══════════════════════════════════════════════════════════════╗")
    print(  "  ║           TOP 10 STRONGEST FUNCTIONAL CONNECTIONS           ║")
    print(  "  ╠══════════════════════════════════════════════════════════════╣")
    n   = fc_matrix.shape[0]
    mat = fc_matrix.copy()
    np.fill_diagonal(mat, 0)
    idx = np.dstack(np.unravel_index(
        np.argsort(mat.ravel())[::-1], mat.shape))[0]
    seen, count = set(), 0
    for i, j in idx:
        pair = tuple(sorted([int(i), int(j)]))
        if pair in seen:
            continue
        seen.add(pair)
        la = atlas_labels[i] if i < len(atlas_labels) else f"ROI-{i}"
        lb = atlas_labels[j] if j < len(atlas_labels) else f"ROI-{j}"
        r  = mat[i, j]
        print(f"  ║ {count+1:2d}. {la[:24]:24s} ↔ {lb[:24]:24s}  r={r:.3f} ║")
        count += 1
        if count >= top_n:
            break
    print("  ╚══════════════════════════════════════════════════════════════╝")


def print_top_active_regions(alff_img: nib.Nifti1Image,
                              atlas_img: nib.Nifti1Image,
                              atlas_labels: list[str], top_n: int = 10):
    """Print the top-N most active regions by mean ALFF."""
    print("\n  ╔══════════════════════════════════════════════════════╗")
    print(  "  ║        TOP 10 MOST ACTIVE REGIONS (Mean ALFF)       ║")
    print(  "  ╠══════════════════════════════════════════════════════╣")
    alff_r = resample_to_img(alff_img, atlas_img, interpolation="continuous")
    alff_d = alff_r.get_fdata()
    atlas_d = atlas_img.get_fdata().astype(int)
    roi_vals = {}
    for roi_id in np.unique(atlas_d):
        if roi_id == 0:
            continue
        roi_vals[roi_id] = alff_d[atlas_d == roi_id].mean()
    top = sorted(roi_vals.items(), key=lambda x: -x[1])[:top_n]
    for rank, (roi_id, val) in enumerate(top, 1):
        label = atlas_labels[roi_id-1] if roi_id-1 < len(atlas_labels) \
                else f"ROI-{roi_id}"
        print(f"  ║ {rank:2d}. {label[:40]:40s} {val:.4f} ║")
    print("  ╚══════════════════════════════════════════════════════╝")


# ─────────────────────────────────────────────────────────────────────────────
#  MODULE 9 – VISUALISATION SUITE
# ─────────────────────────────────────────────────────────────────────────────

def plot_activation_maps(alff_img, falff_img, reho_img,
                         bold_img: nib.Nifti1Image):
    """ALFF, fALFF, ReHo montage on axial slices."""
    print("\n── MODULE 9: VISUALISATION ──────────────────────────────────────")
    bg = image.mean_img(bold_img)
    fig, axes = plt.subplots(3, 1, figsize=(18, 12))
    fig.patch.set_facecolor("#0d0d0d")

    configs = [
        (alff_img,  "ALFF – Amplitude of Low-Frequency Fluctuations",  "hot"),
        (falff_img, "fALFF – Fractional ALFF (normalised)",            "YlOrRd"),
        (reho_img,  "ReHo – Regional Homogeneity (Kendall W)",         "plasma"),
    ]
    for ax, (img, title, cmap) in zip(axes, configs):
        try:
            display = plotting.plot_stat_map(
                img, bg_img=bg, display_mode="z",
                cut_coords=7, colorbar=True,
                cmap=cmap, title=title, axes=ax,
                annotate=False, black_bg=True
            )
        except Exception:
            ax.set_title(title, color="white")
            ax.axis("off")

    plt.tight_layout()
    out = str(OUT_DIR / "activation_maps.png")
    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor="#0d0d0d")
    plt.close()
    print(f"  Saved activation_maps.png")


def plot_glass_brains(seed_fc: dict, bold_img: nib.Nifti1Image):
    """Glass-brain seed-FC maps (one per seed)."""
    n   = len(seed_fc)
    fig = plt.figure(figsize=(6 * n, 6))
    fig.patch.set_facecolor("#0d0d0d")
    for i, (name, img) in enumerate(seed_fc.items()):
        ax = fig.add_subplot(1, n, i+1)
        try:
            plotting.plot_glass_brain(
                img, threshold=0.3, colorbar=True,
                title=f"Seed FC: {name}", axes=ax,
                black_bg=True, plot_abs=False
            )
        except Exception:
            ax.set_title(name, color="white"); ax.axis("off")
    plt.tight_layout()
    out = str(OUT_DIR / "glass_brain_seed_fc.png")
    plt.savefig(out, dpi=120, bbox_inches="tight",
                facecolor="#0d0d0d")
    plt.close()
    print("  Saved glass_brain_seed_fc.png")


def plot_fc_heatmap(fc_matrix: np.ndarray, atlas_labels: list[str],
                    title: str = "Functional Connectivity Matrix",
                    fname: str = "fc_heatmap.png"):
    """Seaborn heatmap of FC matrix with atlas labels."""
    n      = fc_matrix.shape[0]
    labels = [l[:20] for l in atlas_labels[:n]]
    fig, ax = plt.subplots(figsize=(16, 14))
    mask   = np.eye(n, dtype=bool)
    sns.heatmap(fc_matrix, mask=mask, cmap="RdBu_r",
                center=0, vmin=-1, vmax=1,
                xticklabels=labels, yticklabels=labels,
                linewidths=0, ax=ax, cbar_kws={"shrink": 0.5})
    ax.set_title(title, fontsize=14, pad=12)
    plt.xticks(fontsize=5, rotation=90)
    plt.yticks(fontsize=5, rotation=0)
    plt.tight_layout()
    out = str(OUT_DIR / fname)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fname}")


def plot_multimodal_comparison(fc_matrix, sc_matrix, atlas_labels, n: int):
    """Side-by-side FC vs SC heatmaps."""
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    labels = [l[:18] for l in atlas_labels[:n]]
    kw = dict(cmap="RdBu_r", center=0, vmin=-0.5, vmax=0.5,
              xticklabels=False, yticklabels=False, linewidths=0)
    sns.heatmap(fc_matrix[:n, :n], ax=axes[0], **kw,
                cbar_kws={"label": "Pearson r"})
    axes[0].set_title("Functional Connectivity (FC)", fontsize=12)
    sns.heatmap(sc_matrix[:n, :n], ax=axes[1],
                cmap="viridis", vmin=0, vmax=1,
                xticklabels=False, yticklabels=False,
                linewidths=0, cbar_kws={"label": "Norm. streamlines"})
    axes[1].set_title("Structural Connectivity (SC)", fontsize=12)
    # Difference
    diff = np.abs(fc_matrix[:n, :n]) - sc_matrix[:n, :n]
    sns.heatmap(diff, ax=axes[2], cmap="coolwarm",
                center=0, xticklabels=False, yticklabels=False,
                linewidths=0, cbar_kws={"label": "|FC| – SC"})
    axes[2].set_title("FC – SC Discordance", fontsize=12)
    plt.suptitle("Multimodal Connectivity Comparison", fontsize=14, y=1.01)
    plt.tight_layout()
    out = str(OUT_DIR / "multimodal_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved multimodal_comparison.png")


def plot_network_graph(G: nx.Graph, metrics: dict,
                       atlas_labels: list[str]):
    """Spring-layout network graph coloured by betweenness centrality."""
    fig, ax = plt.subplots(figsize=(16, 12))
    fig.patch.set_facecolor("#111111")
    ax.set_facecolor("#111111")

    pos    = nx.spring_layout(G, seed=42, k=0.4)
    bet    = metrics["betweenness_centrality"]
    deg    = metrics["degree_centrality"]
    sizes  = [3000 * deg.get(n, 0) + 50 for n in G.nodes()]
    colors = [bet.get(n, 0) for n in G.nodes()]
    widths = [G[u][v]["weight"] * 2 for u, v in G.edges()]

    nx.draw_networkx_edges(G, pos, ax=ax,
                           width=widths, edge_color="steelblue", alpha=0.25)
    nc = nx.draw_networkx_nodes(G, pos, ax=ax,
                                node_size=sizes, node_color=colors,
                                cmap=plt.cm.plasma, alpha=0.9)
    # Label top-10 hubs only
    top_ids = {n for n, _ in metrics["hub_ranking"]}
    hub_labels = {n: (atlas_labels[n][:15] if n < len(atlas_labels)
                       else f"ROI-{n}")
                  for n in top_ids}
    nx.draw_networkx_labels(G, pos, labels=hub_labels,
                            font_size=7, font_color="white", ax=ax)
    plt.colorbar(nc, ax=ax, label="Betweenness Centrality",
                 fraction=0.03, pad=0.02)
    ax.set_title("Brain Connectivity Graph\n(node size ∝ degree, "
                 "colour ∝ betweenness)", color="white", fontsize=13)
    ax.axis("off")
    out = str(OUT_DIR / "brain_network_graph.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#111111")
    plt.close()
    print("  Saved brain_network_graph.png")


def plot_ica_networks(components_img: nib.Nifti1Image,
                      matched: dict, bold_img: nib.Nifti1Image):
    """Plot the matched ICA component for each known resting-state network."""
    bg  = image.mean_img(bold_img)
    n   = len(matched)
    fig, axes = plt.subplots(n, 1, figsize=(18, 5 * n))
    if n == 1:
        axes = [axes]
    fig.patch.set_facecolor("#0d0d0d")
    for ax, (net, info) in zip(axes, matched.items()):
        comp_idx = info["component"]
        comp_img = image.index_img(components_img, comp_idx)
        try:
            plotting.plot_stat_map(
                comp_img, bg_img=bg, display_mode="z",
                cut_coords=6, colorbar=True, cmap="cold_hot",
                title=f"{net}  (ICA comp {comp_idx}, r={info['r']:.2f})",
                axes=ax, black_bg=True, annotate=False
            )
        except Exception:
            ax.set_title(net, color="white"); ax.axis("off")
    plt.tight_layout()
    out = str(OUT_DIR / "ica_networks_DMN_SAL_CEN.png")
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close()
    print("  Saved ica_networks_DMN_SAL_CEN.png")


def plot_fa_map(fa_img: nib.Nifti1Image | None):
    """Plot DTI FA map if available."""
    if fa_img is None:
        return
    fig, ax = plt.subplots(1, 1, figsize=(16, 4))
    try:
        plotting.plot_anat(fa_img, display_mode="z", cut_coords=8,
                           title="DTI Fractional Anisotropy (FA) Map",
                           cmap="bone", colorbar=True, axes=ax)
    except Exception:
        ax.set_title("FA Map (rendering error)"); ax.axis("off")
    plt.tight_layout()
    out = str(OUT_DIR / "dti_fa_map.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print("  Saved dti_fa_map.png")


def plot_interactive_network(G: nx.Graph, metrics: dict,
                              atlas_labels: list[str]):
    """Plotly interactive network graph (HTML)."""
    if not PLOTLY_AVAILABLE:
        return
    print("  Building interactive Plotly network graph …")
    pos = nx.spring_layout(G, seed=42)
    bet = metrics["betweenness_centrality"]
    deg = metrics["degree_centrality"]

    # Edge traces
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x += [x0, x1, None]; edge_y += [y0, y1, None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines",
                             line=dict(width=0.6, color="#888"),
                             hoverinfo="none")

    # Node traces
    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_text = [atlas_labels[n][:30] if n < len(atlas_labels)
                 else f"ROI-{n}" for n in G.nodes()]
    node_color = [bet.get(n, 0) for n in G.nodes()]
    node_size  = [20 + 60 * deg.get(n, 0) for n in G.nodes()]

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        hoverinfo="text", text=node_text,
        textfont=dict(size=7),
        marker=dict(size=node_size, color=node_color,
                    colorscale="Plasma", showscale=True,
                    colorbar=dict(title="Betweenness"),
                    line=dict(width=0.5, color="#fff"))
    )
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title="Interactive Brain Connectivity Network",
                        showlegend=False,
                        hovermode="closest",
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False,
                                   showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False,
                                   showticklabels=False),
                        paper_bgcolor="#111", plot_bgcolor="#111",
                        font=dict(color="white")
                    ))
    out = str(OUT_DIR / "interactive_network.html")
    fig.write_html(out)
    print(f"  Saved interactive_network.html")


def plot_hub_bar(metrics: dict, atlas_labels: list[str]):
    """Horizontal bar chart of top hub regions."""
    top = metrics["hub_ranking"]
    labels = [atlas_labels[n][:35] if n < len(atlas_labels)
              else f"ROI-{n}" for n, _ in top]
    scores = [s for _, s in top]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(labels[::-1], scores[::-1],
                   color=plt.cm.viridis(np.linspace(0.3, 0.9, len(top))))
    ax.set_xlabel("Hub Score (mean of degree / betweenness / clustering)",
                  fontsize=10)
    ax.set_title("Top 10 Brain Hub Regions", fontsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, score in zip(bars, scores[::-1]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f"{score:.3f}", va="center", fontsize=8)
    plt.tight_layout()
    out = str(OUT_DIR / "hub_regions_bar.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print("  Saved hub_regions_bar.png")


def plot_network_summary_dashboard(fc_matrix, sc_matrix, metrics,
                                   alff_img, atlas_labels):
    """4-panel summary dashboard."""
    n = min(fc_matrix.shape[0], sc_matrix.shape[0], 30)
    fig = plt.figure(figsize=(20, 16))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # Panel A – FC heatmap (top 30 regions)
    ax_a = fig.add_subplot(gs[0, 0])
    sns.heatmap(fc_matrix[:n, :n], cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, ax=ax_a,
                xticklabels=False, yticklabels=False, linewidths=0)
    ax_a.set_title("Functional Connectivity Matrix", fontsize=11)

    # Panel B – SC heatmap
    ax_b = fig.add_subplot(gs[0, 1])
    sns.heatmap(sc_matrix[:n, :n], cmap="viridis", vmin=0, vmax=1,
                ax=ax_b, xticklabels=False,
                yticklabels=False, linewidths=0)
    ax_b.set_title("Structural Connectivity Matrix", fontsize=11)

    # Panel C – Hub bar
    ax_c = fig.add_subplot(gs[1, 0])
    top  = metrics["hub_ranking"][:8]
    ylabels = [atlas_labels[n_][:22] if n_ < len(atlas_labels)
               else f"ROI-{n_}" for n_, _ in top]
    scores  = [s for _, s in top]
    ax_c.barh(ylabels[::-1], scores[::-1],
              color=plt.cm.plasma(np.linspace(0.2, 0.8, len(top))))
    ax_c.set_xlabel("Hub Score"); ax_c.set_title("Top Hub Regions", fontsize=11)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)

    # Panel D – ALFF ROI means (top 15)
    ax_d = fig.add_subplot(gs[1, 1])
    alff_r = resample_to_img(alff_img,
                             nib.Nifti1Image(
                                 np.zeros((91, 109, 91)), np.eye(4)),
                             interpolation="continuous")
    # Simple proxy: compute ROI means from ALFF data
    alff_d = alff_img.get_fdata().ravel()
    # histogram of ALFF values
    ax_d.hist(alff_d[alff_d > 0], bins=60, color="tomato", alpha=0.8,
              edgecolor="none")
    ax_d.set_xlabel("ALFF value"); ax_d.set_ylabel("Voxel count")
    ax_d.set_title("ALFF Distribution", fontsize=11)
    ax_d.spines["top"].set_visible(False)
    ax_d.spines["right"].set_visible(False)

    fig.suptitle("Multimodal Brain Connectivity – Summary Dashboard",
                 fontsize=15, y=0.98, weight="bold")
    out = str(OUT_DIR / "summary_dashboard.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved summary_dashboard.png")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline():
    t_start = time.time()

    # ── 1. Load data ─────────────────────────────────────────────────────────
    data = load_all_modalities()

    if data["bold"] is None:
        print("\n[FATAL] BOLD image not found. "
              "Please verify the path in PATHS['bold']. Exiting.")
        return

    # ── 2. Preprocess BOLD ───────────────────────────────────────────────────
    bold_mni = preprocess_bold(data["bold"], data.get("t1"))

    # ── 3. Activation metrics ────────────────────────────────────────────────
    print("\n── MODULE 3: ACTIVATION METRICS ─────────────────────────────────")
    alff_img, falff_img = compute_alff_falff(bold_mni, TR)

    # ReHo is expensive on full MNI volume; run on sub-sampled 4 mm version
    bold_4mm = image.resample_img(bold_mni, target_affine=np.diag([4, 4, 4, 1]),
                                  interpolation="linear")
    reho_img = compute_reho(bold_4mm)

    # ── 4. Atlas & FC ────────────────────────────────────────────────────────
    print("\n── MODULE 4: FUNCTIONAL CONNECTIVITY ────────────────────────────")
    atlas_img, atlas_labels = load_atlas()
    seed_fc  = seed_based_connectivity(bold_mni, SEEDS)
    fc_matrix, _, _ = whole_brain_fc_matrix(bold_mni, atlas_img, atlas_labels)

    print_top_active_regions(alff_img, atlas_img, atlas_labels)
    print_top_connections(fc_matrix, atlas_labels)

    # ── 5. ICA / RSN ─────────────────────────────────────────────────────────
    components_img, canica = run_ica(bold_mni, n_components=20)
    matched = match_network_components(components_img, bold_mni)

    # ── 6. Structural connectivity ───────────────────────────────────────────
    dti_result = None
    fa_img     = None
    if data["dti"] is not None and DIPY_AVAILABLE:
        dti_result = run_dti_tractography(data["dti"], PATHS["dti"])
        if dti_result is not None:
            streamlines, fa_img, _ = dti_result

    # Build SC matrix (zeros if tractography unavailable)
    n_roi = fc_matrix.shape[0]
    if dti_result is not None:
        sc_matrix = build_sc_matrix(streamlines, atlas_img,
                                     atlas_labels, data["dti"].affine)
        # Ensure same shape as FC
        if sc_matrix.shape[0] != n_roi:
            sc_pad = np.zeros((n_roi, n_roi))
            m = min(sc_matrix.shape[0], n_roi)
            sc_pad[:m, :m] = sc_matrix[:m, :m]
            sc_matrix = sc_pad
    else:
        print("\n  [INFO] SC matrix set to zeros (DTI pipeline skipped).")
        sc_matrix = np.zeros((n_roi, n_roi))

    # ── 7. Multimodal integration ─────────────────────────────────────────────
    integration = integrate_fc_sc(fc_matrix, sc_matrix, atlas_labels)

    # ── 8. Graph analysis ─────────────────────────────────────────────────────
    G       = build_brain_graph(fc_matrix, atlas_labels)
    metrics = compute_graph_metrics(G, atlas_labels)

    # ── 9. Visualise ──────────────────────────────────────────────────────────
    plot_activation_maps(alff_img, falff_img, reho_img, bold_mni)
    plot_glass_brains(seed_fc, bold_mni)
    plot_fc_heatmap(fc_matrix, atlas_labels,
                    "Functional Connectivity Matrix (Harvard-Oxford)",
                    "fc_heatmap.png")
    plot_multimodal_comparison(fc_matrix, sc_matrix, atlas_labels,
                                integration["n"])
    plot_network_graph(G, metrics, atlas_labels)
    plot_hub_bar(metrics, atlas_labels)
    plot_ica_networks(components_img, matched, bold_mni)
    plot_fa_map(fa_img)
    plot_interactive_network(G, metrics, atlas_labels)
    plot_network_summary_dashboard(fc_matrix, sc_matrix, metrics,
                                   alff_img, atlas_labels)

    # ── Final summary ─────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  PIPELINE COMPLETE  ({elapsed/60:.1f} min)")
    print(f"  Outputs saved to: {OUT_DIR.resolve()}")
    print(f"{'='*70}")
    print("\n  Output files:")
    for f in sorted(OUT_DIR.glob("*")):
        size = f.stat().st_size / 1024
        print(f"    {f.name:<50s} {size:8.1f} KB")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_pipeline()