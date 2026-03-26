"""
=============================================================================
MULTIMODAL MRI BRAIN CONNECTIVITY PIPELINE
=============================================================================
Fixes vs previous version:
  ① atlas_labels length always equals fc_matrix.shape[0]  (root-cause fix)
  ② Removed duplicate function definitions (compute_graph_metrics etc.)
  ③ save_additional_data uses labels_clean = atlas_labels[:n] as safe guard
  ④ print() calls now also go through the Tee logger after initialization

New visualizations added (Module 10):
  • Circular / chord-style connectivity diagram
  • Community detection (greedy modularity) + coloured graph
  • Centrality radar (spider) chart for top hubs
  • FC vs SC edge-weight scatter plot
  • Degree distribution + power-law overlay
  • Module-ordered FC heatmap
  • Rich-club coefficient curve
  • 6-panel comprehensive network analysis dashboard
  • Louvain-reordered community heatmap
  • Weighted edge histogram
=============================================================================
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os, glob, warnings, time, sys
from pathlib import Path
warnings.filterwarnings("ignore")

# ── Numerical / scientific ────────────────────────────────────────────────────
import numpy as np
import scipy.ndimage as ndi
from scipy.signal import butter, filtfilt
from scipy.stats import pearsonr

# ── NIfTI I/O ─────────────────────────────────────────────────────────────────
import nibabel as nib

# ── nilearn ───────────────────────────────────────────────────────────────────
from nilearn import image, plotting, signal, decomposition
from nilearn.connectome import ConnectivityMeasure
from nilearn.maskers import NiftiLabelsMasker, NiftiMasker
from nilearn.datasets import fetch_atlas_harvard_oxford, load_mni152_template
from nilearn.image import clean_img, smooth_img, resample_to_img

# ── DIPY ──────────────────────────────────────────────────────────────────────
try:
    from dipy.io.gradients import read_bvals_bvecs
    from dipy.core.gradients import gradient_table
    from dipy.reconst.dti import TensorModel
    from dipy.direction import peaks_from_model
    from dipy.data import get_sphere
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
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import seaborn as sns

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# ── NetworkX ──────────────────────────────────────────────────────────────────
import networkx as nx
from networkx.algorithms import community as nx_community

# ── Pandas ────────────────────────────────────────────────────────────────────
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
#  LOGGING (Tee: terminal + file)
# ─────────────────────────────────────────────────────────────────────────────
class Tee:
    def __init__(self, filename, mode="w"):
        self.file   = open(filename, mode, encoding="utf-8")
        self.stdout = sys.stdout
    def write(self, msg):
        self.stdout.write(msg)
        self.file.write(msg)
        self.file.flush()
    def flush(self):
        self.stdout.flush()
        self.file.flush()

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
PATHS = {
    "t1":  "你的地址/T1",
    "bold":"你的地址/BOLD",
    "dti": "你的地址/DTI",
    "dsi": "你的地址/DSI",
}

OUT_DIR          = Path("./brain_pipeline_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE         = OUT_DIR / "pipeline_terminal_output.txt"
CSV_SUMMARY_FILE = OUT_DIR / "pipeline_results_summary.csv"

# Preprocessing
TR        = 1.0
FWHM      = 6.0
HP_FREQ   = 0.01
LP_FREQ   = 0.10
N_DUMMIES = 5
FA_THRESH = 0.20

SEEDS = {
    "PCC":         (0,  -52,  26),
    "Hippocampus": (24, -22, -20),
    "mPFC":        (0,   52,  -6),
    "Insula":      (38,   2,   0),
    "DLPFC":       (44,  36,  20),
}

NETWORK_SEEDS = {
    "DMN":      [(0, -52, 26), (-46, -64, 28), (46, -64, 28), (0, 52, -6)],
    "Salience": [(38, 2,   0), (-38,  2,   0), (0,  16, 44)],
    "CEN":      [(44, 36, 20), (-44, 36,  20), (40,-52, 48), (-40,-52, 48)],
}

# Start logging
sys.stdout = Tee(str(LOG_FILE), mode="w")

print("=" * 70)
print("  MULTIMODAL MRI BRAIN CONNECTIVITY PIPELINE  v3")
print("=" * 70)
print(f"  Log  → {LOG_FILE}")
print(f"  CSV  → {CSV_SUMMARY_FILE}")
print("=" * 70)

# ═════════════════════════════════════════════════════════════════════════════
#  MODULE 1 – DATA DISCOVERY
# ═════════════════════════════════════════════════════════════════════════════

def discover_niftis(directory):
    p1 = os.path.join(directory, "**", "*.nii.gz")
    p2 = os.path.join(directory, "**", "*.nii")
    return sorted(glob.glob(p1, recursive=True) + glob.glob(p2, recursive=True))


def validate_and_pick(files, label, expected_ndim=None):
    best, best_vols = None, 0
    for f in files:
        try:
            img   = nib.load(f)
            shape = img.shape
            ndim  = len(shape)
            vols  = shape[3] if ndim == 4 else 1
            print(f"  [{label}] {os.path.basename(f)}  shape={shape}  dtype={img.get_data_dtype()}")
            if expected_ndim is None:
                if vols > best_vols: best, best_vols = img, vols
            elif ndim == expected_ndim:
                if vols > best_vols: best, best_vols = img, vols
        except Exception as e:
            print(f"  [WARN] Could not load {f}: {e}")
    return best


def load_all_modalities():
    print("\n── MODULE 1: DATA DISCOVERY ──────────────────────────────────────")
    data = {}
    for key, path in PATHS.items():
        files = discover_niftis(path)
        print(f"\n[{key.upper()}] Found {len(files)} NIfTI file(s) in {path}")
        if not files:
            print(f"  [WARN] No files found for {key}.")
            data[key] = None
            continue
        exp = 4 if key in ("bold", "dti", "dsi") else 3
        data[key] = validate_and_pick(files, key, expected_ndim=exp)
    print("\n── Loaded Modalities ─────────────────────────────────────────────")
    for k, v in data.items():
        print(f"  {k:6s}: {'shape='+str(v.shape) if v else 'MISSING'}")
    return data


# ═════════════════════════════════════════════════════════════════════════════
#  MODULE 2 – rs-fMRI PREPROCESSING
# ═════════════════════════════════════════════════════════════════════════════

def drop_dummies(img_4d, n=N_DUMMIES):
    arr = img_4d.get_fdata()[..., n:]
    return nib.Nifti1Image(arr, img_4d.affine, img_4d.header)


def bandpass_filter(img_4d, tr, lp, hp):
    arr = img_4d.get_fdata()
    n_t = arr.shape[3]
    nyq = 0.5 / tr
    b, a = butter(2, [hp/nyq, lp/nyq], btype="band")
    flat   = arr.reshape(-1, n_t)
    mask   = np.std(flat, axis=1) > 0
    result = flat.copy()
    result[mask] = filtfilt(b, a, flat[mask], axis=1)
    return nib.Nifti1Image(result.reshape(arr.shape), img_4d.affine, img_4d.header)


def preprocess_bold(bold_img, t1_img=None):
    print("\n── MODULE 2: rs-fMRI PREPROCESSING ──────────────────────────────")
    t0 = time.time()
    print("  Step 1/4: Dropping dummy scans …")
    bold = drop_dummies(bold_img)
    print(f"    → shape: {bold.shape}")
    print(f"  Step 2/4: Spatial smoothing (FWHM={FWHM} mm) …")
    bold = smooth_img(bold, fwhm=FWHM)
    print(f"  Step 3/4: Bandpass filtering ({HP_FREQ}–{LP_FREQ} Hz) …")
    bold = bandpass_filter(bold, tr=TR, lp=LP_FREQ, hp=HP_FREQ)
    print("  Step 4/4: Resampling to MNI152 2 mm …")
    mni      = load_mni152_template(resolution=2)
    bold_mni = resample_to_img(bold, mni, interpolation="linear")
    print(f"    → MNI shape: {bold_mni.shape}")
    print(f"  Done in {time.time()-t0:.1f}s")
    nib.save(bold_mni, str(OUT_DIR / "bold_preprocessed_mni.nii.gz"))
    return bold_mni


# ═════════════════════════════════════════════════════════════════════════════
#  MODULE 3 – ACTIVATION METRICS
# ═════════════════════════════════════════════════════════════════════════════

def compute_alff_falff(bold_img, tr):
    print("\n  Computing ALFF / fALFF …")
    arr   = bold_img.get_fdata()
    n_t   = arr.shape[3]
    freqs = np.fft.rfftfreq(n_t, d=tr)
    lf    = (freqs >= HP_FREQ) & (freqs <= LP_FREQ)
    fft   = np.abs(np.fft.rfft(arr, axis=3))
    alff  = fft[..., lf].mean(axis=3)
    tot   = fft.mean(axis=3)
    falff = np.where(tot > 0, alff / tot, 0)
    alff_img  = nib.Nifti1Image(alff,  bold_img.affine)
    falff_img = nib.Nifti1Image(falff, bold_img.affine)
    nib.save(alff_img,  str(OUT_DIR / "alff.nii.gz"))
    nib.save(falff_img, str(OUT_DIR / "falff.nii.gz"))
    print("    Saved alff.nii.gz  falff.nii.gz")
    return alff_img, falff_img


def compute_reho(bold_img, neighbourhood=1):
    print("  Computing ReHo (26-neighbour Kendall W) …")
    from scipy.stats import rankdata
    arr  = bold_img.get_fdata().astype(np.float32)
    x, y, z, n_t = arr.shape
    k    = (2*neighbourhood+1)**3
    reho = np.zeros((x, y, z), dtype=np.float32)
    arr_ranked = np.apply_along_axis(rankdata, 3, arr)
    d = neighbourhood
    for xi in range(d, x-d):
        for yi in range(d, y-d):
            for zi in range(d, z-d):
                cube = arr_ranked[xi-d:xi+d+1, yi-d:yi+d+1, zi-d:zi+d+1, :]
                ts_mat    = cube.reshape(k, n_t)
                rank_sums = ts_mat.sum(axis=0)
                S = np.sum((rank_sums - rank_sums.mean())**2)
                reho[xi, yi, zi] = 12*S / (k**2*(n_t**3-n_t)+1e-9)
    reho_img = nib.Nifti1Image(reho, bold_img.affine)
    nib.save(reho_img, str(OUT_DIR / "reho.nii.gz"))
    print("    Saved reho.nii.gz")
    return reho_img


# ═════════════════════════════════════════════════════════════════════════════
#  MODULE 4 – FUNCTIONAL CONNECTIVITY
# ═════════════════════════════════════════════════════════════════════════════

def load_atlas():
    """
    Load Harvard-Oxford atlas and return (atlas_img, atlas_labels).

    FIX: We return the raw labels from nilearn and let whole_brain_fc_matrix
    determine the authoritative label list from the masker itself.
    The raw labels are kept here only for reference; the "working" labels
    used everywhere else come from _align_labels_to_matrix().
    """
    print("\n  Loading Harvard-Oxford atlas …")
    ho = fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
    atlas_img    = ho.maps
    atlas_labels = list(ho.labels)
    print(f"    Raw label count from nilearn: {len(atlas_labels)}")
    return atlas_img, atlas_labels


def _align_labels_to_matrix(atlas_labels, n_rois):
    """
    ── ROOT-CAUSE FIX ──
    nilearn's NiftiLabelsMasker extracts n_rois ROIs (skips background = 0).
    ho.labels may have len = n_rois+1 (background at index 0) or even n_rois+2
    depending on the nilearn version.

    Strategy:
      1. Drop any empty-string entries (background markers).
      2. Truncate / pad to exactly n_rois.

    This ensures len(labels) == fc_matrix.shape[0] always.
    """
    clean = [l for l in atlas_labels if str(l).strip() != ""]
    if len(clean) >= n_rois:
        return clean[:n_rois]
    # If somehow we have fewer, pad with generic names
    while len(clean) < n_rois:
        clean.append(f"ROI-{len(clean)+1}")
    return clean


def seed_based_connectivity(bold_img, seeds):
    print("\n  Seed-based functional connectivity …")
    arr     = bold_img.get_fdata()
    affine  = bold_img.affine
    inv_aff = np.linalg.inv(affine)
    results = {}
    for name, mni_coord in seeds.items():
        vox = np.round(inv_aff @ np.array([*mni_coord, 1]))[:3].astype(int)
        vx, vy, vz = np.clip(vox, 0, [arr.shape[0]-1, arr.shape[1]-1, arr.shape[2]-1])
        seed_ts  = arr[vx, vy, vz, :]
        flat     = arr.reshape(-1, arr.shape[3])
        std_flat = flat.std(axis=1)
        seed_z   = (seed_ts - seed_ts.mean()) / (seed_ts.std() + 1e-9)
        flat_z   = np.where(std_flat[:, None] > 0,
                            (flat - flat.mean(axis=1, keepdims=True)) /
                            (std_flat[:, None] + 1e-9), 0)
        corr_map = (flat_z @ seed_z / arr.shape[3]).reshape(arr.shape[:3])
        img = nib.Nifti1Image(corr_map, affine)
        nib.save(img, str(OUT_DIR / f"seed_fc_{name}.nii.gz"))
        print(f"    {name} → vox {[vx,vy,vz]}  r∈[{corr_map.min():.3f},{corr_map.max():.3f}]")
        results[name] = img
    return results


def whole_brain_fc_matrix(bold_img, atlas_img, atlas_labels_raw):
    """
    Returns (fc_matrix, ts_matrix, labels) where labels is GUARANTEED to
    have exactly fc_matrix.shape[0] entries.
    """
    print("\n  Whole-brain functional connectivity matrix …")
    masker = NiftiLabelsMasker(
        labels_img=atlas_img, standardize=True, detrend=True,
        t_r=TR, memory_level=0, verbose=0
    )
    ts_matrix = masker.fit_transform(bold_img)
    n_rois    = ts_matrix.shape[1]
    print(f"    Time series shape: {ts_matrix.shape}")

    labels = _align_labels_to_matrix(atlas_labels_raw, n_rois)
    print(f"    Aligned label count: {len(labels)}  (matches {n_rois} ROIs)")

    conn     = ConnectivityMeasure(kind="correlation")
    fc_matrix = conn.fit_transform([ts_matrix])[0]
    np.save(str(OUT_DIR / "fc_matrix.npy"), fc_matrix)
    print(f"    FC matrix: {fc_matrix.shape}  saved fc_matrix.npy")
    return fc_matrix, ts_matrix, labels


# ═════════════════════════════════════════════════════════════════════════════
#  MODULE 5 – ICA / RESTING-STATE NETWORKS
# ═════════════════════════════════════════════════════════════════════════════

def run_ica(bold_img, n_components=20):
    print("\n── MODULE 5: ICA / RESTING-STATE NETWORKS ────────────────────────")
    print(f"  Running CanICA (n_components={n_components}) …")
    canica = decomposition.CanICA(n_components=n_components, threshold=3.0,
                                  random_state=42, verbose=0)
    canica.fit(bold_img)
    comp_img = canica.components_img_
    nib.save(comp_img, str(OUT_DIR / "ica_components.nii.gz"))
    print(f"    Saved {n_components} components → ica_components.nii.gz")
    return comp_img, canica


def match_network_components(components_img, bold_img):
    print("  Matching ICA components to known networks …")
    comp_data = components_img.get_fdata()
    n_comp    = comp_data.shape[3]
    affine    = components_img.affine
    matched   = {}
    for net_name, centres in NETWORK_SEEDS.items():
        template = np.zeros(comp_data.shape[:3])
        inv_aff  = np.linalg.inv(affine)
        for c in centres:
            vox = np.round(inv_aff @ np.array([*c, 1]))[:3].astype(int)
            vx, vy, vz = np.clip(vox, 0, np.array(template.shape)-1)
            xi, yi, zi = np.ogrid[:template.shape[0], :template.shape[1], :template.shape[2]]
            template[(xi-vx)**2 + (yi-vy)**2 + (zi-vz)**2 <= 16] = 1
        t_flat = template.ravel()
        best_r, best_idx = -np.inf, 0
        for i in range(n_comp):
            r = np.corrcoef(t_flat, comp_data[..., i].ravel())[0, 1]
            if r > best_r: best_r, best_idx = r, i
        matched[net_name] = {"component": best_idx, "r": best_r}
        print(f"    {net_name:12s} → component {best_idx:2d}  (r={best_r:.3f})")
    return matched


# ═════════════════════════════════════════════════════════════════════════════
#  MODULE 6 – STRUCTURAL CONNECTIVITY (DTI)
# ═════════════════════════════════════════════════════════════════════════════

def find_bval_bvec(dti_dir):
    bvals = glob.glob(os.path.join(dti_dir, "**", "*.bval"), recursive=True)
    bvecs = glob.glob(os.path.join(dti_dir, "**", "*.bvec"), recursive=True)
    return (bvals[0] if bvals else None), (bvecs[0] if bvecs else None)


def run_dti_tractography(dti_img, dti_dir):
    if not DIPY_AVAILABLE:
        print("  [SKIP] DIPY not available.")
        return None
    print("\n── MODULE 6: STRUCTURAL CONNECTIVITY (DTI) ──────────────────────")
    bval_f, bvec_f = find_bval_bvec(dti_dir)
    if not bval_f or not bvec_f:
        nii_files = discover_niftis(dti_dir)
        if nii_files:
            base   = nii_files[0].replace(".nii.gz","").replace(".nii","")
            bval_f = base+".bval" if os.path.exists(base+".bval") else None
            bvec_f = base+".bvec" if os.path.exists(base+".bvec") else None
    if not bval_f or not bvec_f:
        print("  [ERROR] Cannot locate bval/bvec. Skipping DTI.")
        return None

    print(f"  Loading gradients: {bval_f}")
    bvals, bvecs = read_bvals_bvecs(bval_f, bvec_f)
    gtab   = gradient_table(bvals, bvecs)
    data   = dti_img.get_fdata()
    affine = dti_img.affine

    print("  Brain masking …")
    _, mask = median_otsu(data[..., 0], median_radius=2, numpass=1)

    print("  Fitting tensor model …")
    tensor_model = TensorModel(gtab)
    tensor_fit   = tensor_model.fit(data, mask=mask)
    fa_map       = tensor_fit.fa
    print(f"    FA ∈ [{fa_map.min():.3f}, {fa_map.max():.3f}]")
    fa_img = nib.Nifti1Image(fa_map, affine)
    nib.save(fa_img, str(OUT_DIR / "fa_map.nii.gz"))

    print(f"  Deterministic tractography (FA > {FA_THRESH}) …")
    stop   = ThresholdStoppingCriterion(fa_map, FA_THRESH)
    sphere = get_sphere("repulsion724")
    peaks  = peaks_from_model(
        model=tensor_model, data=data, sphere=sphere, mask=mask,
        relative_peak_threshold=0.5, min_separation_angle=25,
        npeaks=5, normalize_peaks=True,
    )
    seeds = tracking_utils.seeds_from_mask(fa_map > FA_THRESH, affine, density=1)
    streamlines = Streamlines(
        LocalTracking(peaks, stop, seeds, affine, step_size=0.5, max_cross=1)
    )
    print(f"    Generated {len(streamlines):,} streamlines")
    return streamlines, fa_img, tensor_fit


def build_sc_matrix(streamlines, atlas_img, atlas_labels, dti_affine):
    """
    Build SC matrix.  DIPY LocalTracking with an affine already outputs
    streamlines in world-mm coordinates → only apply inv_atlas (4×4), never
    apply dti_affine again.
    """
    if not DIPY_AVAILABLE or streamlines is None:
        return np.zeros((len(atlas_labels), len(atlas_labels)))

    print("  Building structural connectivity matrix …")
    atlas_data  = atlas_img.get_fdata().astype(int)
    atlas_shape = atlas_data.shape
    inv_atlas   = np.linalg.inv(atlas_img.affine)
    n_roi       = int(atlas_data.max())
    sc          = np.zeros((n_roi, n_roi), dtype=float)

    def world_to_label(pt):
        vh = inv_atlas @ np.array([pt[0], pt[1], pt[2], 1.0])
        vi = np.round(vh[:3]).astype(int)
        if all(0 <= vi[a] < atlas_shape[a] for a in range(3)):
            return int(atlas_data[vi[0], vi[1], vi[2]])
        return 0

    for sl in streamlines:
        if len(sl) < 2: continue
        r0, r1 = world_to_label(sl[0]), world_to_label(sl[-1])
        if r0 > 0 and r1 > 0 and r0 != r1:
            sc[r0-1, r1-1] += 1
            sc[r1-1, r0-1] += 1

    if sc.max() > 0: sc /= sc.max()
    np.save(str(OUT_DIR / "sc_matrix.npy"), sc)
    print(f"    SC matrix: {sc.shape}  saved sc_matrix.npy")
    return sc


# ═════════════════════════════════════════════════════════════════════════════
#  MODULE 7 – MULTIMODAL INTEGRATION
# ═════════════════════════════════════════════════════════════════════════════

def integrate_fc_sc(fc_matrix, sc_matrix, atlas_labels):
    print("\n── MODULE 7: MULTIMODAL INTEGRATION ─────────────────────────────")
    n  = min(fc_matrix.shape[0], sc_matrix.shape[0])
    fc = fc_matrix[:n, :n].copy(); np.fill_diagonal(fc, 0)
    sc = sc_matrix[:n, :n].copy(); np.fill_diagonal(sc, 0)

    def z_thr(mat, z=1.5):
        return mat > mat.mean() + z*mat.std()

    fc_s = z_thr(fc); sc_s = z_thr(sc)
    fc_only = fc_s & ~sc_s
    sc_only = sc_s & ~fc_s
    core    = fc_s & sc_s
    print(f"  Strong FC:   {fc_s.sum()//2:4d}")
    print(f"  Strong SC:   {sc_s.sum()//2:4d}")
    print(f"  FC-only:     {fc_only.sum()//2:4d}")
    print(f"  SC-only:     {sc_only.sum()//2:4d}")
    print(f"  CORE (both): {core.sum()//2:4d}")
    np.save(str(OUT_DIR/"fc_only_mask.npy"), fc_only)
    np.save(str(OUT_DIR/"sc_only_mask.npy"), sc_only)
    np.save(str(OUT_DIR/"core_mask.npy"),    core)
    return {"fc_only":fc_only,"sc_only":sc_only,"core":core,"n":n,"labels":atlas_labels[:n]}


# ═════════════════════════════════════════════════════════════════════════════
#  MODULE 8 – GRAPH THEORY
# ═════════════════════════════════════════════════════════════════════════════

def build_brain_graph(fc_matrix, atlas_labels, threshold_pct=0.15):
    print("\n── MODULE 8: GRAPH-THEORETIC NETWORK ANALYSIS ───────────────────")
    n   = fc_matrix.shape[0]
    mat = np.abs(fc_matrix.copy()); np.fill_diagonal(mat, 0)
    thr = np.percentile(mat, (1-threshold_pct)*100)
    adj = np.where(mat >= thr, mat, 0)
    G   = nx.Graph()
    for i, lab in enumerate(atlas_labels[:n]):
        G.add_node(i, label=lab)
    for i in range(n):
        for j in range(i+1, n):
            if adj[i,j] > 0:
                G.add_edge(i, j, weight=float(adj[i,j]))
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def compute_graph_metrics(G, atlas_labels):
    """
    Compute graph metrics AND save to CSV.
    Single authoritative definition (no duplicate).
    """
    print("  Computing graph metrics …")
    deg_cent = nx.degree_centrality(G)
    bet_cent = nx.betweenness_centrality(G, weight="weight", normalized=True)
    clust    = nx.clustering(G, weight="weight")

    all_nodes = list(G.nodes())
    scores    = {n: (deg_cent.get(n,0)+bet_cent.get(n,0)+clust.get(n,0))/3
                 for n in all_nodes}
    top_hubs  = sorted(scores.items(), key=lambda x:-x[1])[:10]

    print("\n  ╔══════════════════════════════════════════════════════╗")
    print("  ║           TOP 10 HUB REGIONS (Brain Graph)          ║")
    print("  ╠══════════════════════════════════════════════════════╣")
    hub_rows = []
    for rank, (nid, score) in enumerate(top_hubs, 1):
        label = atlas_labels[nid] if nid < len(atlas_labels) else f"ROI-{nid}"
        print(f"  ║ {rank:2d}. {label[:40]:40s} {score:.3f} ║")
        hub_rows.append({"Type":"Hub","Rank":rank,"Region":label,
                         "HubScore":round(score,4),
                         "Degree":round(deg_cent.get(nid,0),4),
                         "Betweenness":round(bet_cent.get(nid,0),4),
                         "Clustering":round(clust.get(nid,0),4)})
    print("  ╚══════════════════════════════════════════════════════╝")

    if PANDAS_AVAILABLE:
        pd.DataFrame(hub_rows).to_csv(CSV_SUMMARY_FILE, mode="a",
                                      index=False, header=True)

    metrics = {
        "degree_centrality":      deg_cent,
        "betweenness_centrality": bet_cent,
        "clustering_coefficient": clust,
        "hub_ranking":            top_hubs,
        "scores":                 scores,
    }
    return metrics


def print_top_connections(fc_matrix, atlas_labels, top_n=10):
    """Single authoritative definition."""
    print("\n  ╔════════════════════════════════════════════════════════════════╗")
    print("  ║            TOP 10 STRONGEST FUNCTIONAL CONNECTIONS            ║")
    print("  ╠════════════════════════════════════════════════════════════════╣")
    mat = fc_matrix.copy(); np.fill_diagonal(mat, 0)
    idx = np.dstack(np.unravel_index(np.argsort(mat.ravel())[::-1], mat.shape))[0]
    seen, count, rows = set(), 0, []
    for i, j in idx:
        pair = tuple(sorted([int(i), int(j)]))
        if pair in seen: continue
        seen.add(pair)
        la = atlas_labels[i] if i < len(atlas_labels) else f"ROI-{i}"
        lb = atlas_labels[j] if j < len(atlas_labels) else f"ROI-{j}"
        r  = mat[i, j]
        print(f"  ║ {count+1:2d}. {la[:26]:26s} ↔ {lb[:26]:26s}  r={r:.3f} ║")
        rows.append({"Type":"FC","Rank":count+1,"RegionA":la,"RegionB":lb,"r":round(float(r),4)})
        count += 1
        if count >= top_n: break
    print("  ╚════════════════════════════════════════════════════════════════╝")
    if PANDAS_AVAILABLE:
        pd.DataFrame(rows).to_csv(CSV_SUMMARY_FILE, mode="a", index=False, header=False)


def print_top_active_regions(alff_img, atlas_img, atlas_labels, top_n=10):
    """Single authoritative definition."""
    print("\n  ╔══════════════════════════════════════════════════════╗")
    print("  ║        TOP 10 MOST ACTIVE REGIONS (Mean ALFF)       ║")
    print("  ╠══════════════════════════════════════════════════════╣")
    alff_r  = resample_to_img(alff_img, atlas_img, interpolation="continuous")
    alff_d  = alff_r.get_fdata()
    atlas_d = atlas_img.get_fdata().astype(int)
    roi_vals = {rid: alff_d[atlas_d==rid].mean()
                for rid in np.unique(atlas_d) if rid != 0}
    top  = sorted(roi_vals.items(), key=lambda x:-x[1])[:top_n]
    rows = []
    for rank, (rid, val) in enumerate(top, 1):
        label = atlas_labels[rid-1] if rid-1 < len(atlas_labels) else f"ROI-{rid}"
        print(f"  ║ {rank:2d}. {label[:40]:40s} {val:.4f} ║")
        rows.append({"Type":"ALFF","Rank":rank,"Region":label,"MeanALFF":round(float(val),4)})
    print("  ╚══════════════════════════════════════════════════════╝")
    if PANDAS_AVAILABLE:
        pd.DataFrame(rows).to_csv(CSV_SUMMARY_FILE, mode="a", index=False, header=False)


# ═════════════════════════════════════════════════════════════════════════════
#  MODULE 9 – ORIGINAL VISUALISATIONS
# ═════════════════════════════════════════════════════════════════════════════

def plot_activation_maps(alff_img, falff_img, reho_img, bold_img):
    print("\n── MODULE 9: VISUALISATION ──────────────────────────────────────")
    bg = image.mean_img(bold_img)
    fig, axes = plt.subplots(3, 1, figsize=(18, 12))
    fig.patch.set_facecolor("#0d0d0d")
    for ax, (img, title, cmap) in zip(axes, [
        (alff_img,  "ALFF – Amplitude of Low-Frequency Fluctuations", "hot"),
        (falff_img, "fALFF – Fractional ALFF (normalised)",           "YlOrRd"),
        (reho_img,  "ReHo – Regional Homogeneity (Kendall W)",        "plasma"),
    ]):
        try:
            plotting.plot_stat_map(img, bg_img=bg, display_mode="z",
                                   cut_coords=7, colorbar=True, cmap=cmap,
                                   title=title, axes=ax, black_bg=True, annotate=False)
        except Exception:
            ax.set_title(title, color="white"); ax.axis("off")
    plt.tight_layout()
    plt.savefig(str(OUT_DIR/"activation_maps.png"), dpi=150, bbox_inches="tight",
                facecolor="#0d0d0d"); plt.close()
    print("  Saved activation_maps.png")


def plot_glass_brains(seed_fc, bold_img):
    n = len(seed_fc)
    fig = plt.figure(figsize=(6*n, 6)); fig.patch.set_facecolor("#0d0d0d")
    for i, (name, img) in enumerate(seed_fc.items()):
        ax = fig.add_subplot(1, n, i+1)
        try:
            plotting.plot_glass_brain(img, threshold=0.3, colorbar=True,
                                      title=f"Seed FC: {name}", axes=ax,
                                      black_bg=True, plot_abs=False)
        except Exception:
            ax.set_title(name, color="white"); ax.axis("off")
    plt.tight_layout()
    plt.savefig(str(OUT_DIR/"glass_brain_seed_fc.png"), dpi=120,
                bbox_inches="tight", facecolor="#0d0d0d"); plt.close()
    print("  Saved glass_brain_seed_fc.png")


def plot_fc_heatmap(fc_matrix, atlas_labels, title="FC Matrix", fname="fc_heatmap.png"):
    n      = fc_matrix.shape[0]
    labels = [l[:18] for l in atlas_labels[:n]]
    fig, ax = plt.subplots(figsize=(16, 14))
    mask = np.eye(n, dtype=bool)
    sns.heatmap(fc_matrix, mask=mask, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                xticklabels=labels, yticklabels=labels, linewidths=0, ax=ax,
                cbar_kws={"shrink":0.5})
    ax.set_title(title, fontsize=13, pad=10)
    plt.xticks(fontsize=5, rotation=90); plt.yticks(fontsize=5)
    plt.tight_layout()
    plt.savefig(str(OUT_DIR/fname), dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved {fname}")


def plot_multimodal_comparison(fc_matrix, sc_matrix, atlas_labels, n):
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    kw = dict(cmap="RdBu_r", center=0, vmin=-0.5, vmax=0.5,
              xticklabels=False, yticklabels=False, linewidths=0)
    sns.heatmap(fc_matrix[:n,:n], ax=axes[0], **kw, cbar_kws={"label":"Pearson r"})
    axes[0].set_title("Functional Connectivity (FC)")
    sns.heatmap(sc_matrix[:n,:n], ax=axes[1], cmap="viridis", vmin=0, vmax=1,
                xticklabels=False, yticklabels=False, linewidths=0,
                cbar_kws={"label":"Norm. streamlines"})
    axes[1].set_title("Structural Connectivity (SC)")
    diff = np.abs(fc_matrix[:n,:n]) - sc_matrix[:n,:n]
    sns.heatmap(diff, ax=axes[2], cmap="coolwarm", center=0,
                xticklabels=False, yticklabels=False, linewidths=0,
                cbar_kws={"label":"|FC|–SC"})
    axes[2].set_title("FC–SC Discordance")
    plt.suptitle("Multimodal Connectivity Comparison", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(str(OUT_DIR/"multimodal_comparison.png"), dpi=150,
                bbox_inches="tight"); plt.close()
    print("  Saved multimodal_comparison.png")


def plot_network_graph(G, metrics, atlas_labels):
    fig, ax = plt.subplots(figsize=(16, 12))
    fig.patch.set_facecolor("#111"); ax.set_facecolor("#111")
    pos    = nx.spring_layout(G, seed=42, k=0.4)
    bet    = metrics["betweenness_centrality"]
    deg    = metrics["degree_centrality"]
    sizes  = [3000*deg.get(n,0)+50  for n in G.nodes()]
    colors = [bet.get(n,0)           for n in G.nodes()]
    widths = [G[u][v]["weight"]*2    for u,v in G.edges()]
    nx.draw_networkx_edges(G, pos, ax=ax, width=widths, edge_color="steelblue", alpha=0.25)
    nc = nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes, node_color=colors,
                                 cmap=plt.cm.plasma, alpha=0.9)
    top_ids = {n for n,_ in metrics["hub_ranking"]}
    hub_labels = {n: atlas_labels[n][:14] if n<len(atlas_labels) else f"ROI-{n}"
                  for n in top_ids}
    nx.draw_networkx_labels(G, pos, labels=hub_labels,
                            font_size=7, font_color="white", ax=ax)
    plt.colorbar(nc, ax=ax, label="Betweenness Centrality", fraction=0.03, pad=0.02)
    ax.set_title("Brain Connectivity Graph\n(node size∝degree, colour∝betweenness)",
                 color="white", fontsize=13)
    ax.axis("off")
    plt.savefig(str(OUT_DIR/"brain_network_graph.png"), dpi=150,
                bbox_inches="tight", facecolor="#111"); plt.close()
    print("  Saved brain_network_graph.png")


def plot_hub_bar(metrics, atlas_labels):
    top    = metrics["hub_ranking"]
    labels = [atlas_labels[n][:35] if n<len(atlas_labels) else f"ROI-{n}" for n,_ in top]
    scores = [s for _,s in top]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(labels[::-1], scores[::-1],
                   color=plt.cm.viridis(np.linspace(0.3, 0.9, len(top))))
    ax.set_xlabel("Hub Score"); ax.set_title("Top 10 Brain Hub Regions", fontsize=13)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    for bar, s in zip(bars, scores[::-1]):
        ax.text(bar.get_width()+0.001, bar.get_y()+bar.get_height()/2,
                f"{s:.3f}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(str(OUT_DIR/"hub_regions_bar.png"), dpi=130, bbox_inches="tight"); plt.close()
    print("  Saved hub_regions_bar.png")


def plot_ica_networks(components_img, matched, bold_img):
    bg  = image.mean_img(bold_img)
    n   = len(matched)
    fig, axes = plt.subplots(n, 1, figsize=(18, 5*n))
    if n == 1: axes = [axes]
    fig.patch.set_facecolor("#0d0d0d")
    for ax, (net, info) in zip(axes, matched.items()):
        comp_img = image.index_img(components_img, info["component"])
        try:
            plotting.plot_stat_map(comp_img, bg_img=bg, display_mode="z",
                                   cut_coords=6, colorbar=True, cmap="cold_hot",
                                   title=f"{net}  (comp {info['component']}, r={info['r']:.2f})",
                                   axes=ax, black_bg=True, annotate=False)
        except Exception:
            ax.set_title(net, color="white"); ax.axis("off")
    plt.tight_layout()
    plt.savefig(str(OUT_DIR/"ica_networks_DMN_SAL_CEN.png"), dpi=130,
                bbox_inches="tight", facecolor="#0d0d0d"); plt.close()
    print("  Saved ica_networks_DMN_SAL_CEN.png")


def plot_fa_map(fa_img):
    if fa_img is None: return
    fig, ax = plt.subplots(figsize=(16, 4))
    try:
        plotting.plot_anat(fa_img, display_mode="z", cut_coords=8,
                           title="DTI Fractional Anisotropy (FA) Map",
                           cmap="bone", colorbar=True, axes=ax)
    except Exception:
        ax.set_title("FA Map"); ax.axis("off")
    plt.tight_layout()
    plt.savefig(str(OUT_DIR/"dti_fa_map.png"), dpi=130, bbox_inches="tight"); plt.close()
    print("  Saved dti_fa_map.png")


def plot_interactive_network(G, metrics, atlas_labels):
    if not PLOTLY_AVAILABLE: return
    print("  Building interactive Plotly network graph …")
    pos = nx.spring_layout(G, seed=42)
    bet = metrics["betweenness_centrality"]
    deg = metrics["degree_centrality"]
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0,y0 = pos[u]; x1,y1 = pos[v]
        edge_x += [x0,x1,None]; edge_y += [y0,y1,None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines",
                             line=dict(width=0.6,color="#888"), hoverinfo="none")
    node_x     = [pos[n][0] for n in G.nodes()]
    node_y     = [pos[n][1] for n in G.nodes()]
    node_text  = [atlas_labels[n][:28] if n<len(atlas_labels) else f"ROI-{n}"
                  for n in G.nodes()]
    node_hover = [
        f"<b>{atlas_labels[n][:35] if n<len(atlas_labels) else f'ROI-{n}'}</b><br>"
        f"Degree: {deg.get(n,0):.3f}<br>"
        f"Betweenness: {bet.get(n,0):.3f}<br>"
        f"Clustering: {metrics['clustering_coefficient'].get(n,0):.3f}"
        for n in G.nodes()
    ]
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        hovertext=node_hover, hoverinfo="text", text=node_text,
        textfont=dict(size=7),
        marker=dict(size=[20+60*deg.get(n,0) for n in G.nodes()],
                    color=[bet.get(n,0) for n in G.nodes()],
                    colorscale="Plasma", showscale=True,
                    colorbar=dict(title="Betweenness"),
                    line=dict(width=0.5, color="#fff"))
    )
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title="Interactive Brain Connectivity Network",
                        showlegend=False, hovermode="closest",
                        margin=dict(b=20,l=5,r=5,t=40),
                        xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
                        yaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
                        paper_bgcolor="#111", plot_bgcolor="#111",
                        font=dict(color="white")))
    fig.write_html(str(OUT_DIR/"interactive_network.html"))
    print("  Saved interactive_network.html")


def plot_network_summary_dashboard(fc_matrix, sc_matrix, metrics, alff_img, atlas_labels):
    n   = min(fc_matrix.shape[0], sc_matrix.shape[0], 30)
    fig = plt.figure(figsize=(20, 16))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    ax_a = fig.add_subplot(gs[0,0])
    sns.heatmap(fc_matrix[:n,:n], cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                ax=ax_a, xticklabels=False, yticklabels=False, linewidths=0)
    ax_a.set_title("Functional Connectivity Matrix")
    ax_b = fig.add_subplot(gs[0,1])
    sns.heatmap(sc_matrix[:n,:n], cmap="viridis", vmin=0, vmax=1,
                ax=ax_b, xticklabels=False, yticklabels=False, linewidths=0)
    ax_b.set_title("Structural Connectivity Matrix")
    ax_c = fig.add_subplot(gs[1,0])
    top  = metrics["hub_ranking"][:8]
    yl   = [atlas_labels[n_][:22] if n_<len(atlas_labels) else f"ROI-{n_}" for n_,_ in top]
    sc_  = [s for _,s in top]
    ax_c.barh(yl[::-1], sc_[::-1], color=plt.cm.plasma(np.linspace(0.2,0.8,len(top))))
    ax_c.set_xlabel("Hub Score"); ax_c.set_title("Top Hub Regions")
    ax_c.spines["top"].set_visible(False); ax_c.spines["right"].set_visible(False)
    ax_d = fig.add_subplot(gs[1,1])
    alff_d = alff_img.get_fdata().ravel()
    ax_d.hist(alff_d[alff_d>0], bins=60, color="tomato", alpha=0.8, edgecolor="none")
    ax_d.set_xlabel("ALFF value"); ax_d.set_ylabel("Voxel count")
    ax_d.set_title("ALFF Distribution")
    ax_d.spines["top"].set_visible(False); ax_d.spines["right"].set_visible(False)
    fig.suptitle("Multimodal Brain Connectivity – Summary Dashboard", fontsize=15, weight="bold")
    plt.savefig(str(OUT_DIR/"summary_dashboard.png"), dpi=150, bbox_inches="tight"); plt.close()
    print("  Saved summary_dashboard.png")


# ═════════════════════════════════════════════════════════════════════════════
#  MODULE 10 – EXTENDED NETWORK & CONNECTIVITY VISUALISATIONS
# ═════════════════════════════════════════════════════════════════════════════

# ─── 10-A  Community detection + coloured network ────────────────────────────
def detect_communities(G):
    """Greedy modularity maximisation → community partition."""
    comms = list(nx_community.greedy_modularity_communities(G, weight="weight"))
    node2comm = {}
    for ci, comm in enumerate(comms):
        for n in comm:
            node2comm[n] = ci
    modularity = nx_community.modularity(G, comms, weight="weight")
    print(f"  Communities detected: {len(comms)}  (modularity Q={modularity:.3f})")
    return node2comm, comms, modularity


def plot_community_graph(G, metrics, atlas_labels, node2comm):
    """Spring-layout graph coloured by community membership."""
    print("  Plotting community graph …")
    n_comm = max(node2comm.values()) + 1
    cmap   = plt.cm.get_cmap("tab20", n_comm)
    colors = [cmap(node2comm.get(n, 0)) for n in G.nodes()]
    sizes  = [2500*metrics["degree_centrality"].get(n,0)+60 for n in G.nodes()]

    pos = nx.spring_layout(G, seed=42, k=0.45)
    fig, ax = plt.subplots(figsize=(16, 12))
    fig.patch.set_facecolor("#111"); ax.set_facecolor("#111")

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.15,
                           edge_color="white",
                           width=[G[u][v]["weight"]*1.8 for u,v in G.edges()])
    nc = nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes,
                                 node_color=colors, alpha=0.92)

    # Label all nodes (small font)
    all_labels = {n: atlas_labels[n][:12] if n<len(atlas_labels) else f"R{n}"
                  for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=all_labels,
                            font_size=5.5, font_color="white", ax=ax)

    patches = [mpatches.Patch(color=cmap(i), label=f"Community {i+1}")
               for i in range(n_comm)]
    ax.legend(handles=patches, loc="lower right", fontsize=7,
              facecolor="#222", edgecolor="none", labelcolor="white")
    ax.set_title("Brain Network – Community Structure\n"
                 "(colour = community, size = degree centrality)",
                 color="white", fontsize=13)
    ax.axis("off")
    plt.savefig(str(OUT_DIR/"community_graph.png"), dpi=150,
                bbox_inches="tight", facecolor="#111"); plt.close()
    print("  Saved community_graph.png")


# ─── 10-B  Community-reordered FC heatmap ────────────────────────────────────
def plot_community_fc_heatmap(fc_matrix, atlas_labels, node2comm):
    """Reorder FC matrix rows/cols by community → block structure visible."""
    print("  Plotting community-reordered FC heatmap …")
    order  = sorted(range(fc_matrix.shape[0]), key=lambda x: node2comm.get(x, 0))
    fc_ord = fc_matrix[np.ix_(order, order)]
    labels = [atlas_labels[i][:14] if i<len(atlas_labels) else f"ROI-{i}"
              for i in order]

    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(fc_ord, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                xticklabels=labels, yticklabels=labels, linewidths=0, ax=ax,
                cbar_kws={"shrink":0.5, "label":"Pearson r"})
    ax.set_title("FC Matrix Reordered by Community", fontsize=13)
    plt.xticks(fontsize=5, rotation=90); plt.yticks(fontsize=5)

    # Draw community boundary lines
    sizes = [sum(1 for v in node2comm.values() if v==ci)
             for ci in range(max(node2comm.values())+1)]
    cum = 0
    for s in sizes[:-1]:
        cum += s
        ax.axhline(cum, color="gold", lw=0.8)
        ax.axvline(cum, color="gold", lw=0.8)

    plt.tight_layout()
    plt.savefig(str(OUT_DIR/"community_fc_heatmap.png"), dpi=150,
                bbox_inches="tight"); plt.close()
    print("  Saved community_fc_heatmap.png")


# ─── 10-C  Circular / chord connectivity diagram ─────────────────────────────
def plot_circular_connectivity(fc_matrix, atlas_labels, top_k=30):
    """
    Circular layout: nodes arranged on a circle, top-k edges drawn as arcs.
    Colour encodes connection strength; line width encodes |r|.
    """
    print("  Plotting circular connectivity diagram …")
    n      = fc_matrix.shape[0]
    labels = atlas_labels[:n]
    mat    = fc_matrix.copy(); np.fill_diagonal(mat, 0)

    # Keep only top_k strongest connections
    flat_idx = np.argsort(np.abs(mat).ravel())[::-1]
    edges = []
    seen  = set()
    for idx in flat_idx:
        i, j = divmod(idx, n)
        pair  = tuple(sorted([i, j]))
        if pair in seen: continue
        seen.add(pair)
        edges.append((i, j, float(mat[i,j])))
        if len(edges) >= top_k: break

    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    xs = np.cos(angles); ys = np.sin(angles)

    fig, ax = plt.subplots(figsize=(14, 14))
    fig.patch.set_facecolor("#0a0a0a"); ax.set_facecolor("#0a0a0a")
    ax.set_aspect("equal"); ax.axis("off")

    vmax = max(abs(e[2]) for e in edges) if edges else 1.0
    cmap_pos = plt.cm.Reds; cmap_neg = plt.cm.Blues

    for i, j, r in edges:
        xi, yi = xs[i], ys[i]; xj, yj = xs[j], ys[j]
        # Bezier-like arc through origin-scaled midpoint
        mx, my = (xi+xj)*0.5*0.4, (yi+yj)*0.5*0.4
        t  = np.linspace(0, 1, 60)
        bx = (1-t)**2*xi + 2*(1-t)*t*mx + t**2*xj
        by = (1-t)**2*yi + 2*(1-t)*t*my + t**2*yj
        color = cmap_pos(abs(r)/vmax) if r > 0 else cmap_neg(abs(r)/vmax)
        lw    = 0.5 + 2.5*abs(r)/vmax
        ax.plot(bx, by, color=color, lw=lw, alpha=0.7, zorder=1)

    # Draw nodes
    ax.scatter(xs, ys, s=80, c="white", zorder=3, linewidths=0.5, edgecolors="#555")

    # Labels
    for i, (x, y, lbl) in enumerate(zip(xs, ys, labels)):
        angle_deg = np.degrees(angles[i])
        ha = "left" if -90 < angle_deg < 90 else "right"
        ax.text(x*1.12, y*1.12, lbl[:18], fontsize=6, color="#ddd",
                ha=ha, va="center",
                rotation=angle_deg if ha=="left" else angle_deg+180,
                rotation_mode="anchor")

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0],[0],color=cmap_pos(0.8),lw=2,label="Positive FC"),
        Line2D([0],[0],color=cmap_neg(0.8),lw=2,label="Negative FC"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9,
              facecolor="#111", edgecolor="none", labelcolor="white")
    ax.set_title(f"Circular Connectivity Diagram (top {top_k} edges)",
                 color="white", fontsize=14, pad=20)
    plt.savefig(str(OUT_DIR/"circular_connectivity.png"), dpi=150,
                bbox_inches="tight", facecolor="#0a0a0a"); plt.close()
    print("  Saved circular_connectivity.png")


# ─── 10-D  Centrality radar / spider chart ───────────────────────────────────
def plot_centrality_radar(metrics, atlas_labels, top_n=8):
    """Spider / radar chart comparing 3 centrality measures for top-N hub regions."""
    print("  Plotting centrality radar chart …")
    top     = metrics["hub_ranking"][:top_n]
    node_ids  = [n for n,_ in top]
    reg_names = [atlas_labels[n][:20] if n<len(atlas_labels) else f"ROI-{n}"
                 for n in node_ids]

    deg_vals = [metrics["degree_centrality"].get(n,0)     for n in node_ids]
    bet_vals = [metrics["betweenness_centrality"].get(n,0) for n in node_ids]
    clu_vals = [metrics["clustering_coefficient"].get(n,0) for n in node_ids]

    # Normalise each to [0,1]
    def norm(v): mx = max(v) if max(v)>0 else 1; return [x/mx for x in v]
    deg_n, bet_n, clu_n = norm(deg_vals), norm(bet_vals), norm(clu_vals)

    categories = ["Degree", "Betweenness", "Clustering"]
    N_cat = len(categories)
    angles = np.linspace(0, 2*np.pi, N_cat, endpoint=False).tolist()
    angles += angles[:1]                           # close polygon

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#0d0d0d"); ax.set_facecolor("#0d0d0d")
    cmap = plt.cm.plasma

    for idx, (name, dv, bv, cv) in enumerate(zip(reg_names,deg_n,bet_n,clu_n)):
        values = [dv, bv, cv, dv]
        color  = cmap(idx/top_n)
        ax.plot(angles, values, "o-", lw=1.8, color=color, label=name, markersize=4)
        ax.fill(angles, values, alpha=0.07, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color="white", fontsize=12)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25","0.50","0.75","1.0"], color="grey", fontsize=7)
    ax.tick_params(colors="white")
    ax.spines["polar"].set_color("#444")
    ax.grid(color="#333", linestyle="--", lw=0.5)

    legend = ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1),
                       fontsize=8, facecolor="#111", edgecolor="none")
    for text in legend.get_texts(): text.set_color("white")
    ax.set_title("Hub Region Centrality Comparison\n(normalised per measure)",
                 color="white", fontsize=13, pad=20)

    plt.savefig(str(OUT_DIR/"centrality_radar.png"), dpi=150,
                bbox_inches="tight", facecolor="#0d0d0d"); plt.close()
    print("  Saved centrality_radar.png")


# ─── 10-E  FC vs SC scatter plot ─────────────────────────────────────────────
def plot_fc_sc_scatter(fc_matrix, sc_matrix, atlas_labels):
    """Scatter plot: each point = one (i,j) edge; x=SC weight, y=FC strength."""
    print("  Plotting FC vs SC scatter …")
    n   = min(fc_matrix.shape[0], sc_matrix.shape[0])
    fc  = fc_matrix[:n,:n].copy(); np.fill_diagonal(fc, 0)
    sc  = sc_matrix[:n,:n].copy(); np.fill_diagonal(sc, 0)

    fc_vals = fc[np.triu_indices(n, k=1)]
    sc_vals = sc[np.triu_indices(n, k=1)]

    # Colour by FC sign
    colors = np.where(fc_vals > 0, "#e06c75", "#61afef")

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(sc_vals, fc_vals, c=colors, alpha=0.45, s=18, linewidths=0)
    # Linear trend
    if sc_vals.std() > 0:
        m, b = np.polyfit(sc_vals, fc_vals, 1)
        xfit = np.linspace(sc_vals.min(), sc_vals.max(), 100)
        ax.plot(xfit, m*xfit+b, "w--", lw=1.5, label=f"trend (slope={m:.2f})")
        ax.legend(fontsize=9)

    ax.axhline(0, color="grey", lw=0.5, ls=":")
    ax.set_xlabel("Structural Connectivity (normalised streamlines)", fontsize=11)
    ax.set_ylabel("Functional Connectivity (Pearson r)", fontsize=11)
    ax.set_title("FC–SC Relationship\n(one dot per ROI pair)", fontsize=12)
    ax.set_facecolor("#1a1a2e"); fig.patch.set_facecolor("#1a1a2e")
    ax.tick_params(colors="white"); ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white"); ax.title.set_color("white")
    for spine in ax.spines.values(): spine.set_color("#444")

    patches = [mpatches.Patch(color="#e06c75", label="Positive FC"),
               mpatches.Patch(color="#61afef", label="Negative FC")]
    leg = ax.legend(handles=patches, fontsize=9, facecolor="#111", edgecolor="none")
    for t in leg.get_texts(): t.set_color("white")

    plt.tight_layout()
    plt.savefig(str(OUT_DIR/"fc_sc_scatter.png"), dpi=150,
                bbox_inches="tight", facecolor="#1a1a2e"); plt.close()
    print("  Saved fc_sc_scatter.png")


# ─── 10-F  Degree distribution ───────────────────────────────────────────────
def plot_degree_distribution(G, atlas_labels):
    """Log-log degree distribution with power-law reference line."""
    print("  Plotting degree distribution …")
    degrees = sorted([d for _, d in G.degree()], reverse=True)
    unique_degs, counts = np.unique(degrees, return_counts=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes: ax.set_facecolor("#1a1a2e")

    # Linear scale
    axes[0].bar(unique_degs, counts, color="#c678dd", alpha=0.8, edgecolor="none")
    axes[0].set_xlabel("Degree"); axes[0].set_ylabel("Count")
    axes[0].set_title("Degree Distribution (linear)", color="white")

    # Log-log scale + power-law
    axes[1].scatter(unique_degs, counts, color="#e5c07b", s=40, zorder=3)
    if len(unique_degs) > 2 and unique_degs[0] > 0:
        log_x = np.log(unique_degs.astype(float))
        log_y = np.log(counts.astype(float))
        m, b  = np.polyfit(log_x, log_y, 1)
        xfit  = np.linspace(unique_degs.min(), unique_degs.max(), 100)
        axes[1].plot(xfit, np.exp(b)*xfit**m, "r--", lw=1.5,
                     label=f"power law γ={-m:.2f}")
        axes[1].legend(fontsize=9, facecolor="#111",
                        edgecolor="none", labelcolor="white")
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("Degree (log)"); axes[1].set_ylabel("Count (log)")
    axes[1].set_title("Degree Distribution (log-log)", color="white")

    for ax in axes:
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
        for spine in ax.spines.values(): spine.set_color("#444")

    plt.suptitle("Network Degree Distribution", color="white", fontsize=13)
    plt.tight_layout()
    plt.savefig(str(OUT_DIR/"degree_distribution.png"), dpi=150,
                bbox_inches="tight", facecolor="#1a1a2e"); plt.close()
    print("  Saved degree_distribution.png")


# ─── 10-G  Rich-club coefficient ─────────────────────────────────────────────
def plot_rich_club(G):
    """Rich-club coefficient ϕ(k): fraction of possible edges among nodes with degree ≥ k."""
    print("  Plotting rich-club coefficient …")
    rc = nx.rich_club_coefficient(G, normalized=False)
    if not rc:
        print("  [SKIP] Not enough data for rich-club curve.")
        return
    ks   = sorted(rc.keys())
    vals = [rc[k] for k in ks]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#1a1a2e"); ax.set_facecolor("#1a1a2e")
    ax.plot(ks, vals, "o-", color="#98c379", lw=2, markersize=5)
    ax.fill_between(ks, vals, alpha=0.2, color="#98c379")
    ax.set_xlabel("Degree threshold k", fontsize=11)
    ax.set_ylabel("Rich-club coefficient ϕ(k)", fontsize=11)
    ax.set_title("Rich-Club Coefficient\n(high ϕ = hubs preferentially connect to hubs)",
                 fontsize=12)
    ax.tick_params(colors="white"); ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white"); ax.title.set_color("white")
    for spine in ax.spines.values(): spine.set_color("#444")
    ax.grid(color="#333", ls="--", lw=0.5)
    plt.tight_layout()
    plt.savefig(str(OUT_DIR/"rich_club.png"), dpi=150,
                bbox_inches="tight", facecolor="#1a1a2e"); plt.close()
    print("  Saved rich_club.png")


# ─── 10-H  6-panel network analysis dashboard ────────────────────────────────
def plot_network_analysis_dashboard(G, metrics, fc_matrix, atlas_labels, node2comm):
    """6-panel comprehensive network analysis figure."""
    print("  Plotting 6-panel network analysis dashboard …")
    n   = fc_matrix.shape[0]
    deg  = metrics["degree_centrality"]
    bet  = metrics["betweenness_centrality"]
    clu  = metrics["clustering_coefficient"]
    top  = metrics["hub_ranking"]

    fig  = plt.figure(figsize=(22, 18))
    gs   = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ─ Panel A: FC matrix ───────────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    sns.heatmap(fc_matrix, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                ax=ax_a, xticklabels=False, yticklabels=False, linewidths=0,
                cbar_kws={"shrink":0.7,"label":"r"})
    ax_a.set_title("A  Functional Connectivity Matrix", fontsize=10, fontweight="bold")

    # ─ Panel B: Degree centrality bar ───────────────────────────────────────
    ax_b  = fig.add_subplot(gs[0, 1])
    top8  = top[:8]
    ylbls = [atlas_labels[ni][:18] if ni<len(atlas_labels) else f"ROI-{ni}" for ni,_ in top8]
    dvals = [deg.get(ni,0) for ni,_ in top8]
    ax_b.barh(ylbls[::-1], dvals[::-1], color=plt.cm.viridis(np.linspace(0.2,0.85,8)))
    ax_b.set_xlabel("Degree Centrality"); ax_b.set_title("B  Degree Centrality (Top 8)",
                                                          fontsize=10, fontweight="bold")
    ax_b.spines["top"].set_visible(False); ax_b.spines["right"].set_visible(False)

    # ─ Panel C: Betweenness bar ──────────────────────────────────────────────
    ax_c  = fig.add_subplot(gs[0, 2])
    bvals = [bet.get(ni,0) for ni,_ in top8]
    ax_c.barh(ylbls[::-1], bvals[::-1], color=plt.cm.magma(np.linspace(0.25,0.9,8)))
    ax_c.set_xlabel("Betweenness Centrality"); ax_c.set_title("C  Betweenness (Top 8)",
                                                               fontsize=10, fontweight="bold")
    ax_c.spines["top"].set_visible(False); ax_c.spines["right"].set_visible(False)

    # ─ Panel D: Clustering coefficient bar ──────────────────────────────────
    ax_d  = fig.add_subplot(gs[1, 0])
    cvals = [clu.get(ni,0) for ni,_ in top8]
    ax_d.barh(ylbls[::-1], cvals[::-1], color=plt.cm.cool(np.linspace(0.2,0.9,8)))
    ax_d.set_xlabel("Clustering Coefficient"); ax_d.set_title("D  Clustering (Top 8)",
                                                               fontsize=10, fontweight="bold")
    ax_d.spines["top"].set_visible(False); ax_d.spines["right"].set_visible(False)

    # ─ Panel E: Degree distribution ─────────────────────────────────────────
    ax_e = fig.add_subplot(gs[1, 1])
    degs = [d for _,d in G.degree()]
    ax_e.hist(degs, bins=15, color="#e06c75", edgecolor="none", alpha=0.85)
    ax_e.set_xlabel("Degree"); ax_e.set_ylabel("Count")
    ax_e.set_title("E  Degree Distribution", fontsize=10, fontweight="bold")
    ax_e.spines["top"].set_visible(False); ax_e.spines["right"].set_visible(False)

    # ─ Panel F: Community network (spring) ──────────────────────────────────
    ax_f   = fig.add_subplot(gs[1, 2])
    ax_f.set_facecolor("#111")
    n_comm = max(node2comm.values())+1
    cmapc  = plt.cm.get_cmap("tab20", n_comm)
    cols   = [cmapc(node2comm.get(nd,0)) for nd in G.nodes()]
    szs    = [1200*deg.get(nd,0)+30 for nd in G.nodes()]
    pos    = nx.spring_layout(G, seed=42, k=0.35)
    nx.draw_networkx_edges(G, pos, ax=ax_f, alpha=0.12, edge_color="white",
                           width=[G[u][v]["weight"]*1.5 for u,v in G.edges()])
    nx.draw_networkx_nodes(G, pos, ax=ax_f, node_size=szs,
                            node_color=cols, alpha=0.9)
    nx.draw_networkx_labels(G, pos,
                            labels={nd: atlas_labels[nd][:9] if nd<len(atlas_labels)
                                    else f"R{nd}" for nd in G.nodes()},
                            font_size=4.5, font_color="white", ax=ax_f)
    ax_f.set_title("F  Community Structure", fontsize=10, fontweight="bold", color="black")
    ax_f.axis("off")

    # ─ Panel G: Hub composite score ─────────────────────────────────────────
    ax_g   = fig.add_subplot(gs[2, 0])
    top10  = top[:10]
    g_lbls = [atlas_labels[ni][:20] if ni<len(atlas_labels) else f"ROI-{ni}" for ni,_ in top10]
    g_vals = [s for _,s in top10]
    bars   = ax_g.barh(g_lbls[::-1], g_vals[::-1],
                        color=plt.cm.plasma(np.linspace(0.15,0.9,10)))
    for bar, v in zip(bars, g_vals[::-1]):
        ax_g.text(bar.get_width()+0.001, bar.get_y()+bar.get_height()/2,
                  f"{v:.3f}", va="center", fontsize=7)
    ax_g.set_xlabel("Hub Score"); ax_g.set_title("G  Top 10 Hubs (composite)",
                                                  fontsize=10, fontweight="bold")
    ax_g.spines["top"].set_visible(False); ax_g.spines["right"].set_visible(False)

    # ─ Panel H: FC edge weight histogram ────────────────────────────────────
    ax_h = fig.add_subplot(gs[2, 1])
    fc_mat = fc_matrix.copy(); np.fill_diagonal(fc_mat, 0)
    fc_vals = fc_mat[np.triu_indices(n, k=1)]
    ax_h.hist(fc_vals, bins=50, color="#61afef", edgecolor="none", alpha=0.85)
    ax_h.axvline(0, color="white", lw=0.8, ls="--")
    mu, sd = fc_vals.mean(), fc_vals.std()
    ax_h.set_xlabel("FC (Pearson r)"); ax_h.set_ylabel("Edge count")
    ax_h.set_title(f"H  FC Distribution  μ={mu:.3f} σ={sd:.3f}",
                   fontsize=10, fontweight="bold")
    ax_h.spines["top"].set_visible(False); ax_h.spines["right"].set_visible(False)

    # ─ Panel I: Centrality scatter (degree vs betweenness) ──────────────────
    ax_i = fig.add_subplot(gs[2, 2])
    all_d = [deg.get(nd,0) for nd in G.nodes()]
    all_b = [bet.get(nd,0) for nd in G.nodes()]
    all_c = [clu.get(nd,0) for nd in G.nodes()]
    sc_plot = ax_i.scatter(all_d, all_b, c=all_c, cmap="viridis",
                            s=60, alpha=0.85)
    plt.colorbar(sc_plot, ax=ax_i, label="Clustering")
    top5 = top[:5]
    for nid, _ in top5:
        lbl = atlas_labels[nid][:12] if nid<len(atlas_labels) else f"R{nid}"
        ax_i.annotate(lbl, (deg.get(nid,0), bet.get(nid,0)),
                      fontsize=6.5, ha="center", va="bottom",
                      xytext=(0, 5), textcoords="offset points")
    ax_i.set_xlabel("Degree Centrality"); ax_i.set_ylabel("Betweenness Centrality")
    ax_i.set_title("I  Degree vs Betweenness\n(colour = clustering)",
                   fontsize=10, fontweight="bold")
    ax_i.spines["top"].set_visible(False); ax_i.spines["right"].set_visible(False)

    fig.suptitle("Comprehensive Network Analysis Dashboard", fontsize=16,
                 fontweight="bold", y=0.99)
    plt.savefig(str(OUT_DIR/"network_analysis_dashboard.png"), dpi=150,
                bbox_inches="tight"); plt.close()
    print("  Saved network_analysis_dashboard.png")


# ─── 10-I  Interactive Plotly community graph ─────────────────────────────────
def plot_interactive_community_network(G, metrics, atlas_labels, node2comm):
    """Interactive Plotly graph with community colouring and rich hover info."""
    if not PLOTLY_AVAILABLE: return
    print("  Building interactive community network (Plotly) …")
    pos    = nx.spring_layout(G, seed=42)
    deg    = metrics["degree_centrality"]
    bet    = metrics["betweenness_centrality"]
    clu    = metrics["clustering_coefficient"]
    n_comm = max(node2comm.values())+1

    import plotly.colors as pc
    palette = pc.qualitative.Plotly + pc.qualitative.D3
    while len(palette) < n_comm: palette += palette

    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0,y0=pos[u]; x1,y1=pos[v]
        edge_x+=[x0,x1,None]; edge_y+=[y0,y1,None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines",
                             line=dict(width=0.5,color="rgba(180,180,180,0.25)"),
                             hoverinfo="none")

    node_traces = []
    for ci in range(n_comm):
        comm_nodes = [nd for nd in G.nodes() if node2comm.get(nd)==ci]
        if not comm_nodes: continue
        nx_ = [pos[nd][0] for nd in comm_nodes]
        ny_ = [pos[nd][1] for nd in comm_nodes]
        nt  = [f"<b>{atlas_labels[nd][:35] if nd<len(atlas_labels) else f'ROI-{nd}'}</b><br>"
               f"Community: {ci+1}<br>"
               f"Degree: {deg.get(nd,0):.3f}<br>"
               f"Betweenness: {bet.get(nd,0):.3f}<br>"
               f"Clustering: {clu.get(nd,0):.3f}"
               for nd in comm_nodes]
        ns  = [18+55*deg.get(nd,0) for nd in comm_nodes]
        node_traces.append(go.Scatter(
            x=nx_, y=ny_, mode="markers+text",
            name=f"Community {ci+1}",
            hovertext=nt, hoverinfo="text",
            text=[atlas_labels[nd][:10] if nd<len(atlas_labels)
                  else f"R{nd}" for nd in comm_nodes],
            textfont=dict(size=6, color="white"),
            marker=dict(size=ns, color=palette[ci],
                        line=dict(width=0.5, color="white"))))

    fig = go.Figure(data=[edge_trace]+node_traces,
                    layout=go.Layout(
                        title="Interactive Brain Network – Community Structure",
                        showlegend=True, hovermode="closest",
                        margin=dict(b=20,l=5,r=5,t=50),
                        xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
                        yaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
                        paper_bgcolor="#111", plot_bgcolor="#111",
                        font=dict(color="white"),
                        legend=dict(bgcolor="#222", bordercolor="#555",
                                    font=dict(color="white"))))
    fig.write_html(str(OUT_DIR/"interactive_community_network.html"))
    print("  Saved interactive_community_network.html")


# ═════════════════════════════════════════════════════════════════════════════
#  SAVE ADDITIONAL DATA  (fixed: always uses labels_clean = labels[:n])
# ═════════════════════════════════════════════════════════════════════════════

def save_additional_data(fc_matrix, sc_matrix, alff_img, atlas_img, atlas_labels, out_dir):
    if not PANDAS_AVAILABLE: return
    print("\n  Saving additional CSV / NPY data …")

    n = fc_matrix.shape[0]
    # ── CRITICAL FIX: always align label length to matrix size ──────────────
    labels_clean = atlas_labels[:n]

    # FC matrix
    df_fc = pd.DataFrame(fc_matrix, columns=labels_clean, index=labels_clean)
    df_fc.to_csv(out_dir / "fc_matrix.csv", encoding="utf-8")
    print("    Saved fc_matrix.csv")

    # SC matrix
    n_sc = sc_matrix.shape[0]
    sc_labels = atlas_labels[:n_sc]
    df_sc = pd.DataFrame(sc_matrix, columns=sc_labels, index=sc_labels)
    df_sc.to_csv(out_dir / "sc_matrix.csv", encoding="utf-8")
    print("    Saved sc_matrix.csv")

    # ALFF by ROI
    alff_r    = resample_to_img(alff_img, atlas_img, interpolation="continuous")
    alff_data = alff_r.get_fdata()
    atl_data  = atlas_img.get_fdata().astype(int)
    rows = []
    for roi_id in np.unique(atl_data):
        if roi_id == 0: continue
        mean_alff = float(np.mean(alff_data[atl_data==roi_id]))
        roi_name  = atlas_labels[roi_id-1] if roi_id-1 < len(atlas_labels) \
                    else f"ROI_{roi_id}"
        rows.append({"ROI_ID":roi_id,"ROI_Name":roi_name,"Mean_ALFF":mean_alff})
    pd.DataFrame(rows).to_csv(out_dir / "alff_by_roi.csv", index=False, encoding="utf-8")
    print("    Saved alff_by_roi.csv")


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════════════════

def run_pipeline():
    t_start = time.time()

    # 1. Load data
    data = load_all_modalities()
    if data["bold"] is None:
        print("[FATAL] BOLD image not found. Check PATHS['bold']. Exiting.")
        return

    # 2. Preprocess BOLD
    bold_mni = preprocess_bold(data["bold"], data.get("t1"))

    # 3. Activation metrics
    print("\n── MODULE 3: ACTIVATION METRICS ─────────────────────────────────")
    alff_img, falff_img = compute_alff_falff(bold_mni, TR)
    bold_4mm = image.resample_img(bold_mni,
                                  target_affine=np.diag([4,4,4,1]),
                                  interpolation="linear")
    reho_img = compute_reho(bold_4mm)

    # 4. Atlas & FC
    print("\n── MODULE 4: FUNCTIONAL CONNECTIVITY ────────────────────────────")
    atlas_img, atlas_labels_raw = load_atlas()
    seed_fc = seed_based_connectivity(bold_mni, SEEDS)
    fc_matrix, _, atlas_labels = whole_brain_fc_matrix(bold_mni, atlas_img, atlas_labels_raw)
    # atlas_labels is now GUARANTEED to have len == fc_matrix.shape[0]

    print_top_active_regions(alff_img, atlas_img, atlas_labels)
    print_top_connections(fc_matrix, atlas_labels)

    # 5. ICA / RSN
    components_img, _ = run_ica(bold_mni, n_components=20)
    matched = match_network_components(components_img, bold_mni)

    # 6. Structural connectivity
    dti_result, fa_img = None, None
    if data["dti"] is not None and DIPY_AVAILABLE:
        dti_result = run_dti_tractography(data["dti"], PATHS["dti"])
        if dti_result is not None:
            streamlines, fa_img, _ = dti_result

    n_roi = fc_matrix.shape[0]
    if dti_result is not None:
        sc_matrix = build_sc_matrix(streamlines, atlas_img, atlas_labels, data["dti"].affine)
        if sc_matrix.shape[0] != n_roi:
            sc_pad = np.zeros((n_roi, n_roi))
            m = min(sc_matrix.shape[0], n_roi)
            sc_pad[:m,:m] = sc_matrix[:m,:m]
            sc_matrix = sc_pad
    else:
        print("\n  [INFO] SC matrix = zeros (DTI pipeline skipped).")
        sc_matrix = np.zeros((n_roi, n_roi))

    # 7. Multimodal integration
    integration = integrate_fc_sc(fc_matrix, sc_matrix, atlas_labels)

    # 8. Graph analysis
    G       = build_brain_graph(fc_matrix, atlas_labels)
    metrics = compute_graph_metrics(G, atlas_labels)

    # 9. Original visualisations
    plot_activation_maps(alff_img, falff_img, reho_img, bold_mni)
    plot_glass_brains(seed_fc, bold_mni)
    plot_fc_heatmap(fc_matrix, atlas_labels,
                    "Functional Connectivity Matrix (Harvard-Oxford)", "fc_heatmap.png")
    plot_multimodal_comparison(fc_matrix, sc_matrix, atlas_labels, integration["n"])
    plot_network_graph(G, metrics, atlas_labels)
    plot_hub_bar(metrics, atlas_labels)
    plot_ica_networks(components_img, matched, bold_mni)
    plot_fa_map(fa_img)
    plot_interactive_network(G, metrics, atlas_labels)
    plot_network_summary_dashboard(fc_matrix, sc_matrix, metrics, alff_img, atlas_labels)

    # 10. Extended network visualisations
    print("\n── MODULE 10: EXTENDED NETWORK VISUALISATIONS ───────────────────")
    node2comm, comms, modularity = detect_communities(G)
    plot_community_graph(G, metrics, atlas_labels, node2comm)
    plot_community_fc_heatmap(fc_matrix, atlas_labels, node2comm)
    plot_circular_connectivity(fc_matrix, atlas_labels, top_k=35)
    plot_centrality_radar(metrics, atlas_labels, top_n=8)
    plot_fc_sc_scatter(fc_matrix, sc_matrix, atlas_labels)
    plot_degree_distribution(G, atlas_labels)
    plot_rich_club(G)
    plot_network_analysis_dashboard(G, metrics, fc_matrix, atlas_labels, node2comm)
    plot_interactive_community_network(G, metrics, atlas_labels, node2comm)

    # Save CSVs
    save_additional_data(fc_matrix, sc_matrix, alff_img, atlas_img, atlas_labels, OUT_DIR)

    # Close log
    sys.stdout.file.close()
    sys.stdout = sys.stdout.stdout

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  PIPELINE COMPLETE  ({elapsed/60:.1f} min)")
    print(f"  Outputs saved to: {OUT_DIR.resolve()}")
    print(f"{'='*70}")
    print("\n  Output files:")
    for f in sorted(OUT_DIR.glob("*")):
        print(f"    {f.name:<52s}  {f.stat().st_size/1024:8.1f} KB")


if __name__ == "__main__":
    run_pipeline()
