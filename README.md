# 🧠 多模态脑磁共振影像全流程处理工具
# Multimodal Brain MRI Full-Processing Pipeline



---

## 📑 目录 | Table of Contents
1. [项目简介 | Project Overview](#项目简介--project-overview)
2. [环境搭建 | Environment Setup](#环境搭建--environment-setup)
3. [数据准备与规范 | Data Preparation & Specification](#数据准备与规范--data-preparation--specification)
4. [DICOM转NIfTI工具使用指南 | DICOM to NIfTI Tool Guide](#dicom转nifti工具使用指南--dicom-to-nifti-tool-guide)
5. [多模态脑连接分析流水线使用指南 | Multimodal Brain Connectivity Pipeline Guide](#多模态脑连接分析流水线使用指南--multimodal-brain-connectivity-pipeline-guide)
6. [脑区激活分析工具使用指南 | Brain Region Activation Analysis Guide](#脑区激活分析工具使用指南--brain-region-activation-analysis-guide)
7. [3D脑网络可视化工具使用指南 | 3D Brain Network Visualization Guide](#3d脑网络可视化工具使用指南--3d-brain-network-visualization-guide)
8. [FreeSurfer模型导出与合并工具 | FreeSurfer Model Export & Merge Tools](#freerfer模型导出与合并工具--freerfer-model-export--merge-tools)
9. [扫描参数与质控规范 | Scan Parameters & Quality Control Standards](#扫描参数与质控规范--scan-parameters--quality-control-standards)
10. [常见问题 | FAQ](#常见问题--faq)
11. [输出图片结果 | Figure outputs](#figure_outputs)
12. [参考文献 | References](#参考文献--references)

---

## 📖 项目简介 | Project Overview

### 🇨🇳 中文
本项目严格遵循《T/CHIA 48-2024 精神影像脑结构功能成像技术与信息处理规范》，实现了脑磁共振影像从**原始DICOM数据格式转换**到**多模态定量分析、脑连接组构建、图论分析与可视化**的全流程自动化处理。

**支持的模态包括：** T1WI结构像、BOLD静息态功能像、DTI弥散张量像、FLAIR、SWI、QSM、ASL、TOF-MRA等，适配1.5T/3.0T临床与科研型磁共振设备，可满足精神影像临床科研的标准化分析需求。

**核心功能模块：**
- 🔄 DICOM → NIfTI 批量转换
- 🧠 多模态脑连接分析（功能+结构）
- 📊 zALFF脑区激活定量分析
- 🎨 3D脑网络可视化（AAL166图谱）
- 📐 FreeSurfer彩色模型导出与合并

### 🇬🇧 English
This project strictly follows the *T/CHIA 48-2024 Specification for structural and functional imaging technology and information processing in psychoradiology*, and implements a fully automated full workflow for brain MRI processing, from **raw DICOM data format conversion** to **multimodal quantitative analysis, brain connectome construction, graph theory analysis and visualization**.

**Supported modalities include:** T1WI structural image, BOLD resting-state functional image, DTI diffusion tensor image, FLAIR, SWI, QSM, ASL, TOF-MRA, etc. It is compatible with 1.5T/3.0T clinical and research MRI scanners, and can meet the standardized analysis requirements of clinical research on psychiatric imaging.

**Core Functional Modules:**
- 🔄 DICOM → NIfTI Batch Conversion
- 🧠 Multimodal Brain Connectivity Analysis (FC + SC)
- 📊 zALFF Brain Region Activation Quantitative Analysis
- 🎨 3D Brain Network Visualization (AAL166 Atlas)
- 📐 FreeSurfer Colored Model Export & Merge

---

## 🛠️ 环境搭建 | Environment Setup

### 🇨🇳 中文
本项目运行依赖**系统级工具**与**Python第三方库**，以下为分系统的分步安装指南。

#### 1. 基础环境要求
| 项目 | 要求 |
|------|------|
| 操作系统 | Windows 10/11、macOS 10.15+、Linux (Ubuntu 18.04+/CentOS 7+) |
| Python版本 | Python 3.8 ~ 3.11（推荐3.10，兼容性最佳） |
| 内存 | 最低16GB，推荐32GB及以上（全脑高分辨率分析需64GB+） |
| 存储 | 最低10GB可用空间，建议SSD以提升数据读写速度 |

#### 2. 系统级工具安装（dcm2niix）
`dcm2niix` 是DICOM转NIfTI格式的核心工具，必须提前安装：

##### macOS
```bash
# 安装Homebrew（如未安装）
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
# 安装dcm2niix
brew install dcm2niix
# 验证安装
dcm2niix --version
```

##### Windows
```bash
# 方式一：通过conda安装（推荐）
conda install -c conda-forge dcm2niix
# 验证安装
dcm2niix --version

# 方式二：下载预编译二进制文件
# 前往 https://github.com/rordenlab/dcm2niix/releases 下载Windows版本
# 解压后将exe所在路径添加到系统环境变量PATH中
```

##### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install dcm2niix
dcm2niix --version
```

#### 3. Python虚拟环境配置（推荐）
```bash
# 方式一：conda（推荐）
conda create -n mri_pipeline python=3.10
conda activate mri_pipeline

# 方式二：venv
python -m venv mri_pipeline
# Windows
mri_pipeline\Scripts\activate
# macOS/Linux
source mri_pipeline/bin/activate
```

#### 4. Python依赖库安装
```bash
# 必选核心依赖
pip install numpy scipy nibabel nilearn dipy matplotlib seaborn networkx pandas
# 可选依赖（交互式可视化功能）
pip install plotly
```

| 依赖库 | 核心用途 |
|--------|----------|
| nibabel | NIfTI格式数据的读写与处理 |
| nilearn | fMRI数据预处理、脑图谱加载、功能连接分析 |
| dipy | DTI弥散数据建模、纤维束追踪、结构连接分析 |
| numpy/scipy | 数值计算、信号处理与统计分析 |
| matplotlib/seaborn | 静态可视化绘图 |
| networkx | 脑网络图论分析 |
| pandas | 分析结果表格化输出 |
| plotly | 交互式网络可视化 |

### 🇬🇧 English
The operation of this project depends on **system-level tools** and **Python third-party libraries**.

#### 1. Basic Environment Requirements
| Item | Requirements |
|------|------|
| Operating System | Windows 10/11, macOS 10.15+, Linux (Ubuntu 18.04+/CentOS 7+) |
| Python Version | Python 3.8 ~ 3.11 (3.10 recommended for best compatibility) |
| Memory | Minimum 16GB, 32GB+ recommended (64GB+ for whole-brain high-resolution analysis) |
| Storage | Minimum 10GB free space, SSD recommended |

#### 2. System-level Tool Installation (dcm2niix)
`dcm2niix` is the core tool for DICOM to NIfTI format conversion.

##### macOS
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install dcm2niix
dcm2niix --version
```

##### Windows
```bash
conda install -c conda-forge dcm2niix
dcm2niix --version
```

##### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install dcm2niix
dcm2niix --version
```

#### 3. Python Virtual Environment Configuration
```bash
# Method 1: conda (Recommended)
conda create -n mri_pipeline python=3.10
conda activate mri_pipeline

# Method 2: venv
python -m venv mri_pipeline
# Windows
mri_pipeline\Scripts\activate
# macOS/Linux
source mri_pipeline/bin/activate
```

#### 4. Python Dependency Installation
```bash
# Mandatory core dependencies
pip install numpy scipy nibabel nilearn dipy matplotlib seaborn networkx pandas
# Optional dependency (for interactive visualization)
pip install plotly
```

| Library | Core Purpose |
|--------|----------|
| nibabel | Read, write and process NIfTI format data |
| nilearn | fMRI preprocessing, brain atlas loading, functional connectivity analysis |
| dipy | DTI diffusion modeling, fiber tractography, structural connectivity analysis |
| numpy/scipy | Numerical calculation, signal processing and statistical analysis |
| matplotlib/seaborn | Static visualization plotting |
| networkx | Brain network graph theory analysis |
| pandas | Tabular output of analysis results |
| plotly | Interactive network visualization |

---

## 📁 数据准备与规范 | Data Preparation & Specification

### 🇨🇳 中文
#### 1. 原始DICOM数据目录结构
原始DICOM数据需按**序列分文件夹存放**，每个序列一个独立文件夹：
```
├── sort/  # 原始DICOM根目录
│   ├── 001_T1_MPRAGE/      # T1结构像序列
│   ├── 002_BOLD_REST/      # BOLD静息态功能像序列
│   ├── 003_DTI_64DIR/      # DTI弥散张量像序列
│   ├── 004_FLAIR_3D/       # 3D FLAIR序列
│   ├── 005_SWI/            # SWI磁敏感加权序列
│   └── 006_TOF_MRA/        # TOF血管成像序列
```

#### 2. 模态命名关键词规则
| 模态 | 识别关键词 |
|------|------------|
| T1WI | T1、TFE |
| BOLD | BOLD、MB2 |
| DTI | DTI |
| DSI | DSI |
| FLAIR | FLAIR |
| SWI | SWI、SWIp |
| QSM | QSM |

#### 3. DTI数据特殊要求
DTI数据需配套`.bval`和`.bvec`梯度文件，需与DICOM文件放在同一文件夹下。

#### 4. 配套数据文件
本项目需配合以下标准模板和图谱文件使用（已上传至仓库）：
| 文件名 | 用途 |
|--------|------|
| AAL3v1.nii.gz | AAL3脑区图谱（166区域） |
| AAL3v1.nii.txt | AAL3图谱标签文件 |
| mni_icbm152_t1_tal_nlin_sym_09a.nii | MNI152标准模板 |
| mni_icbm152_t1_tal_nlin_sym_09a_mask.nii.gz | MNI152脑掩膜 |

### 🇬🇧 English
#### 1. Raw DICOM Data Directory Structure
```
├── sort/  # Root directory of raw DICOM
│   ├── 001_T1_MPRAGE/      # T1 structural image sequence
│   ├── 002_BOLD_REST/      # BOLD resting-state functional image sequence
│   ├── 003_DTI_64DIR/      # DTI diffusion tensor image sequence
│   ├── 004_FLAIR_3D/       # 3D FLAIR sequence
│   ├── 005_SWI/            # SWI susceptibility weighted sequence
│   └── 006_TOF_MRA/        # TOF angiography sequence
```

#### 2. Modality Naming Keyword Rules
| Modality | Recognition Keywords |
|------|------------|
| T1WI | T1、TFE |
| BOLD | BOLD、MB2 |
| DTI | DTI |
| DSI | DSI |
| FLAIR | FLAIR |
| SWI | SWI、SWIp |
| QSM | QSM |

#### 3. Special Requirements for DTI Data
DTI data must be accompanied by `.bval` and `.bvec` gradient files.

#### 4. Supporting Data Files
The following standard templates and atlas files are required (uploaded to repository):
| File Name | Purpose |
|--------|------|
| AAL3v1.nii.gz | AAL3 Brain Atlas (166 regions) |
| AAL3v1.nii.txt | AAL3 Atlas Label File |
| mni_icbm152_t1_tal_nlin_sym_09a.nii | MNI152 Standard Template |
| mni_icbm152_t1_tal_nlin_sym_09a_mask.nii.gz | MNI152 Brain Mask |

---

## 🔄 DICOM转NIfTI工具使用指南 | DICOM to NIfTI Tool Guide

### 🇨🇳 中文
**脚本文件：** `DICOM2NIfTI.py`

本工具用于批量将原始DICOM序列转换为NIfTI格式（`.nii.gz`压缩格式），并自动按模态分类存放。

#### 1. 脚本配置修改
```python
# ======================
# 路径配置 | Path Configuration
# ======================
# 原始DICOM数据根目录
BASE_DIR = "/Users/xxx/Downloads/brain_data/sort"
# 转换后NIfTI文件输出根目录
OUTPUT_DIR = "/Users/xxx/Downloads/brain_data/nifti"
```

#### 2. 运行脚本
```bash
python DICOM2NIfTI.py
```

#### 3. 输出结果说明
```
├── nifti/  # 输出根目录
│   ├── T1/    # T1结构像转换结果
│   ├── BOLD/  # BOLD功能像转换结果
│   ├── DTI/   # DTI弥散像转换结果（含.bval/.bvec）
│   ├── FLAIR/ # FLAIR序列转换结果
│   ├── SWI/   # SWI序列转换结果
│   └── OTHER/ # 未识别模态的转换结果
```

#### 4. 核心功能
- ✅ 自动递归扫描原始目录下的所有序列文件夹
- ✅ 基于关键词自动识别模态并分类存放
- ✅ 自动生成规范的文件名，避免重名
- ✅ 输出`.nii.gz`压缩格式，节省存储空间
- ✅ 终端实时打印转换进度与成功/失败状态

### 🇬🇧 English
**Script File:** `DICOM2NIfTI.py`

This tool is used to batch convert raw DICOM sequences to NIfTI format (`.nii.gz` compressed format).

#### 1. Script Configuration
```python
# Path Configuration
BASE_DIR = "/Users/xxx/Downloads/brain_data/sort"
OUTPUT_DIR = "/Users/xxx/Downloads/brain_data/nifti"
```

#### 2. Run the Script
```bash
python DICOM2NIfTI.py
```

#### 3. Output Structure
```
├── nifti/
│   ├── T1/    # T1 structural images
│   ├── BOLD/  # BOLD functional images
│   ├── DTI/   # DTI diffusion images (with .bval/.bvec)
│   ├── FLAIR/ # FLAIR sequences
│   ├── SWI/   # SWI sequences
│   └── OTHER/ # Unrecognized modalities
```

#### 4. Core Features
- ✅ Auto recursive scan of all sequence folders
- ✅ Auto modality recognition and classification
- ✅ Standardized file naming to avoid duplication
- ✅ `.nii.gz` compressed output format
- ✅ Real-time progress and status printing

---

## 🧠 多模态脑连接分析流水线使用指南 | Multimodal Brain Connectivity Pipeline Guide

### 🇨🇳 中文
**脚本文件：** `multimodal_brain_connectivity_pipeline.py`

本流水线是核心分析工具，实现了多模态脑MRI数据的全流程标准化分析，严格遵循T/CHIA 48-2024规范。

#### 1. 核心功能模块
| 模块 | 功能 |
|------|------|
| 📥 数据发现与校验 | 自动加载各模态NIfTI数据，校验数据维度与完整性 |
| 🧹 rs-fMRI预处理 | 丢弃dummy扫描、空间平滑、带通滤波、MNI152标准化 |
| 📊 脑激活指标计算 | ALFF/fALFF（低频振幅）、ReHo（区域同质性） |
| 🔗 功能连接分析 | 种子点功能连接、全脑功能连接矩阵、脑网络识别 |
| 🧬 结构连接分析 | 张量模型拟合、FA图计算、纤维束追踪、SC矩阵构建 |
| 🔄 多模态整合 | FC与SC的一致性与差异性分析 |
| 🕸️ 图论网络分析 | 度中心性/介数中心性/聚类系数、核心脑区识别 |
| 🎨 可视化 | 20+种可视化图表（激活图、热图、网络图等） |

#### 2. 脚本配置修改
```python
# ======================
# 路径配置 | Path Configuration
# ======================
PATHS = {
    "t1":   "/Users/xxx/Downloads/brain_data/nifti/T1",
    "bold": "/Users/xxx/Downloads/brain_data/nifti/BOLD",
    "dti":  "/Users/xxx/Downloads/brain_data/nifti/DTI",
    "atlas_nii":    "/path/to/AAL3v1.nii.gz",
    "atlas_labels": "/path/to/AAL3v1.nii.txt"
}
# 分析结果输出目录
OUT_DIR = Path("./brain_pipeline_outputs")

# ======================
# 预处理参数 | Preprocessing Parameters
# ======================
TR        = 1.0     # BOLD序列重复时间(ms)
FWHM      = 6.0     # 空间平滑核大小(mm)
HP_FREQ   = 0.01    # 带通滤波低频截止(Hz)
LP_FREQ   = 0.10    # 带通滤波高频截止(Hz)
N_DUMMIES = 5       # 丢弃的初始dummy扫描帧数
FA_THRESH = 0.20    # DTI纤维束追踪FA阈值
```

#### 3. 运行流水线
```bash
python multimodal_brain_connectivity_pipeline.py
```

#### 4. 输出结果
| 文件类型 | 核心文件 | 说明 |
|----------|----------|------|
| 预处理NIfTI | `bold_preprocessed_mni.nii.gz`、`alff.nii.gz`、`fa_map.nii.gz` | 预处理后的功能像、激活指标图、FA图 |
| 连接矩阵 | `fc_matrix.npy/csv`、`sc_matrix.npy/csv` | 功能/结构连接矩阵 |
| 定量表格 | `pipeline_results_summary.csv`、`alff_by_roi.csv` | Hub排名、最强连接、脑区ALFF均值 |
| 静态图表 | `activation_maps.png`、`fc_heatmap.png`、`brain_network_graph.png` | 20+种标准化可视化图表 |
| 交互图表 | `interactive_network.html` | 可交互脑网络图，支持放大、悬停查看 |
| 日志文件 | `pipeline_terminal_output.txt` | 全流程终端输出日志 |

### 🇬🇧 English
**Script File:** `multimodal_brain_connectivity_pipeline.py`

This pipeline is the core analysis tool for multimodal brain MRI data.

#### 1. Core Functional Modules
| Module | Function |
|------|------|
| 📥 Data Discovery & Validation | Auto load NIfTI data, verify dimension and integrity |
| 🧹 rs-fMRI Preprocessing | Drop dummy scans, smoothing, filtering, MNI152 normalization |
| 📊 Activation Metrics | ALFF/fALFF, ReHo calculation |
| 🔗 Functional Connectivity | Seed-based FC, whole-brain FC matrix, network identification |
| 🧬 Structural Connectivity | Tensor fitting, FA map, tractography, SC matrix |
| 🔄 Multimodal Integration | FC-SC consistency and difference analysis |
| 🕸️ Graph Theory Analysis | Degree/betweenness centrality, clustering coefficient, hub identification |
| 🎨 Visualization | 20+ visualization charts |

#### 2. Script Configuration
```python
# Path Configuration
PATHS = {
    "t1":   "/Users/xxx/Downloads/brain_data/nifti/T1",
    "bold": "/Users/xxx/Downloads/brain_data/nifti/BOLD",
    "dti":  "/Users/xxx/Downloads/brain_data/nifti/DTI",
    "atlas_nii":    "/path/to/AAL3v1.nii.gz",
    "atlas_labels": "/path/to/AAL3v1.nii.txt"
}
OUT_DIR = Path("./brain_pipeline_outputs")

# Preprocessing Parameters
TR        = 1.0     # BOLD TR (ms)
FWHM      = 6.0     # Smoothing kernel (mm)
HP_FREQ   = 0.01    # Bandpass low cutoff (Hz)
LP_FREQ   = 0.10    # Bandpass high cutoff (Hz)
N_DUMMIES = 5       # Dummy scans to drop
FA_THRESH = 0.20    # DTI FA threshold
```

#### 3. Run the Pipeline
```bash
python multimodal_brain_connectivity_pipeline.py
```

#### 4. Output Files
| File Type | Core Files | Description |
|----------|----------|------|
| Preprocessed NIfTI | `bold_preprocessed_mni.nii.gz`, `alff.nii.gz`, `fa_map.nii.gz` | Preprocessed images |
| Connectivity Matrix | `fc_matrix.npy/csv`, `sc_matrix.npy/csv` | FC/SC matrices |
| Quantitative Tables | `pipeline_results_summary.csv`, `alff_by_roi.csv` | Hub ranking, strongest connections |
| Static Charts | `activation_maps.png`, `fc_heatmap.png`, `brain_network_graph.png` | 20+ visualization charts |
| Interactive Charts | `interactive_network.html` | Interactive brain network |
| Log File | `pipeline_terminal_output.txt` | Full process log |

---

## 📊 脑区激活分析工具使用指南 | Brain Region Activation Analysis Guide
### 🇨🇳 中文
**脚本文件：** `MNI152_zALFF_Brain_Region_Activation_Analysis.py`

本工具专注于静息态zALFF脑区激活的定量分析，输出最活跃脑区的排名与MNI坐标。

#### 1. 核心功能
| 功能 | 说明 |
|------|------|
| 🎯 MNI152空间对齐 | 标准化到MNI空间进行分析 |
| 📈 TOP 10活跃脑区 | 输出最活跃的10个区域及统计值 |
| 📍 MNI坐标提取 | 提供精确的MNI坐标信息 |
| 📋 全脑区统计 | 输出所有脑区的激活统计表格 |

#### 2. 运行脚本
```bash
python MNI152_zALFF_Brain_Region_Activation_Analysis.py
```

#### 3. 输出结果
| 文件 | 说明 |
|------|------|
| `TOP10_active_regions.csv` | TOP 10活跃脑区排名及统计值 |
| `top10_coordinates.csv` | TOP 10脑区的MNI坐标 |
| `all_brain_regions_activity.csv` | 所有脑区的激活统计表格 |

### 🇬🇧 English
**Script File:** `MNI152_zALFF_Brain_Region_Activation_Analysis.py`

This tool focuses on quantitative analysis of resting-state zALFF brain region activation.

#### 1. Core Features
| Feature | Description |
|------|------|
| 🎯 MNI152 Alignment | Standardized to MNI space |
| 📈 TOP 10 Active Regions | Output top 10 most active regions |
| 📍 MNI Coordinates | Precise MNI coordinate information |
| 📋 Whole Brain Statistics | Activation statistics for all regions |

#### 2. Run the Script
```bash
python MNI152_zALFF_Brain_Region_Activation_Analysis.py
```

#### 3. Output Files
| File | Description |
|------|------|
| `TOP10_active_regions.csv` | TOP 10 active regions ranking |
| `top10_coordinates.csv` | MNI coordinates of TOP 10 regions |
| `all_brain_regions_activity.csv` | Activation statistics for all regions |

---

## 🎨 3D脑网络可视化工具使用指南 | 3D Brain Network Visualization Guide

### 🇨🇳 中文
**脚本文件：** `plot_3D_brain_network_AAL166.py`

本工具基于AAL166图谱生成3D脑网络可视化图，展示脑区间的功能连接通路。

#### 1. 核心功能
| 功能 | 说明 |
|------|------|
| 🧠 AAL166图谱支持 | 166个脑区的完整覆盖 |
| 🔗 最强连接通路 | 显示TOP 8最强功能连接 |
| 🎨 彩色节点分类 | 按功能网络分类着色 |
| 📐 3D空间定位 | 基于MNI坐标的3D空间展示 |

#### 2. 运行脚本
```bash
python plot_3D_brain_network_AAL166.py
```

#### 3. 输出结果
| 文件 | 说明 |
|------|------|
| `3D_brain_network_AAL166.png` | 3D脑网络可视化图 |
| `network_edges.csv` | 显示的连接边数据 |

### 🇬🇧 English
**Script File:** `plot_3D_brain_network_AAL166.py`

This tool generates 3D brain network visualization based on AAL166 atlas.

#### 1. Core Features
| Feature | Description |
|------|------|
| 🧠 AAL166 Atlas | Full coverage of 166 brain regions |
| 🔗 Strongest Connections | Display TOP 8 strongest functional connections |
| 🎨 Colored Node Classification | Color-coded by functional network |
| 📐 3D Spatial Positioning | 3D display based on MNI coordinates |

#### 2. Run the Script
```bash
python plot_3D_brain_network_AAL166.py
```

#### 3. Output Files
| File | Description |
|------|------|
| `3D_brain_network_AAL166.png` | 3D brain network visualization |
| `network_edges.csv` | Displayed connection edge data |

---

## 📐 FreeSurfer模型导出与合并工具 | FreeSurfer Model Export & Merge Tools

本项目地址：https://github.com/Karcen/freesurfer-recon-freeview-3dstats

### 🇨🇳 中文
本模块包含两个脚本，用于从FreeSurfer处理结果导出彩色3D脑模型并合并为完整大脑。

#### 1. export_colored_ply_from_freesurfer.py
**功能：** 从FreeSurfer导出彩色PLY格式脑模型

| 功能 | 说明 |
|------|------|
| 🎨 解剖颜色映射 | 保留脑区原生颜色 |
| 📐 标准PLY格式 | 兼容Blender/MeshLab等3D软件 |
| 🔄 左右脑分离 | 分别导出左右半球模型 |

```bash
python export_colored_ply_from_freesurfer.py
```

#### 2. merge_colored_brain_ply.py
**功能：** 合并左右脑PLY模型为完整大脑

| 功能 | 说明 |
|------|------|
| 🔀 左右脑合并 | 合并为完整大脑模型 |
| 📋 索引偏移处理 | 自动处理面索引避免冲突 |
| ✅ ASCII格式输出 | 最佳兼容性 |

```bash
python merge_colored_brain_ply.py
```

#### 3. 输出结果
| 文件 | 说明 |
|------|------|
| `left_hemisphere.ply` | 左半球彩色模型 |
| `right_hemisphere.ply` | 右半球彩色模型 |
| `merged_brain.ply` | 合并后的完整大脑模型 |

### 🇬🇧 English
This module contains two scripts for exporting colored 3D brain models from FreeSurfer and merging them into a complete brain.

The project can be found at: https://github.com/Karcen/freesurfer-recon-freeview-3dstats

#### 1. export_colored_ply_from_freesurfer.py
**Function:** Export colored PLY format brain models from FreeSurfer

| Feature | Description |
|------|------|
| 🎨 Anatomical Color Mapping | Preserve native brain region colors |
| 📐 Standard PLY Format | Compatible with Blender/MeshLab |
| 🔄 Hemisphere Separation | Export left and right hemispheres separately |

```bash
python export_colored_ply_from_freesurfer.py
```

#### 2. merge_colored_brain_ply.py
**Function:** Merge left and right hemisphere PLY models

| Feature | Description |
|------|------|
| 🔀 Hemisphere Merge | Combine into complete brain model |
| 📋 Index Offset Handling | Auto handle face index to avoid conflicts |
| ✅ ASCII Format Output | Best compatibility |

```bash
python merge_colored_brain_ply.py
```

#### 3. Output Files
| File | Description |
|------|------|
| `left_hemisphere.ply` | Left hemisphere colored model |
| `right_hemisphere.ply` | Right hemisphere colored model |
| `merged_brain.ply` | Merged complete brain model |

---

## 📋 扫描参数与质控规范 | Scan Parameters & Quality Control Standards {#扫描参数与质控规范--scan-parameters--quality-control-standards}

### 🇨🇳 中文
本章节内容严格遵循《T/CHIA 48-2024 精神影像脑结构功能成像技术与信息处理规范》。

#### 1. 设备基础参数要求
| 磁场强度 | 应用场景 | 梯度性能 | 独立射频通道数 | 线圈要求 |
|----------|----------|----------|----------------|----------|
| 1.5T | 科研 | 33mT/m，125mT/m/s | 24 | 头颈联合线圈≥16通道 |
| 1.5T | 临床 | 30mT/m，100mT/m/s | 16 | 头颈联合线圈≥10通道 |
| 3.0T | 科研 | 44mT/m，200mT/m/s | 48/64/128 | 专用头部线圈≥32通道 |
| 3.0T | 临床 | 36mT/m，150mT/m/s | 32 | 头颈联合线圈头部≥20通道 |

#### 2. 核心序列推荐参数（3.0T科研场景）
| 序列 | 核心参数 |
|------|----------|
| 3D-T1WI结构像 | TR=2300ms，TE=2.3ms，TI=900ms，FA=8°，体素0.8~1mm³ |
| BOLD功能像 | TR=2000~3000ms，TE=30ms，FA=90°，体素1.2~2mm³，时间点500~600 |
| DTI弥散像 | TR=7000ms，TE=89ms，体素1.5~2mm³，b=1000/2000/3000 s/mm²，梯度≥64方向 |
| 3D-FLAIR | TR=8000ms，TE=150ms，TI=2000ms，体素0.8~1mm³ |
| 3D-ASL | TR=3500~6500ms，TI=1500/2000/2500/3000ms，层厚3mm，采集7min |
| SWI | TR=20~30ms，TE=20ms，层厚1.0~1.5mm，面内分辨率0.5~0.7mm |

#### 3. 核心序列质控要求
| 序列 | 质控指标 | 合格要求 |
|------|----------|----------|
| 3D-T1WI | SNR | ≥65db |
| | CNR | ≥60db |
| | 图像均匀性 | ≥80% |
| BOLD | SNR | ≥65db |
| | 几何畸变率 | ≤5% |
| | 头动控制 | 平动位移≤1/2体素 |
| DTI | SNR | ≥65db |
| | 几何畸变率 | ≤5% |
| | FA图像 | 白质FA显著高于灰质 |

### 🇬🇧 English
This section strictly follows the *T/CHIA 48-2024 Specification for Psychoradiology*.

#### 1. Basic Equipment Parameter Requirements
| Field Strength | Application | Gradient Performance | RF Channels | Coil Requirements |
|----------|----------|----------|----------------|----------|
| 1.5T | Research | 33mT/m, 125mT/m/s | 24 | Head-neck coil ≥16 channels |
| 1.5T | Clinical | 30mT/m, 100mT/m/s | 16 | Head-neck coil ≥10 channels |
| 3.0T | Research | 44mT/m, 200mT/m/s | 48/64/128 | Head coil ≥32 channels |
| 3.0T | Clinical | 36mT/m, 150mT/m/s | 32 | Head-neck coil head ≥20 channels |

#### 2. Recommended Core Sequence Parameters (3.0T Research)
| Sequence | Core Parameters |
|------|----------|
| 3D-T1WI | TR=2300ms, TE=2.3ms, TI=900ms, FA=8°, Voxel 0.8~1mm³ |
| BOLD | TR=2000~3000ms, TE=30ms, FA=90°, Voxel 1.2~2mm³, Time points 500~600 |
| DTI | TR=7000ms, TE=89ms, Voxel 1.5~2mm³, b=1000/2000/3000 s/mm², ≥64 directions |
| 3D-FLAIR | TR=8000ms, TE=150ms, TI=2000ms, Voxel 0.8~1mm³ |
| 3D-ASL | TR=3500~6500ms, TI=1500/2000/2500/3000ms, Slice 3mm, 7min |
| SWI | TR=20~30ms, TE=20ms, Slice 1.0~1.5mm, In-plane 0.5~0.7mm |

#### 3. Core Sequence QC Requirements
| Sequence | QC Index | Requirement |
|------|----------|----------|
| 3D-T1WI | SNR | ≥65db |
| | CNR | ≥60db |
| | Uniformity | ≥80% |
| BOLD | SNR | ≥65db |
| | Distortion | ≤5% |
| | Head Motion | ≤1/2 voxel |
| DTI | SNR | ≥65db |
| | Distortion | ≤5% |
| | FA Image | WM FA > GM FA |

---

## ❓ 常见问题 | FAQ 

### 🇨🇳 中文
#### Q1: 运行DICOM2NIfTI.py时提示"dcm2niix: command not found"
**A:** dcm2niix未正确安装或未加入系统环境变量。请重新安装dcm2niix，或在脚本中直接指定dcm2niix的绝对路径。

#### Q2: 运行流水线时提示"ImportError: No module named 'xxx'"
**A:** 对应的Python依赖库未安装。请激活虚拟环境后，重新执行依赖安装命令。

#### Q3: BOLD数据预处理失败，提示"维度不匹配"
**A:** 检查BOLD数据是否为4维（x, y, z, 时间点）。确认`TR`参数与扫描参数一致，`N_DUMMIES`小于总时间点数。

#### Q4: DTI模块运行失败，提示"找不到bval/bvec文件"
**A:** 确保`.bval`和`.bvec`文件与DTI的NIfTI文件同目录，且文件名前缀完全一致（如`DTI_001.nii.gz`对应`DTI_001.bval`和`DTI_001.bvec`）。

#### Q5: 可视化图片显示不全、文字重叠
**A:** 检查matplotlib版本（推荐3.5+）。可修改脚本中绘图的`figsize`参数调整画布大小。

#### Q6: FreeSurfer模型导出失败
**A:** 确保已完成FreeSurfer的`recon-all`处理，且`SUBJECTS_DIR`环境变量已正确设置。

### 🇬🇧 English
#### Q1: "dcm2niix: command not found" when running DICOM2NIfTI.py
**A:** dcm2niix is not installed correctly or not in system PATH. Reinstall dcm2niix or specify absolute path in script.

#### Q2: "ImportError: No module named 'xxx'" when running pipeline
**A:** Python dependency not installed. Activate virtual environment and reinstall dependencies.

#### Q3: BOLD preprocessing failed, "dimension mismatch"
**A:** Check BOLD data is 4D (x, y, z, time). Verify `TR` matches scan parameters, `N_DUMMIES` < total time points.

#### Q4: DTI module failed, "cannot find bval/bvec files"
**A:** Ensure `.bval` and `.bvec` files are in same directory as DTI NIfTI file with matching filename prefix.

#### Q5: Visualization images incomplete, text overlapping
**A:** Check matplotlib version (3.5+ recommended). Modify `figsize` parameter in script to adjust canvas size.

#### Q6: FreeSurfer model export failed
**A:** Ensure FreeSurfer `recon-all` is completed and `SUBJECTS_DIR` environment variable is set correctly.

---

## 输出图片结果 Figure outputs 
<img width="2628" height="2180" alt="summary_dashboard" src="https://github.com/user-attachments/assets/921bfb4c-e8b9-415d-90f8-b2eea73f78bc" />
<img width="1335" height="729" alt="rich_club" src="https://github.com/user-attachments/assets/f53bce52-1de6-4a54-877c-7355eb8b761a" />
<img width="2862" height="2468" alt="network_analysis_dashboard" src="https://github.com/user-attachments/assets/ed3b581b-a5ad-4b81-bca5-34e78bd58b1b" />
<img width="3245" height="1067" alt="multimodal_comparison" src="https://github.com/user-attachments/assets/979c68a9-b8e2-4d97-a085-1c0a2d15e98b" />
<img width="2327" height="1937" alt="ica_networks_DMN_SAL_CEN" src="https://github.com/user-attachments/assets/e4f39a8e-2d3a-4905-8b60-a1bd7b24a674" />
<img width="1286" height="766" alt="hub_regions_bar" src="https://github.com/user-attachments/assets/80a6cf98-ecb2-44ec-aeaf-e97ca2c90eb4" />
<img width="3588" height="707" alt="glass_brain_seed_fc" src="https://github.com/user-attachments/assets/9a654535-44e8-442d-82bd-6da49eb23d8d" />
<img width="1336" height="1029" alt="fc_sc_scatter" src="https://github.com/user-attachments/assets/986bad1e-e82c-458e-ba24-b1a73713f7d2" />
<img width="2175" height="2082" alt="fc_heatmap" src="https://github.com/user-attachments/assets/8b35b3b5-4b1f-4606-8d7e-6b648b48a9e0" />
<img width="2067" height="507" alt="dti_fa_map" src="https://github.com/user-attachments/assets/3051abf9-bf91-43ae-9436-722290d00453" />
<img width="1934" height="738" alt="degree_distribution" src="https://github.com/user-attachments/assets/56284b8b-9817-473d-9743-75dd99fe3b96" />
<img width="1890" height="1478" alt="community_graph" src="https://github.com/user-attachments/assets/1c305ab2-71fb-4938-afe1-b67cef5e3b6b" />
<img width="2201" height="2083" alt="community_fc_heatmap" src="https://github.com/user-attachments/assets/363b5492-6001-44d4-bc8a-530915c0b036" />
<img width="1914" height="1928" alt="circular_connectivity" src="https://github.com/user-attachments/assets/0de9766b-6364-4edc-8a92-4efb82fceb7e" />
<img width="1580" height="1292" alt="centrality_radar" src="https://github.com/user-attachments/assets/0d3a61d4-2558-4605-97c6-4dff75dda1a1" />
<img width="1992" height="1478" alt="brain_network_graph" src="https://github.com/user-attachments/assets/6f7b16cc-09f1-43a4-a3b5-505349491a81" />
<img width="2685" height="1785" alt="activation_maps" src="https://github.com/user-attachments/assets/9342d4b3-3175-408a-8866-14fe216f3b81" />


## 📚 参考文献 | References

1. T/CHIA 48-2024, 精神影像脑结构功能成像技术与信息处理规范[S]. 中国卫生信息与健康医疗大数据学会, 2024.

📜 本项目采用 MIT开源协议，允许个人、企业自由使用、修改、分发本项目代码，无需支付授权费用，仅需在分发时保留原作者版权声明即可。

📜 This project adopts theMIT Open Source License, which allows individuals and enterprises to freely use, modify, and distribute the project code without paying authorization fees. It is only required to retain the original author's copyright notice when distributing.
