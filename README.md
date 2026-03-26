# 多模态脑磁共振影像全流程处理工具
# Multimodal Brain MRI Full-Processing Pipeline

---

## 目录 | Table of Contents
1. [项目简介 | Project Overview](#项目简介--project-overview)
2. [环境搭建 | Environment Setup](#环境搭建--environment-setup)
3. [数据准备与规范 | Data Preparation & Specification](#数据准备与规范--data-preparation--specification)
4. [DICOM转NIfTI工具使用指南 | DICOM to NIfTI Tool Guide](#dicom转nifti工具使用指南--dicom-to-nifti-tool-guide)
5. [多模态fMRI处理流水线使用指南 | Multimodal fMRI Pipeline Guide](#多模态fmri处理流水线使用指南--multimodal-fmri-pipeline-guide)
6. [扫描参数与质控规范 | Scan Parameters & Quality Control Standards](#扫描参数与质控规范--scan-parameters--quality-control-standards)
7. [常见问题 | FAQ](#常见问题--faq)
8. [参考文献 | References](#参考文献--references)

---

## 项目简介 | Project Overview
### 中文
本项目严格遵循《T/CHIA 48-2024 精神影像脑结构功能成像技术与信息处理规范》，实现了脑磁共振影像从**原始DICOM数据格式转换**到**多模态定量分析、脑连接组构建、图论分析与可视化**的全流程自动化处理。

支持的模态包括：T1WI结构像、BOLD静息态功能像、DTI弥散张量像、FLAIR、SWI、QSM、ASL、TOF-MRA等，适配1.5T/3.0T临床与科研型磁共振设备，可满足精神影像临床科研的标准化分析需求。

### English
This project strictly follows the *T/CHIA 48-2024 Specification for structural and functional imaging technology and information processing in psychoradiology*, and implements a fully automated full workflow for brain MRI processing, from **raw DICOM data format conversion** to **multimodal quantitative analysis, brain connectome construction, graph theory analysis and visualization**.

Supported modalities include: T1WI structural image, BOLD resting-state functional image, DTI diffusion tensor image, FLAIR, SWI, QSM, ASL, TOF-MRA, etc. It is compatible with 1.5T/3.0T clinical and research MRI scanners, and can meet the standardized analysis requirements of clinical research on psychiatric imaging.

---

## 环境搭建 | Environment Setup
### 中文
本项目运行依赖**系统级工具**与**Python第三方库**，以下为分系统的分步安装指南。

#### 1. 基础环境要求
- 操作系统：Windows 10/11、macOS 10.15+、Linux (Ubuntu 18.04+/CentOS 7+)
- Python版本：Python 3.8 ~ 3.11（推荐3.10，兼容性最佳）
- 内存：最低16GB，推荐32GB及以上（全脑高分辨率分析需64GB+）
- 存储：最低10GB可用空间，建议SSD以提升数据读写速度

#### 2. 系统级工具安装（dcm2niix）
`dcm2niix` 是DICOM转NIfTI格式的核心工具，必须提前安装：
##### macOS
通过Homebrew安装（推荐）：
```bash
# 安装Homebrew（如未安装）
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
# 安装dcm2niix
brew install dcm2niix
# 验证安装
dcm2niix --version
```

##### Windows
1. 方式一（通过conda安装，推荐）：
```bash
conda install -c conda-forge dcm2niix
# 验证安装
dcm2niix --version
```
2. 方式二：下载预编译二进制文件
   - 前往 [dcm2niix官方发布页](https://github.com/rordenlab/dcm2niix/releases) 下载Windows版本
   - 解压后将exe所在路径添加到系统环境变量`PATH`中
   - 重启终端后验证安装

##### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install dcm2niix
# 验证安装
dcm2niix --version
```

#### 3. Python虚拟环境配置（推荐）
为避免依赖冲突，建议使用conda或venv创建独立虚拟环境：
##### 方式一：conda（推荐）
```bash
# 创建虚拟环境
conda create -n mri_pipeline python=3.10
# 激活环境
conda activate mri_pipeline
```

##### 方式二：venv
```bash
# 创建虚拟环境
python -m venv mri_pipeline
# 激活环境（Windows）
mri_pipeline\Scripts\activate
# 激活环境（macOS/Linux）
source mri_pipeline/bin/activate
```

#### 4. Python依赖库安装
激活虚拟环境后，执行以下命令安装所有依赖：
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

### English
### Environment Setup
The operation of this project depends on **system-level tools** and **Python third-party libraries**. The following is a step-by-step installation guide for different operating systems.

#### 1. Basic Environment Requirements
- Operating System: Windows 10/11, macOS 10.15+, Linux (Ubuntu 18.04+/CentOS 7+)
- Python Version: Python 3.8 ~ 3.11 (3.10 is recommended for best compatibility)
- Memory: Minimum 16GB, 32GB+ recommended (64GB+ required for whole-brain high-resolution analysis)
- Storage: Minimum 10GB free space, SSD is recommended to improve data read/write speed

#### 2. System-level Tool Installation (dcm2niix)
`dcm2niix` is the core tool for DICOM to NIfTI format conversion and must be installed in advance:
##### macOS
Install via Homebrew (recommended):
```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
# Install dcm2niix
brew install dcm2niix
# Verify installation
dcm2niix --version
```

##### Windows
1. Method 1 (Install via conda, recommended):
```bash
conda install -c conda-forge dcm2niix
# Verify installation
dcm2niix --version
```
2. Method 2: Download precompiled binary
   - Go to [dcm2niix official releases](https://github.com/rordenlab/dcm2niix/releases) to download the Windows version
   - After decompression, add the path of the exe file to the system environment variable `PATH`
   - Restart the terminal and verify the installation

##### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install dcm2niix
# Verify installation
dcm2niix --version
```

#### 3. Python Virtual Environment Configuration (Recommended)
To avoid dependency conflicts, it is recommended to use conda or venv to create an independent virtual environment:
##### Method 1: conda (Recommended)
```bash
# Create virtual environment
conda create -n mri_pipeline python=3.10
# Activate environment
conda activate mri_pipeline
```

##### Method 2: venv
```bash
# Create virtual environment
python -m venv mri_pipeline
# Activate environment (Windows)
mri_pipeline\Scripts\activate
# Activate environment (macOS/Linux)
source mri_pipeline/bin/activate
```

#### 4. Python Dependency Installation
After activating the virtual environment, execute the following command to install all dependencies:
```bash
# Mandatory core dependencies
pip install numpy scipy nibabel nilearn dipy matplotlib seaborn networkx pandas
# Optional dependency (for interactive visualization)
pip install plotly
```

| Library | Core Purpose |
|--------|----------|
| nibabel | Read, write and process NIfTI format data |
| nilearn | fMRI data preprocessing, brain atlas loading, functional connectivity analysis |
| dipy | DTI diffusion data modeling, fiber tractography, structural connectivity analysis |
| numpy/scipy | Numerical calculation, signal processing and statistical analysis |
| matplotlib/seaborn | Static visualization plotting |
| networkx | Brain network graph theory analysis |
| pandas | Tabular output of analysis results |
| plotly | Interactive network visualization |

---

## 数据准备与规范 | Data Preparation & Specification
### 中文
#### 1. 原始DICOM数据目录结构
原始DICOM数据需按**序列分文件夹存放**，每个序列一个独立文件夹，文件夹名称需包含对应模态关键词，示例结构如下：
```
├── sort/  # 原始DICOM根目录
│   ├── 001_T1_MPRAGE/  # T1结构像序列
│   ├── 002_BOLD_REST/  # BOLD静息态功能像序列
│   ├── 003_DTI_64DIR/  # DTI弥散张量像序列
│   ├── 004_FLAIR_3D/   # 3D FLAIR序列
│   ├── 005_SWI/         # SWI磁敏感加权序列
│   └── 006_TOF_MRA/     # TOF血管成像序列
```

#### 2. 模态命名关键词规则
转换脚本将通过文件夹名称自动识别模态，识别规则如下，文件夹名称需包含对应关键词（不区分大小写）：

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
DTI数据需配套`.bval`和`.bvec`梯度文件，需与DICOM文件放在同一文件夹下，或与转换后的NIfTI文件同名同路径。

#### 4. 扫描参数规范要求
为保证分析结果的可靠性与可重复性，扫描参数需严格符合《T/CHIA 48-2024》规范，核心参数要求详见[扫描参数与质控规范](#扫描参数与质控规范--scan-parameters--quality-control-standards)章节。

### English
### Data Preparation & Specification
#### 1. Raw DICOM Data Directory Structure
Raw DICOM data must be stored in **separate folders by sequence**, with one independent folder for each sequence. The folder name must contain keywords corresponding to the modality. The example structure is as follows:
```
├── sort/  # Root directory of raw DICOM
│   ├── 001_T1_MPRAGE/  # T1 structural image sequence
│   ├── 002_BOLD_REST/  # BOLD resting-state functional image sequence
│   ├── 003_DTI_64DIR/  # DTI diffusion tensor image sequence
│   ├── 004_FLAIR_3D/   # 3D FLAIR sequence
│   ├── 005_SWI/         # SWI susceptibility weighted sequence
│   └── 006_TOF_MRA/     # TOF angiography sequence
```

#### 2. Modality Naming Keyword Rules
The conversion script will automatically recognize the modality through the folder name. The recognition rules are as follows. The folder name must contain the corresponding keywords (case-insensitive):

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
DTI data must be accompanied by `.bval` and `.bvec` gradient files, which should be placed in the same folder as the DICOM files, or in the same path with the same name as the converted NIfTI files.

#### 4. Scan Parameter Specification Requirements
To ensure the reliability and repeatability of the analysis results, the scan parameters must strictly comply with the *T/CHIA 48-2024* specification. For core parameter requirements, please refer to the [Scan Parameters & Quality Control Standards](#扫描参数与质控规范--scan-parameters--quality-control-standards) section.

---

## DICOM转NIfTI工具使用指南 | DICOM to NIfTI Tool Guide
### 中文
本工具`DICOM2NIfTI.py`用于批量将原始DICOM序列转换为NIfTI格式（`.nii.gz`压缩格式），并自动按模态分类存放。

#### 1. 脚本配置修改
打开`DICOM2NIfTI.py`，修改以下核心路径配置：
```python
# ======================
# 路径配置
# ======================
# 原始DICOM数据根目录（按序列分文件夹存放的路径）
BASE_DIR = "/Users/xxx/Downloads/brain_data/sort"
# 转换后NIfTI文件输出根目录
OUTPUT_DIR = "/Users/xxx/Downloads/brain_data/nifti"
```
- Windows系统路径格式示例：`BASE_DIR = "C:\\Users\\xxx\\brain_data\\sort"`
- 若dcm2niix未加入系统环境变量，需修改脚本中`os.environ["PATH"]`行，填写dcm2niix可执行文件所在路径。

#### 2. 运行脚本
激活虚拟环境后，在终端执行：
```bash
python DICOM2NIfTI.py
```

#### 3. 输出结果说明
运行完成后，输出目录结构如下：
```
├── nifti/  # 输出根目录
│   ├── T1/    # T1结构像转换结果
│   │   └── T1_xxx_xxx.nii.gz
│   ├── BOLD/  # BOLD功能像转换结果
│   │   └── BOLD_xxx_xxx.nii.gz
│   ├── DTI/   # DTI弥散像转换结果
│   │   ├── DTI_xxx_xxx.nii.gz
│   │   ├── DTI_xxx_xxx.bval
│   │   └── DTI_xxx_xxx.bvec
│   ├── FLAIR/ # FLAIR序列转换结果
│   ├── SWI/   # SWI序列转换结果
│   └── OTHER/ # 未识别模态的转换结果
```

#### 4. 核心功能说明
- 自动递归扫描原始目录下的所有序列文件夹
- 基于关键词自动识别模态并分类存放
- 自动生成规范的文件名，避免重名
- 输出`.nii.gz`压缩格式，节省存储空间
- 终端实时打印转换进度与成功/失败状态

### English
### DICOM to NIfTI Tool Guide
This tool `DICOM2NIfTI.py` is used to batch convert raw DICOM sequences to NIfTI format (`.nii.gz` compressed format), and automatically store them by modality classification.

#### 1. Script Configuration Modification
Open `DICOM2NIfTI.py` and modify the following core path configuration:
```python
# ======================
# Path Configuration
# ======================
# Root directory of raw DICOM data (path stored in folders by sequence)
BASE_DIR = "/Users/xxx/Downloads/brain_data/sort"
# Root output directory for converted NIfTI files
OUTPUT_DIR = "/Users/xxx/Downloads/brain_data/nifti"
```
- Windows system path format example: `BASE_DIR = "C:\\Users\\xxx\\brain_data\\sort"`
- If dcm2niix is not added to the system environment variable, you need to modify the `os.environ["PATH"]` line in the script and fill in the path of the dcm2niix executable file.

#### 2. Run the Script
After activating the virtual environment, execute in the terminal:
```bash
python DICOM2NIfTI.py
```

#### 3. Output Result Description
After the operation is completed, the output directory structure is as follows:
```
├── nifti/  # Output root directory
│   ├── T1/    # T1 structural image conversion results
│   │   └── T1_xxx_xxx.nii.gz
│   ├── BOLD/  # BOLD functional image conversion results
│   │   └── BOLD_xxx_xxx.nii.gz
│   ├── DTI/   # DTI diffusion image conversion results
│   │   ├── DTI_xxx_xxx.nii.gz
│   │   ├── DTI_xxx_xxx.bval
│   │   └── DTI_xxx_xxx.bvec
│   ├── FLAIR/ # FLAIR sequence conversion results
│   ├── SWI/   # SWI sequence conversion results
│   └── OTHER/ # Conversion results of unrecognized modalities
```

#### 4. Core Function Description
- Automatically and recursively scan all sequence folders under the original directory
- Automatically recognize modalities based on keywords and store them in categories
- Automatically generate standardized file names to avoid duplication
- Output `.nii.gz` compressed format to save storage space
- Real-time printing of conversion progress and success/failure status in the terminal

---

## 多模态fMRI处理流水线使用指南 | Multimodal fMRI Pipeline Guide
### 中文
本流水线`fMRI_pipeline.py`是核心分析工具，实现了多模态脑MRI数据的全流程标准化分析，严格遵循T/CHIA 48-2024规范的处理流程。

#### 1. 流水线核心功能模块
| 模块编号 | 模块名称 | 核心功能 |
|----------|----------|----------|
| 1 | 数据发现与校验 | 自动加载各模态NIfTI数据，校验数据维度与完整性 |
| 2 | rs-fMRI预处理 | 丢弃dummy扫描、空间平滑、带通滤波、空间标准化到MNI152模板 |
| 3 | 脑激活指标计算 | ALFF/fALFF（低频振幅）、ReHo（区域同质性）定量计算 |
| 4 | 功能连接分析 | 种子点功能连接、全脑功能连接矩阵构建、脑网络识别 |
| 5 | ICA静息态网络分析 | 独立成分分析，匹配默认网络、突显网络、中央执行网络等经典脑网络 |
| 6 | DTI结构连接分析 | 张量模型拟合、FA图计算、确定性纤维束追踪、全脑结构连接矩阵构建 |
| 7 | 多模态整合 | 功能连接(FC)与结构连接(SC)的一致性与差异性分析 |
| 8 | 图论网络分析 | 脑网络构建、度中心性/介数中心性/聚类系数计算、核心脑区识别 |
| 9/10 | 可视化 | 激活图、玻璃脑、连接矩阵热图、网络图、社区结构、环形连接图等20+种可视化图表 |

#### 2. 脚本配置修改
打开`fMRI_pipeline.py`，首先修改**核心路径配置**，对应DICOM转换后的NIfTI输出目录：
```python
# ======================
# 路径配置
# ======================
PATHS = {
    "t1":  "/Users/xxx/Downloads/brain_data/nifti/T1",    # T1数据路径
    "bold":"/Users/xxx/Downloads/brain_data/nifti/BOLD",  # BOLD数据路径
    "dti": "/Users/xxx/Downloads/brain_data/nifti/DTI",   # DTI数据路径
    "dsi": "/Users/xxx/Downloads/brain_data/nifti/DSI",   # DSI数据路径（可选）
}
# 分析结果输出目录
OUT_DIR          = Path("./brain_pipeline_outputs")
```

然后根据数据采集参数，修改**预处理关键参数**（需与扫描参数一致，严格遵循T/CHIA 48-2024规范）：
```python
# 预处理核心参数
TR        = 1.0    # BOLD序列的重复时间，单位ms，需与扫描参数一致
FWHM      = 6.0    # 空间平滑核大小，单位mm，规范推荐6mm
HP_FREQ   = 0.01   # 带通滤波低频截止，单位Hz，规范推荐0.01Hz
LP_FREQ   = 0.10   # 带通滤波高频截止，单位Hz，规范推荐0.1Hz
N_DUMMIES = 5      # 丢弃的初始dummy扫描帧数
FA_THRESH = 0.20   # DTI纤维束追踪的FA阈值
```

#### 3. 运行流水线
激活虚拟环境后，在终端执行：
```bash
python fMRI_pipeline.py
```
- 运行过程中，终端会实时打印各模块的执行进度、关键参数与质控信息
- 运行时长取决于数据分辨率与计算机性能，单被试全流程通常需要5-20分钟
- 若某一模态数据缺失，流水线会自动跳过对应模块，不影响其他模态的分析

#### 4. 输出结果说明
所有分析结果、可视化图表、日志文件均保存在`OUT_DIR`指定的目录中，核心输出文件分类如下：

| 文件类型 | 核心文件 | 说明 |
|----------|----------|------|
| 预处理后NIfTI数据 | bold_preprocessed_mni.nii.gz、alff.nii.gz、falff.nii.gz、reho.nii.gz、fa_map.nii.gz | 预处理后的功能像、激活指标图、DTI的FA图等，可用于后续二次分析 |
| 连接矩阵数据 | fc_matrix.npy/csv、sc_matrix.npy/csv | 全脑功能连接矩阵、结构连接矩阵，npy格式用于Python二次分析，csv格式可直接用Excel打开 |
| 定量分析表格 | pipeline_results_summary.csv、alff_by_roi.csv | 核心脑区Hub排名、最强功能连接、各脑区ALFF均值等定量结果表格 |
| 静态可视化图表 | activation_maps.png、fc_heatmap.png、brain_network_graph.png、multimodal_comparison.png等 | 20+种标准化可视化图表，涵盖激活图、连接热图、网络图、多模态对比图等 |
| 交互式可视化 | interactive_network.html、interactive_community_network.html | 可交互的脑网络图，支持放大、悬停查看详细信息，可直接用浏览器打开 |
| 日志文件 | pipeline_terminal_output.txt | 流水线全流程的终端输出日志，用于结果复现与问题排查 |

### English
### Multimodal fMRI Pipeline Guide
This pipeline `fMRI_pipeline.py` is the core analysis tool, which implements the full-process standardized analysis of multimodal brain MRI data, strictly following the processing flow of the T/CHIA 48-2024 specification.

#### 1. Core Functional Modules of the Pipeline
| Module No. | Module Name | Core Function |
|----------|----------|----------|
| 1 | Data Discovery & Validation | Automatically load NIfTI data of each modality, verify data dimension and integrity |
| 2 | rs-fMRI Preprocessing | Drop dummy scans, spatial smoothing, bandpass filtering, spatial normalization to MNI152 template |
| 3 | Brain Activation Metrics Calculation | Quantitative calculation of ALFF/fALFF (Amplitude of Low-Frequency Fluctuations) and ReHo (Regional Homogeneity) |
| 4 | Functional Connectivity Analysis | Seed-based functional connectivity, whole-brain functional connectivity matrix construction, brain network identification |
| 5 | ICA Resting-State Network Analysis | Independent component analysis, matching classic brain networks such as default mode network, salience network, and central executive network |
| 6 | DTI Structural Connectivity Analysis | Tensor model fitting, FA map calculation, deterministic fiber tractography, whole-brain structural connectivity matrix construction |
| 7 | Multimodal Integration | Consistency and difference analysis between functional connectivity (FC) and structural connectivity (SC) |
| 8 | Graph Theory Network Analysis | Brain network construction, degree centrality/betweenness centrality/clustering coefficient calculation, core brain region identification |
| 9/10 | Visualization | 20+ visualization charts including activation maps, glass brain, connectivity matrix heatmap, network graph, community structure, circular connectivity diagram, etc. |

#### 2. Script Configuration Modification
Open `fMRI_pipeline.py`, first modify the **core path configuration**, corresponding to the NIfTI output directory after DICOM conversion:
```python
# ======================
# Path Configuration
# ======================
PATHS = {
    "t1":  "/Users/xxx/Downloads/brain_data/nifti/T1",    # T1 data path
    "bold":"/Users/xxx/Downloads/brain_data/nifti/BOLD",  # BOLD data path
    "dti": "/Users/xxx/Downloads/brain_data/nifti/DTI",   # DTI data path
    "dsi": "/Users/xxx/Downloads/brain_data/nifti/DSI",   # DSI data path (optional)
}
# Analysis result output directory
OUT_DIR          = Path("./brain_pipeline_outputs")
```

Then modify the **key preprocessing parameters** according to the data acquisition parameters (must be consistent with the scan parameters, strictly follow the T/CHIA 48-2024 specification):
```python
# Core preprocessing parameters
TR        = 1.0    # Repetition time of BOLD sequence, in ms, must be consistent with scan parameters
FWHM      = 6.0    # Spatial smoothing kernel size, in mm, 6mm recommended by the specification
HP_FREQ   = 0.01   # Low-frequency cutoff of bandpass filtering, in Hz, 0.01Hz recommended by the specification
LP_FREQ   = 0.10   # High-frequency cutoff of bandpass filtering, in Hz, 0.1Hz recommended by the specification
N_DUMMIES = 5      # Number of initial dummy scans to drop
FA_THRESH = 0.20   # FA threshold for DTI fiber tractography
```

#### 3. Run the Pipeline
After activating the virtual environment, execute in the terminal:
```bash
python fMRI_pipeline.py
```
- During the operation, the terminal will print the execution progress, key parameters and quality control information of each module in real time
- The running time depends on the data resolution and computer performance, and the full process for a single subject usually takes 5-20 minutes
- If data of a certain modality is missing, the pipeline will automatically skip the corresponding module without affecting the analysis of other modalities

#### 4. Output Result Description
All analysis results, visualization charts, and log files are saved in the directory specified by `OUT_DIR`. The core output files are classified as follows:

| File Type | Core Files | Description |
|----------|----------|------|
| Preprocessed NIfTI Data | bold_preprocessed_mni.nii.gz、alff.nii.gz、falff.nii.gz、reho.nii.gz、fa_map.nii.gz | Preprocessed functional images, activation metric maps, DTI FA maps, etc., which can be used for subsequent secondary analysis |
| Connectivity Matrix Data | fc_matrix.npy/csv、sc_matrix.npy/csv | Whole-brain functional connectivity matrix and structural connectivity matrix. The npy format is used for Python secondary analysis, and the csv format can be directly opened with Excel |
| Quantitative Analysis Tables | pipeline_results_summary.csv、alff_by_roi.csv | Quantitative result tables such as core brain region Hub ranking, strongest functional connections, and ALFF mean value of each brain region |
| Static Visualization Charts | activation_maps.png、fc_heatmap.png、brain_network_graph.png、multimodal_comparison.png, etc. | 20+ standardized visualization charts, including activation maps, connectivity heatmaps, network graphs, multimodal comparison charts, etc. |
| Interactive Visualization | interactive_network.html、interactive_community_network.html | Interactive brain network graph, supporting zooming and hovering to view detailed information, can be directly opened with a browser |
| Log File | pipeline_terminal_output.txt | Terminal output log of the whole pipeline process, used for result reproduction and troubleshooting |

---

## 扫描参数与质控规范 | Scan Parameters & Quality Control Standards
### 中文
本章节内容严格遵循《T/CHIA 48-2024 精神影像脑结构功能成像技术与信息处理规范》，列出临床与科研场景下的核心扫描参数与质控要求，确保数据质量符合分析标准。

#### 1. 设备基础参数要求
| 磁场强度 | 应用场景 | 梯度性能 | 独立射频通道数 | 线圈要求 |
|----------|----------|----------|----------------|----------|
| 1.5T | 科研 | 33mT/m，125mT/m/s | 24 | 头颈联合线圈通道数≥16 |
| 1.5T | 临床 | 30mT/m，100mT/m/s | 16 | 头颈联合线圈通道数≥10 |
| 3.0T | 科研 | 44mT/m，200mT/m/s | 48/64/128 | 专用头部线圈≥32通道 |
| 3.0T | 临床 | 36mT/m，150mT/m/s | 32 | 头颈联合线圈头部单通道数≥20 |

#### 2. 核心序列推荐参数（3.0T科研场景）
| 序列 | 核心参数 |
|------|----------|
| 3D-T1WI结构像 | TR=2300ms，TE=2.3ms，TI=900ms，FA=8°，体素大小0.8×0.8×0.8~1×1×1mm |
| BOLD功能像 | TR=2000~3000ms，TE=30ms，FA=90°，体素大小1.2×1.2×1.2~2×2×2mm，层间距0，时间点500~600 |
| DTI弥散像 | TR=7000ms，TE=89ms，体素大小1.5×1.5×1.5~2×2×2mm，b值=1000/2000/3000 s/mm²，梯度方向≥64 |
| 3D-FLAIR | TR=8000ms，TE=150ms，TI=2000ms，体素大小0.8×0.8×0.8~1×1×1mm |
| 3D-ASL | TR=3500~6500ms，TI=1500/2000/2500/3000ms（科研多TI），层厚3mm，采集时间7min |
| SWI | TR=20~30ms，TE=20ms，层厚1.0~1.5mm，层面内分辨率0.5~0.7mm |

#### 3. 核心序列质控要求
| 序列 | 质控指标 | 合格要求 |
|------|----------|----------|
| 3D-T1WI | 信噪比(SNR) | ≥65db |
| | 对比度噪声比(CNR) | ≥60db |
| | 图像整体均匀性 | ≥80% |
| | 灰白质分界 | 清晰可辨 |
| | 伪影 | 无明显运动伪影、并行采集伪影 |
| BOLD | 信噪比(SNR) | ≥65db |
| | 几何畸变率 | ≤5% |
| | 信号漂移值 | ≤1% |
| | 头动控制 | 平动位移≤1/2体素尺寸 |
| DTI | 信噪比(SNR) | ≥65db |
| | 几何畸变率 | ≤5% |
| | FA图像 | 脑白质FA值显著高于灰质 |
| | 纤维束追踪 | 主要纤维束无变形、缺失 |
| MRS | 水峰线宽 | ≤10HZ（人体），≤5HZ（体模） |
| | 抑水率 | ≥95%（人体），≥98%（体模） |
| | 信噪比 | >10 |

### English
### Scan Parameters & Quality Control Standards
The content of this chapter strictly follows the *T/CHIA 48-2024 Specification for structural and functional imaging technology and information processing in psychoradiology*, and lists the core scan parameters and quality control requirements in clinical and research scenarios to ensure that the data quality meets the analysis standards.

#### 1. Basic Equipment Parameter Requirements
| Magnetic Field Strength | Application Scenario | Gradient Performance | Independent RF Channels | Coil Requirements |
|----------|----------|----------|----------------|----------|
| 1.5T | Research | 33mT/m, 125mT/m/s | 24 | Head-neck joint coil ≥16 channels |
| 1.5T | Clinical | 30mT/m, 100mT/m/s | 16 | Head-neck joint coil ≥10 channels |
| 3.0T | Research | 44mT/m, 200mT/m/s | 48/64/128 | Dedicated head coil ≥32 channels |
| 3.0T | Clinical | 36mT/m, 150mT/m/s | 32 | Head single channel in head-neck joint coil ≥20 |

#### 2. Recommended Core Sequence Parameters (3.0T Research Scenario)
| Sequence | Core Parameters |
|------|----------|
| 3D-T1WI Structural | TR=2300ms, TE=2.3ms, TI=900ms, FA=8°, Voxel size 0.8×0.8×0.8~1×1×1mm |
| BOLD Functional | TR=2000~3000ms, TE=30ms, FA=90°, Voxel size 1.2×1.2×1.2~2×2×2mm, Slice gap 0, Time points 500~600 |
| DTI Diffusion | TR=7000ms, TE=89ms, Voxel size 1.5×1.5×1.5~2×2×2mm, b-value=1000/2000/3000 s/mm², Gradient directions ≥64 |
| 3D-FLAIR | TR=8000ms, TE=150ms, TI=2000ms, Voxel size 0.8×0.8×0.8~1×1×1mm |
| 3D-ASL | TR=3500~6500ms, TI=1500/2000/2500/3000ms (multi-TI for research), Slice thickness 3mm, Acquisition time 7min |
| SWI | TR=20~30ms, TE=20ms, Slice thickness 1.0~1.5mm, In-plane resolution 0.5~0.7mm |

#### 3. Core Sequence Quality Control Requirements
| Sequence | QC Index | Qualification Requirement |
|------|----------|----------|
| 3D-T1WI | Signal-to-Noise Ratio (SNR) | ≥65db |
| | Contrast-to-Noise Ratio (CNR) | ≥60db |
| | Image Uniformity | ≥80% |
| | Gray-white Matter Boundary | Clearly distinguishable |
| | Artifacts | No obvious motion artifacts, parallel acquisition artifacts |
| BOLD | Signal-to-Noise Ratio (SNR) | ≥65db |
| | Geometric Distortion Rate | ≤5% |
| | Signal Drift | ≤1% |
| | Head Motion Control | Translation displacement ≤1/2 voxel size |
| DTI | Signal-to-Noise Ratio (SNR) | ≥65db |
| | Geometric Distortion Rate | ≤5% |
| | FA Image | White matter FA value significantly higher than gray matter |
| | Fiber Tractography | No deformation or missing of main fiber bundles |
| MRS | Water Peak Linewidth | ≤10HZ (in vivo), ≤5HZ (phantom) |
| | Water Suppression Rate | ≥95% (in vivo), ≥98% (phantom) |
| | Signal-to-Noise Ratio | >10 |

---

## 常见问题 | FAQ
### 中文
#### Q1: 运行DICOM2NIfTI.py时提示“dcm2niix: command not found”
A: 这是因为dcm2niix未正确安装或未加入系统环境变量。请重新按照环境搭建章节安装dcm2niix，或在脚本中直接指定dcm2niix的绝对路径。

#### Q2: 运行流水线时提示“ImportError: No module named 'xxx'”
A: 这是因为对应的Python依赖库未安装。请激活虚拟环境后，重新执行依赖安装命令，确保所有必选依赖都成功安装。

#### Q3: BOLD数据预处理失败，提示“维度不匹配”
A: 请检查BOLD数据的维度，必须是4维数据（x, y, z, 时间点）。同时检查脚本中`TR`参数是否与扫描参数一致，`N_DUMMIES`设置是否小于总时间点数。

#### Q4: DTI模块运行失败，提示“找不到bval/bvec文件”
A: 请确保`.bval`和`.bvec`文件与DTI的NIfTI文件放在同一目录下，且文件名前缀完全一致。例如：`DTI_001.nii.gz`对应`DTI_001.bval`和`DTI_001.bvec`。

#### Q5: 可视化图片显示不全、文字重叠
A: 请检查matplotlib版本，推荐3.5+版本。可修改脚本中绘图的`figsize`参数，调整画布大小以适配显示。

### English
#### Q1: When running DICOM2NIfTI.py, it prompts "dcm2niix: command not found"
A: This is because dcm2niix is not installed correctly or not added to the system environment variable. Please reinstall dcm2niix according to the Environment Setup chapter, or directly specify the absolute path of dcm2niix in the script.

#### Q2: When running the pipeline, it prompts "ImportError: No module named 'xxx'"
A: This is because the corresponding Python dependency library is not installed. Please activate the virtual environment and re-execute the dependency installation command to ensure that all mandatory dependencies are successfully installed.

#### Q3: BOLD data preprocessing failed, prompting "dimension mismatch"
A: Please check the dimension of the BOLD data, which must be 4-dimensional (x, y, z, time points). At the same time, check whether the `TR` parameter in the script is consistent with the scan parameters, and whether the `N_DUMMIES` setting is less than the total number of time points.

#### Q4: DTI module failed to run, prompting "cannot find bval/bvec files"
A: Please ensure that the `.bval` and `.bvec` files are placed in the same directory as the DTI NIfTI file, and the file name prefixes are exactly the same. For example: `DTI_001.nii.gz` corresponds to `DTI_001.bval` and `DTI_001.bvec`.

#### Q5: The visualization picture is not fully displayed and the text overlaps
A: Please check the matplotlib version, version 3.5+ is recommended. You can modify the `figsize` parameter of the plot in the script to adjust the canvas size to fit the display.

---

## 参考文献 | References
1. T/CHIA 48-2024, 精神影像脑结构功能成像技术与信息处理规范[S]. 中国卫生信息与健康医疗大数据学会, 2024.
2. T/CHIA 48-2024, Specification for structural and functional imaging technology and information processing in psychoradiology[S]. China Health Information and Healthcare Big Data Association, 2024.
3. Abraham A, Pedregosa F, Eickenberg M, et al. Machine learning for neuroimaging with scikit-learn[J]. Frontiers in neuroinformatics, 2014, 8: 14.
4. Garyfallidis E, Brett M, Amirbekian B, et al. Dipy, a library for the analysis of diffusion MRI data[J]. Frontiers in neuroinformatics, 2014, 8: 8.
5. Li X, Morgan PS, Ashburner J, et al. The NIfTI-1 data format: a new standard for neuroimaging informatics[J]. Neuroimage, 2004, 22(4): 1737-1744.
