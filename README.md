# MRI数据 → rs-fMRI分析 全流程教程（实战版）



# 📍 第0步：你的数据情况（重要认知）

在开始所有操作前，先明确自身数据的核心信息，避免走弯路——这是你之前踩坑的关键原因，务必重点关注！

你当前数据核心信息：

- 数据类型：多模态 MRI（包含 T1、BOLD、DTI、SWI、FLAIR 等模态）

- 当前格式：❌ DICOM（.dcm）—— 流水线无法直接识别，必须转换

- 分析目标：✅ rs-fMRI（静息态功能磁共振）+ 神经环路分析

👉 关键结论（必记）：

```text
必须先把 DICOM 格式（.dcm）转换为 NIfTI 格式（.nii.gz），否则无法运行 fMRI_pipeline.py 分析
```

# 🛠️ 第1步：安装DICOM→NIfTI转换工具（dcm2niix）

转换DICOM文件的核心工具是 `dcm2niix`，轻量、高效且支持批量转换，以下是Mac系统的安装方法（Windows/Linux可参考对应教程）：

## Mac 系统安装（终端执行）：

```bash
brew install dcm2niix
```

说明：若未安装 Homebrew（brew命令报错），先执行 `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"` 安装Homebrew，再重新执行上述命令。

# 🔍 第2步：确认工具路径（非常关键，避免后续报错）

安装完成后，需确认 `dcm2niix` 的安装路径，后续写转换脚本时会用到，终端输入以下命令：

```bash
which dcm2niix
```

👉 正常情况下，你会得到类似如下路径（需记住，后续替换到脚本中）：

```bash
/opt/homebrew/bin/dcm2niix
```

⚠️ 注意：每个人的路径可能不同，务必以自己终端输出的路径为准，写错会导致转换失败。

# 🚀 第3步：编写批量转换脚本（DICOM→NIfTI）

手动转换多个DICOM文件夹效率极低，编写Python脚本实现批量转换，无需逐个操作，步骤如下：

## 👉 步骤1：新建脚本文件

在电脑中新建一个文本文件，重命名为 `DICOM2NIfTI.py`（注意后缀是 .py，不是 .txt）。

## 👉 步骤2：复制以下代码（替换对应路径）

```python
import os
import subprocess
from pathlib import Path

# 替换为你的DICOM文件根目录（所有DICOM文件夹都在这个目录下）
BASE_DIR = "你的目录地址"
# 替换为你想要保存NIfTI文件的目录（与后续流水线路径对应）
OUTPUT_DIR = "你的目录地址"

# 替换为第2步得到的dcm2niix路径（关键！不能错）
DCM2NIIX = "/opt/homebrew/bin/dcm2niix"

# 自动创建输出目录（不存在则创建，避免报错）
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 模态映射：根据DICOM文件夹名称自动识别模态，分类保存
MODALITY_MAP = {
    "T1": ["T1", "TFE"],       # 包含这些关键词的文件夹，识别为T1模态
    "BOLD": ["BOLD", "MB2"],   # 包含这些关键词的文件夹，识别为BOLD模态（rs-fMRI核心）
    "DTI": ["DTI"],            # DTI模态
    "DSI": ["DSI"],            # DSI模态
    "FLAIR": ["FLAIR"],        # FLAIR模态
    "SWI": ["SWI"],            # SWI模态
    "QSM": ["QSM"],            # QSM模态
}

def detect_modality(name):
    """根据文件夹名称，自动识别MRI模态"""
    for modality, keys in MODALITY_MAP.items():
        if any(key.lower() in name.lower() for key in keys):
            return modality
    return "OTHER"  # 无法识别的模态，归为OTHER

# 获取所有DICOM文件夹（仅遍历目录，不处理文件）
folders = [f for f in Path(BASE_DIR).iterdir() if f.is_dir()]

# 批量转换每个DICOM文件夹
for folder in folders:
    # 自动识别当前文件夹的模态
    modality = detect_modality(folder.name)
    # 每个模态单独创建文件夹，避免混乱（与流水线PATHS配置对应）
    out_dir = os.path.join(OUTPUT_DIR, modality)
    os.makedirs(out_dir, exist_ok=True)

    # 构建dcm2niix转换命令
    cmd = [
        DCM2NIIX,               # 转换工具路径
        "-z", "y",              # 压缩输出为.nii.gz格式（节省空间）
        "-f", f"{modality}_%p_%s",  # 输出文件名格式：模态_序列_扫描号
        "-o", out_dir,          # 输出目录
        str(folder)             # 当前要转换的DICOM文件夹路径
    ]

    # 打印转换信息，方便查看进度
    print(f"\nProcessing: {folder.name} → 模态：{modality}")

    try:
        # 执行转换命令
        subprocess.run(cmd, check=True)
        print("✅ 转换成功")
    except Exception as e:
        # 转换失败时，打印错误信息，不中断整个脚本
        print(f"❌ 转换失败：{e}")

print("\n🎉 所有DICOM文件转换完成！")
```

⚠️ 重点替换3处内容（必做）：

1. `BASE_DIR`：你的DICOM文件根目录（所有DICOM文件夹都放在这个目录下）；

2. `OUTPUT_DIR`：NIfTI文件的输出目录（建议与你之前的 `fMRI_pipeline.py` 中 `PATHS` 路径一致）；

3. `DCM2NIIX`：第2步得到的 `dcm2niix` 安装路径。

# ▶️ 第4步：运行转换脚本，生成NIfTI文件

脚本编写完成后，打开终端，执行以下步骤：

1. 激活之前创建的虚拟环境（避免库版本冲突）：
        `# Windows：
conda activate neuro
# Mac/Linux：
source activate neuro`

2. 切换到 `DICOM2NIfTI.py` 所在的目录（例如脚本在桌面，输入 `cd ~/Desktop`）；

3. 运行脚本：
       `python DICOM2NIfTI.py`

运行过程中，终端会打印每个文件夹的转换状态，显示「✅ 转换成功」即代表该文件夹的DICOM已成功转为NIfTI格式；若显示失败，可根据错误信息排查（常见原因：路径写错、DICOM文件损坏）。

# 📦 第5步：检查输出结果（必须做，避免后续流水线报错）

转换完成后，需确认NIfTI文件是否生成成功，重点检查BOLD模态（rs-fMRI分析的核心），终端输入以下命令：

```bash
ls 你的目录地址nifti/BOLD/
```

👉 正常情况下，你会看到类似如下文件（缺一不可）：

```text
BOLD_xxx_xxx.nii.gz   ✅（核心文件，流水线唯一识别的格式）
BOLD_xxx_xxx.json     ✅（辅助信息文件，包含扫描参数，可选但建议保留）
```

⚠️ 若未看到 `.nii.gz` 文件，说明转换失败，需重新检查脚本中的3个路径是否正确，或DICOM文件夹是否完整。

# 🧠 第6步：验证BOLD数据是否符合分析要求

rs-fMRI的BOLD数据必须是「4D数据」（x,y,z,time，即空间维度+时间维度），若为3D数据（仅x,y,z），则无法进行功能分析，需验证：

1. 终端输入 `python`，进入Python交互模式；

2. 输入以下代码（替换为你的BOLD文件路径）：`import nibabel as nib

# 替换为你实际的BOLD文件路径（可从第5步的ls结果中复制）
img = nib.load("你的目录地址nifti/BOLD/BOLD_xxx_xxx.nii.gz")
print(img.shape)`

## ✅ 正确结果（符合要求）：

```text
(x, y, z, time)  # 例如：(64, 64, 32, 200)，最后一个数字是时间点，大于100即为正常
```

## ❌ 错误情况（不符合要求）：

```text
(x, y, z)  # 说明该文件不是rs-fMRI的BOLD数据，需重新确认DICOM文件夹是否正确
```

验证完成后，输入 `exit()` 退出Python交互模式。

# ⚙️ 第7步：配置fMRI流水线（fMRI_pipeline.py）

数据转换并验证成功后，需修改 `fMRI_pipeline.py` 中的路径配置，确保流水线能找到NIfTI文件，重点修改 `PATHS` 字典（之前报错的核心原因）：

## 👉 找到代码中的PATHS配置段，替换为以下内容（与你的NIfTI输出路径对应）：

```python
# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION  (edit paths here if needed)
# ─────────────────────────────────────────────────────────────────────────────
PATHS = {
    "t1":  "你的目录地址nifti/T1",  # T1模态文件夹路径（无需带文件名）
    "bold": "你的目录地址nifti/BOLD",  # BOLD模态文件夹路径（核心）
    "dti": "你的目录地址nifti/DTI",  # DTI模态文件夹路径
    "dsi": "你的目录地址nifti/DSI",  # DSI模态文件夹路径
}

OUT_DIR = Path("./brain_pipeline_outputs")  # 流水线结果输出目录（可自定义）
OUT_DIR.mkdir(parents=True, exist_ok=True)
```

⚠️ 关键注意事项（必看）：

- 路径填写「文件夹路径」，**不要带具体的.nii.gz文件名**（流水线会自动搜索文件夹内的NIfTI文件）；

- 路径中的 `你的处理的3D大脑` 需与你实际的文件夹名称一致，避免中文拼写错误；

- 若不需要DSI模态，可删除 `"dsi": ...` 这一行，不影响核心的rs-fMRI分析。

补充配置（可选，根据你的扫描参数调整）：

```python
# Preprocessing parameters（根据你的BOLD扫描参数修改）
TR          = 2.0          # 重复时间（秒），需与你的扫描参数一致（常见2.0/1.5）
FWHM        = 6.0          # 空间平滑核（mm），默认即可
HP_FREQ     = 0.01         # 高通滤波（Hz），默认即可
LP_FREQ     = 0.10         # 低通滤波（Hz），默认即可
N_DUMMIES   = 5            # 丢弃的初始扫描（dummy scans），默认即可
```

# 🚀 第8步：运行fMRI流水线，开始rs-fMRI分析

路径配置完成后，终端执行以下命令，启动流水线（确保已激活neuro虚拟环境）：

```bash
# 切换到fMRI_pipeline.py所在的目录（例如：cd ~/Downloads/你的处理的3D大脑）
# 运行流水线
/opt/anaconda3/envs/neuro/bin/python fMRI_pipeline.py
```

运行说明：

- 流水线会自动执行9个模块（数据验证、BOLD预处理、激活指标计算、功能连接、ICA网络检测等）；

- 运行时间根据数据量而定（BOLD时间点越多，时间越长，通常10-30分钟）；

- 终端会实时打印运行进度，若出现 `[FATAL]` 报错，需优先查看路径是否正确（参考第9步常见错误）。

# 🚨 第9步：常见错误总结（你刚刚已经踩过，重点规避）

整理了流水线运行和数据转换过程中最常见的4个错误，附详细解决方法，直接对照排查即可：

## ❌ 错误1：找不到dcm2niix（转换脚本运行报错）

```text
FileNotFoundError: [Errno 2] No such file or directory: 'dcm2niix'
```

✔️ 解决方法：

重新执行第2步 `which dcm2niix`，获取正确的工具路径，替换 `DICOM2NIfTI.py` 中的 `DCM2NIIX` 变量（确保路径无拼写错误）。

## ❌ 错误2：流水线提示“Found 0 NIfTI files”

```text
[BOLD] Found 0 NIfTI file(s) in /xxx/xxx/nifti/BOLD
  [WARN] No files found for bold. Check path.
```

✔️ 核心原因：

要么未执行DICOM→NIfTI转换，要么 `fMRI_pipeline.py` 中的 `PATHS` 路径错误（填写了文件路径而非文件夹路径，或路径拼写错误）。

✔️ 解决方法：

1. 先执行第5步，确认 `nifti/BOLD` 文件夹下有 `.nii.gz` 文件；

2. 重新检查 `PATHS` 中的路径，确保是「文件夹路径」，无文件名后缀。

## ❌ 错误3：nilearn导入报错（流水线运行报错）

```text
ImportError: cannot import name 'fetch_mni152_template' from 'nilearn.datasets'
```

✔️ 解决方法：

打开 `fMRI_pipeline.py`，找到对应导入语句，替换为：

```python
from nilearn.datasets import load_mni152_template
```

原因：nilearn版本更新后，`fetch_mni152_template` 已更名为 `load_mni152_template`。

## ❌ 错误4：scikit-learn / scipy 报错（环境冲突）

```text
AttributeError: module 'scipy.misc' has no attribute 'imresize'
```

✔️ 核心原因：

Python版本过高（如3.13），与scikit-learn、scipy库不兼容（这些库对高版本Python支持滞后）。

✔️ 解决方法：

1. 删除原有虚拟环境，重新创建Python 3.10版本的环境（兼容性最佳）：
        `# 删除原有neuro环境（若已创建）
conda remove -n neuro --all -y

# 重新创建Python 3.10的虚拟环境
conda create -n neuro python=3.10 -y

# 重新激活环境
source activate neuro  # Mac/Linux
# conda activate neuro  # Windows`

2. 重新执行第2.3步，安装所有依赖库。

# 🧠 第10步：流水线运行成功后，你可以做什么分析

一旦流水线跑通，会自动生成丰富的分析结果，涵盖功能分析、神经网络、图论分析三大类，满足你的rs-fMRI+神经环路分析需求：

## 🔬 功能分析（核心，rs-fMRI重点）

- ALFF（低频振幅）：反映脑区的自发活动强度，数值越高，脑区自发活动越活跃；

- ReHo（局部同步性）：反映某一脑区与周围脑区的活动同步程度，体现局部神经环路的协同性；

- Functional Connectivity（功能连接）：分析不同脑区之间的活动相关性，识别功能网络。

## 🔗 神经网络分析

- Default Mode Network（DMN，默认模式网络）：静息状态下最活跃的网络，与自我认知、记忆相关；

- 海马回路：与学习记忆相关的核心环路，可通过种子点功能连接分析；

- 前额叶网络：与执行功能、情绪调节相关的网络。

## 📊 图论分析（神经环路核心）

- Hub脑区（核心节点）：网络中连接最多、影响力最强的脑区，是神经环路的关键节点；

- 神经环路：识别脑区之间的强连接路径，明确不同功能网络的连接模式；

- 网络强连接：筛选出功能连接或结构连接最强的脑区对，用于后续机制研究。

# 🎯 第11步：最终输出结果说明（重点关注）

流水线运行完成后，所有结果会保存到 `brain_pipeline_outputs` 文件夹（可在代码中修改 `OUT_DIR` 自定义路径），核心结果如下，可直接用于论文绘图和数据分析：

- 🧠 功能指标文件（.nii.gz）：alff.nii.gz、falff.nii.gz、reho.nii.gz（可用于脑图可视化）；

- 🔗 连接矩阵文件（.npy）：fc_matrix.npy（功能连接矩阵）、sc_matrix.npy（结构连接矩阵）；

- 📊 可视化图像（.png）：activation_maps.png（激活图）、brain_network_graph.png（脑连接图）、summary_dashboard.png（结果汇总面板）；

- 🌐 交互式图表（.html）：interactive_network.html（可拖动查看脑网络节点和连接）；

- 📋 文本结果：终端输出的「Top 10活跃脑区」「Top 10强连接」「Hub脑区排名」，可直接复制整理。

# 💡 实战小贴士（避坑关键）

1. 所有路径尽量避免中文空格、特殊字符（如括号、逗号），否则可能导致路径识别失败；

2. 每次修改代码后，保存再运行，避免因未保存导致配置不生效；

3. 若仅需rs-fMRI分析，可删除代码中DTI、DSI相关模块（不影响核心功能），加快运行速度；

4. 若BOLD数据时间点过少（如少于50个），会影响ALFF、ReHo的计算结果，建议确保时间点≥100。
