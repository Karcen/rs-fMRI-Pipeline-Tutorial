import os
import subprocess
from pathlib import Path

import os
os.environ["PATH"] += ":/opt/homebrew/bin/" #cmd输入`which dcm2niix`即可获得地址


# ======================
# 路径配置
# ======================
BASE_DIR = "/Users/karcenzheng/Downloads/Jiacheng_Zheng的3D大脑/sort"
OUTPUT_DIR = "/Users/karcenzheng/Downloads/Jiacheng_Zheng的3D大脑/nifti"

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================
# 模态识别规则（自动分类）
# ======================
MODALITY_MAP = {
    "T1": ["T1", "TFE"],
    "BOLD": ["BOLD", "MB2"],
    "DTI": ["DTI"],
    "DSI": ["DSI"],
    "FLAIR": ["FLAIR"],
    "SWI": ["SWI", "SWIp"],
    "QSM": ["QSM"],
}

def detect_modality(folder_name):
    for modality, keywords in MODALITY_MAP.items():
        for k in keywords:
            if k.lower() in folder_name.lower():
                return modality
    return "OTHER"

# ======================
# 扫描所有序列
# ======================
series_folders = [f for f in Path(BASE_DIR).iterdir() if f.is_dir()]

print("=" * 60)
print("🧠 DICOM → NIfTI 批量转换")
print("=" * 60)

for folder in series_folders:
    folder_path = str(folder)
    modality = detect_modality(folder.name)

    print(f"\n📂 Processing: {folder.name}")
    print(f"   Detected modality: {modality}")

    # 每个模态单独文件夹
    out_dir = os.path.join(OUTPUT_DIR, modality)
    os.makedirs(out_dir, exist_ok=True)

    # dcm2niix 命令
    cmd = [
        "dcm2niix",
        "-z", "y",                  # 压缩为 .nii.gz
        "-f", f"{modality}_%p_%s",  # 文件命名
        "-o", out_dir,
        folder_path
    ]

    try:
        subprocess.run(cmd, check=True)
        print("   ✅ Converted successfully")
    except subprocess.CalledProcessError:
        print("   ❌ Conversion failed")

print("\n🎉 全部转换完成！")
print(f"📁 输出目录: {OUTPUT_DIR}")