# %%
# !/usr/bin/env python3
"""
从FreeSurfer重建结果导出带解剖颜色的PLY模型
适配Mac ARM架构，解决格式兼容/颜色丢失问题
"""
import nibabel.freesurfer as fs
import numpy as np


def export_colored_ply(surf_path, annot_path, output_path):
    """
    核心功能：读取表面+注释，导出带RGB颜色的PLY文件
    :param surf_path: FreeSurfer表面文件（如lh.pial）
    :param annot_path: 解剖注释文件（如lh.aparc.annot）
    :param output_path: 输出彩色PLY路径
    """
    # 1. 读取表面几何（顶点坐标+面索引）
    coords, faces = fs.read_geometry(surf_path)
    # 2. 读取注释（标签+颜色表+脑区名称）
    labels, ctab, _ = fs.read_annot(annot_path)

    # 3. 映射每个顶点的解剖颜色（RGB 0-255）
    vertex_colors = np.zeros((len(coords), 3), dtype=np.uint8)
    for i, label in enumerate(labels):
        if 0 <= label < len(ctab):
            vertex_colors[i] = ctab[label, :3]  # 取原生解剖色
        else:
            vertex_colors[i] = [128, 128, 128]  # 未知区域设为灰色

    # 4. 写入标准PLY文件（避免Blender识别错误）
    with open(output_path, 'w') as f:
        # PLY头部（严格遵循标准格式）
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(coords)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        # 写入顶点+颜色
        for v, c in zip(coords, vertex_colors):
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]} {c[1]} {c[2]}\n")
        # 写入面
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

    print(f"✅ 彩色PLY已生成：{output_path}")


# === 主程序 ===
if __name__ == "__main__":
    # 配置路径（替换为你的实际路径）
    SUBJECTS_DIR = "你的路径"
    OUTPUT_DIR = "你的路径"

    # 导出左脑彩色PLY
    export_colored_ply(
        surf_path=f"{SUBJECTS_DIR}/surf/lh.pial",
        annot_path=f"{SUBJECTS_DIR}/label/lh.aparc.annot",
        output_path=f"{OUTPUT_DIR}/左脑_彩色.ply"
    )

    # 导出右脑彩色PLY
    export_colored_ply(
        surf_path=f"{SUBJECTS_DIR}/surf/rh.pial",
        annot_path=f"{SUBJECTS_DIR}/label/rh.aparc.annot",
        output_path=f"{OUTPUT_DIR}/右脑_彩色.ply"
    )

    print("\n🎨 所有3D彩色模型导出完成！")