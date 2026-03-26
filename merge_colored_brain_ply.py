"""
合并两个彩色 PLY 文件（左脑 + 右脑）为一个文件。

用法:
    python merge_ply.py

如需修改输入/输出路径，请编辑下方的 file1, file2, output 变量。
"""

import struct
import sys
import os


def parse_ply_header(filepath):
    """解析 PLY 文件头，返回 (header_lines, vertex_count, face_count, is_binary, binary_format, properties)"""
    properties = []
    vertex_count = 0
    face_count = 0
    is_binary = False
    binary_format = None
    header_lines = []

    with open(filepath, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"未找到 end_header: {filepath}")
            line_str = line.decode("ascii", errors="replace").strip()
            header_lines.append(line_str)

            if line_str.startswith("format"):
                parts = line_str.split()
                if "binary_little_endian" in line_str:
                    is_binary = True
                    binary_format = "little"
                elif "binary_big_endian" in line_str:
                    is_binary = True
                    binary_format = "big"
                # else: ascii

            elif line_str.startswith("element vertex"):
                vertex_count = int(line_str.split()[-1])
            elif line_str.startswith("element face"):
                face_count = int(line_str.split()[-1])
            elif line_str.startswith("property") and face_count == 0:
                # vertex properties (before face element)
                properties.append(line_str)

            if line_str == "end_header":
                header_end_offset = f.tell()
                break

    return header_lines, vertex_count, face_count, is_binary, binary_format, properties, header_end_offset


# PLY type -> (struct format char, byte size)
PLY_TYPE_MAP = {
    "float": ("f", 4),
    "float32": ("f", 4),
    "double": ("d", 8),
    "float64": ("d", 8),
    "int": ("i", 4),
    "int32": ("i", 4),
    "uint": ("I", 4),
    "uint32": ("I", 4),
    "short": ("h", 2),
    "int16": ("h", 2),
    "ushort": ("H", 2),
    "uint16": ("H", 2),
    "char": ("b", 1),
    "int8": ("b", 1),
    "uchar": ("B", 1),
    "uint8": ("B", 1),
}


def get_vertex_struct(properties, endian="little"):
    """根据属性列表构建 struct 格式"""
    prefix = "<" if endian == "little" else ">"
    fmt = prefix
    for prop in properties:
        parts = prop.split()
        # "property <type> <name>"
        ptype = parts[1]
        if ptype in PLY_TYPE_MAP:
            fmt += PLY_TYPE_MAP[ptype][0]
        else:
            raise ValueError(f"不支持的属性类型: {ptype}")
    return struct.Struct(fmt)


def read_ply(filepath):
    """读取 PLY 文件，返回 (vertices_data_bytes, face_lines_or_bytes, header_info)"""
    header_lines, vcount, fcount, is_binary, binary_format, properties, header_end = \
        parse_ply_header(filepath)

    print(f"  文件: {os.path.basename(filepath)}")
    print(f"  顶点数: {vcount}, 面数: {fcount}, 格式: {'binary' if is_binary else 'ascii'}")
    print(f"  属性: {len(properties)} 个")

    vertices = []
    faces = []

    if is_binary:
        vstruct = get_vertex_struct(properties, binary_format)
        with open(filepath, "rb") as f:
            f.seek(header_end)
            for _ in range(vcount):
                data = f.read(vstruct.size)
                vertices.append(vstruct.unpack(data))
            # 读取面数据（list property）
            for _ in range(fcount):
                # 通常是 uchar + N*int
                n_bytes = f.read(1)
                n = struct.unpack("B", n_bytes)[0]
                idx_fmt = "<" + "i" * n if binary_format == "little" else ">" + "i" * n
                idx_data = f.read(4 * n)
                indices = struct.unpack(idx_fmt, idx_data)
                faces.append(indices)
    else:
        with open(filepath, "r", errors="replace") as f:
            # 跳过 header
            for line in f:
                if line.strip() == "end_header":
                    break
            for _ in range(vcount):
                line = f.readline().strip()
                vertices.append(line)
            for _ in range(fcount):
                line = f.readline().strip()
                faces.append(line)

    return vertices, faces, vcount, fcount, is_binary, binary_format, properties


def merge_and_write(file1, file2, output):
    print(f"\n读取文件 1...")
    v1, f1, vc1, fc1, is_bin1, fmt1, props1 = read_ply(file1)

    print(f"\n读取文件 2...")
    v2, f2, vc2, fc2, is_bin2, fmt2, props2 = read_ply(file2)

    # 检查属性是否一致
    if props1 != props2:
        print("\n⚠️  警告: 两个文件的顶点属性不完全一致，将使用第一个文件的属性格式。")
        print(f"  文件1属性: {props1}")
        print(f"  文件2属性: {props2}")

    total_vertices = vc1 + vc2
    total_faces = fc1 + fc2
    vertex_offset = vc1  # 第二个文件的面索引需要偏移

    print(f"\n合并中...")
    print(f"  总顶点数: {total_vertices}")
    print(f"  总面数: {total_faces}")

    # 统一用 ASCII 格式输出（兼容性最好）
    with open(output, "w") as out:
        # 写 header
        out.write("ply\n")
        out.write("format ascii 1.0\n")
        out.write(f"element vertex {total_vertices}\n")
        for prop in props1:
            out.write(prop + "\n")
        if total_faces > 0:
            out.write(f"element face {total_faces}\n")
            out.write("property list uchar int vertex_indices\n")
        out.write("end_header\n")

        # 写顶点
        for v in v1:
            if isinstance(v, str):
                out.write(v + "\n")
            else:
                out.write(" ".join(str(x) for x in v) + "\n")
        for v in v2:
            if isinstance(v, str):
                out.write(v + "\n")
            else:
                out.write(" ".join(str(x) for x in v) + "\n")

        # 写面（偏移第二个文件的索引）
        for face in f1:
            if isinstance(face, str):
                out.write(face + "\n")
            else:
                out.write(f"{len(face)} " + " ".join(str(i) for i in face) + "\n")

        for face in f2:
            if isinstance(face, str):
                parts = face.split()
                n = int(parts[0])
                indices = [str(int(parts[i + 1]) + vertex_offset) for i in range(n)]
                out.write(f"{n} " + " ".join(indices) + "\n")
            else:
                shifted = [i + vertex_offset for i in face]
                out.write(f"{len(shifted)} " + " ".join(str(i) for i in shifted) + "\n")

    print(f"\n✅ 合并完成! 输出文件: {output}")


if __name__ == "__main__":
    # ===== 修改这里的路径 =====
    file1 = r"你的文件左脑_彩色.ply"
    file2 = r"你的文件右脑_彩色.ply"
    output = r"你的文件合并_大脑_彩色.ply"
    # ==========================

    if not os.path.exists(file1):
        print(f"❌ 文件不存在: {file1}")
        sys.exit(1)
    if not os.path.exists(file2):
        print(f"❌ 文件不存在: {file2}")
        sys.exit(1)

    merge_and_write(file1, file2, output)