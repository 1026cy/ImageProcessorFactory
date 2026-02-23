# -*- coding: utf-8 -*-
# @Time    : 2026/2/22 23:41
# @Author  : cy1026
# @File    : 2.py.py
# @Software: PyCharm
import os
import shutil

# --- 配置区 ---
# 你保存的开发环境 DLL 名单路径
MANIFEST_FILE = "dev_env_dlls.txt"
# 打包后的目标根目录
DIST_ROOT = r"D:\Users\admin\PycharmProjects\ImageProcessorFactory\dist\1"
# ONNX 核心插件目录
ORT_CAPI_DIR = os.path.join(DIST_ROOT, "_internal", "onnxruntime", "capi")

# 必须确保存在的目录
os.makedirs(DIST_ROOT, exist_ok=True)
os.makedirs(ORT_CAPI_DIR, exist_ok=True)


def sync_dependencies():
    if not os.path.exists(MANIFEST_FILE):
        print(f"错误: 找不到 {MANIFEST_FILE}")
        return

    with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
        paths = [line.strip() for line in f.readlines() if line.strip()]

    print(f"开始同步依赖，共 {len(paths)} 项...")

    for src in paths:
        # 跳过名单中的源标记
        if src.startswith("[source"):
            continue

        if not os.path.exists(src):
            print(f"跳过 (文件不存在): {src}")
            continue

        filename = os.path.basename(src).lower()

        # 确定存放位置的逻辑
        # 1. 如果是 onnxruntime 相关文件，放入 capi 目录
        if "onnxruntime" in src.lower():
            target_path = os.path.join(ORT_CAPI_DIR, os.path.basename(src))
        # 2. 如果是核心运行时或图形库，放在根目录以备全局调用 [cite: 1, 2]
        elif any(x in filename for x in ["python3", "vcruntime", "msvcp140", "dxgi", "dbghelp"]):
            target_path = os.path.join(DIST_ROOT, os.path.basename(src))
        else:
            # 其他文件暂时略过，避免污染
            continue

        try:
            if not os.path.exists(target_path):
                shutil.copy2(src, target_path)
                print(f"已同步: {os.path.basename(src)} -> {os.path.dirname(target_path)}")
        except Exception as e:
            print(f"复制 {filename} 失败: {e}")

    print("\n同步完成！请重新运行 1.exe 测试。")


if __name__ == "__main__":
    sync_dependencies()