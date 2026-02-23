# -*- coding: utf-8 -*-
# @Time    : 2026/2/22 20:36
# @Author  : cy1026
# @File    : 1.py
# @Software: PyCharm

import os
import sys
import psutil

# 1. 强制加载 onnxruntime
try:
    from onnxruntime.capi import onnxruntime_pybind11_state
    print("--- 导入成功，开始扫描依赖 ---")
except Exception as e:
    print(f"导入失败: {e}")
    sys.exit()

# 2. 获取当前进程所有加载的 DLL (不仅仅是 Python 层的)
process = psutil.Process(os.getpid())
dll_list = []

print(f"{'DLL 名称':<40} | {'完整路径'}")
print("-" * 100)

for dll in process.memory_maps():
    p = dll.path
    if p.lower().endswith(('.dll', '.pyd')):
        # 记录下来，我们可以对比打包环境和这个列表的差异
        print(f"{os.path.basename(p):<40} | {p}")
        dll_list.append(p)

# 3. 将结果保存到文件，带到打包环境去对比
with open("dev_env_dlls.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(dll_list))

print(f"\n[完成] 依赖列表已保存至 dev_env_dlls.txt，共 {len(dll_list)} 个文件。")