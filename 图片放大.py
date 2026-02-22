# -*- coding: utf-8 -*-
# @Time    : 2026/2/22 11:29
# @Author  : cy1026
# @File    : 图片放大.py
# @Software: PyCharm
from PIL import Image


def scale_pixel_art(input_path, output_path, scale_factor):
    """
    专门针对像素画的无损放大
    """
    with Image.open(input_path) as img:
        # 获取原始尺寸
        width, height = img.size
        # 计算新尺寸
        new_size = (width * scale_factor, height * scale_factor)

        # 核心：使用 NEAREST 滤镜，确保像素点边界清晰，不产生模糊
        # 这才是像素游戏素材“高清化”的标准做法
        hd_img = img.resize(new_size, resample=Image.NEAREST)

        hd_img.save(output_path)
        print(f"像素无损放大完成：{width}x{height} -> {new_size[0]}x{new_size[1]}")


# 放大 8 倍示例
scale_pixel_art('Tileset Spring.png', 'Tileset_Pixel_HD.png', 8)