# -*- coding: utf-8 -*-
# @Time    : 2026/3/6 11:58
# @Author  : cy1026
# @File    : 1.py
# @Software: PyCharm
import pygame
import sys

# --- 初始化 ---
FILE_NAME = 'TilesetFloor.png'
TILE_SIZE = 16


def run_debug_tool():
    pygame.init()
    screen = pygame.display.set_mode((1200, 800), pygame.RESIZABLE)
    pygame.display.set_caption("像素对齐终极分析器 - 滚轮缩放，按住拖动")
    font = pygame.font.SysFont("SimHei", 24)

    try:
        img = pygame.image.load(FILE_NAME).convert_alpha()
    except:
        print("没找到图！")
        return

    # 状态变量
    offset_x, offset_y = 50, 50  # 图片渲染的初始位置
    zoom = 2.0  # 初始放大倍数
    dragging = False
    last_mouse_pos = (0, 0)

    clock = pygame.time.Clock()

    while True:
        screen.fill((20, 20, 20))

        # --- 计算当前缩放后的尺寸 ---
        curr_w = int(img.get_width() * zoom)
        curr_h = int(img.get_height() * zoom)
        scaled_img = pygame.transform.scale(img, (curr_w, curr_h))

        # --- 1. 绘制缩放后的图片 ---
        screen.blit(scaled_img, (offset_x, offset_y))

        # --- 2. 绘制 16x16 网格 (基于图片位置) ---
        # 我们要看看 16x16 的网格在缩放后是多少像素
        grid_step = TILE_SIZE * zoom

        # 绘制网格线
        rows = (img.get_height() // TILE_SIZE) + 1
        cols = (img.get_width() // TILE_SIZE) + 1

        for c in range(cols + 1):
            x = offset_x + c * grid_step
            pygame.draw.line(screen, (255, 0, 0, 100), (x, offset_y), (x, offset_y + rows * grid_step), 1)
        for r in range(rows + 1):
            y = offset_y + r * grid_step
            pygame.draw.line(screen, (0, 255, 0, 100), (offset_x, y), (offset_x + cols * grid_step, y), 1)

        # --- 3. UI 调试信息 ---
        # 这里计算“真实偏移”：即图片相对于网格起始点的偏移
        # 如果你发现图片需要向右挪 2 像素才对齐，那么真实 Margin 就是 2
        ui_bg = pygame.Surface((300, 120))
        ui_bg.set_alpha(180)
        ui_bg.fill((0, 0, 0))
        screen.blit(ui_bg, (10, 10))

        text_zoom = font.render(f"缩放倍数 (滚轮): {zoom:.1f}x", True, (255, 255, 255))
        text_off = font.render(f"当前位置: {offset_x}, {offset_y}", True, (255, 255, 255))
        text_hint = font.render("按住鼠标左键拖动图片", True, (0, 255, 0))

        screen.blit(text_zoom, (20, 20))
        screen.blit(text_off, (20, 50))
        screen.blit(text_hint, (20, 80))

        # --- 事件处理 ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit();
                sys.exit()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # 左键拖动
                    dragging = True
                    last_mouse_pos = event.pos
                elif event.button == 4:  # 滚轮上
                    zoom += 0.2
                elif event.button == 5:  # 滚轮下
                    zoom = max(0.2, zoom - 0.2)

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1: dragging = False

            elif event.type == pygame.MOUSEMOTION:
                if dragging:
                    dx = event.pos[0] - last_mouse_pos[0]
                    dy = event.pos[1] - last_mouse_pos[1]
                    offset_x += dx
                    offset_y += dy
                    last_mouse_pos = event.pos

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    run_debug_tool()