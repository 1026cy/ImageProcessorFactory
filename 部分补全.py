# -*- coding: utf-8 -*-
# @Time    : 2026/2/18 13:24
# @Author  : cy1026
# @File    : 部分补全.py
# @Software: PyCharm

import tkinter as tk
from tkinter import filedialog, messagebox, Scale, HORIZONTAL
from PIL import Image, ImageTk, ImageDraw

class ImageRestorerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("图片补全工具 - 手动涂抹")
        self.root.geometry("1200x800")

        # 数据初始化
        self.target_image = None  # 镂空图 (底图)
        self.source_image = None  # 原图 (用于补全的素材)
        self.display_image = None # 用于显示的Tk图片对象
        self.mask_image = None    # 蒙版 (灰度图, 白色为选中区域)
        self.draw = None          # 蒙版绘图对象
        self.brush_size = 20      # 笔刷大小
        self.scale_ratio = 1.0    # 显示缩放比例
        self.image_pos = (0, 0)   # 图片在画布上的偏移量

        # 界面布局
        self.setup_ui()

        # 事件绑定
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.on_paint_release)
        self.root.bind("<Configure>", self.on_resize)

    def setup_ui(self):
        # 顶部控制栏
        self.controls_frame = tk.Frame(self.root, pady=10)
        self.controls_frame.pack(side=tk.TOP, fill=tk.X)

        btn_opts = {'side': tk.LEFT, 'padx': 10}

        self.btn_load_target = tk.Button(self.controls_frame, text="1. 加载镂空图", command=self.load_target, bg="#e1f5fe")
        self.btn_load_target.pack(**btn_opts)

        self.btn_load_source = tk.Button(self.controls_frame, text="2. 加载原图", command=self.load_source, bg="#e1f5fe")
        self.btn_load_source.pack(**btn_opts)

        tk.Label(self.controls_frame, text="笔刷大小:").pack(side=tk.LEFT, padx=(20, 5))
        self.slider_brush = Scale(self.controls_frame, from_=5, to=100, orient=HORIZONTAL, command=self.update_brush_size)
        self.slider_brush.set(self.brush_size)
        self.slider_brush.pack(side=tk.LEFT)

        self.btn_reset = tk.Button(self.controls_frame, text="重置涂抹", command=self.reset_mask, bg="#ffccbc")
        self.btn_reset.pack(**btn_opts)

        self.btn_save = tk.Button(self.controls_frame, text="保存结果", command=self.save_image, bg="#c8e6c9")
        self.btn_save.pack(**btn_opts)

        # 主画布区域
        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.canvas_frame, bg="#333333", cursor="circle")
        self.canvas.pack(fill=tk.BOTH, expand=True)

    def update_brush_size(self, val):
        self.brush_size = int(val)

    def load_target(self):
        path = filedialog.askopenfilename(title="选择镂空图片", filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.webp")])
        if path:
            try:
                self.target_image = Image.open(path).convert("RGBA")
                self.init_canvas()
                self.root.title(f"图片补全工具 - {path}")
            except Exception as e:
                messagebox.showerror("错误", f"无法加载图片: {e}")

    def load_source(self):
        path = filedialog.askopenfilename(title="选择原图", filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.webp")])
        if path:
            try:
                self.source_image = Image.open(path).convert("RGBA")
                if self.target_image:
                    # 如果尺寸不一致，调整原图大小以匹配镂空图
                    if self.source_image.size != self.target_image.size:
                        self.source_image = self.source_image.resize(self.target_image.size, Image.Resampling.LANCZOS)
                        messagebox.showinfo("提示", "原图尺寸已自动调整以匹配镂空图。")
                self.update_display()
            except Exception as e:
                messagebox.showerror("错误", f"无法加载图片: {e}")

    def init_canvas(self):
        if self.target_image is None:
            return
        
        # 创建一个全黑的蒙版 (L模式)
        self.mask_image = Image.new("L", self.target_image.size, 0)
        self.draw = ImageDraw.Draw(self.mask_image)
        self.update_display()

    def reset_mask(self):
        if self.target_image:
            self.mask_image = Image.new("L", self.target_image.size, 0)
            self.draw = ImageDraw.Draw(self.mask_image)
            self.canvas.delete("brush_stroke")
            self.update_display()

    def on_resize(self, event):
        # 窗口大小改变时重新绘制
        if self.target_image:
            self.update_display()

    def update_display(self):
        if self.target_image is None:
            return

        # 获取画布尺寸
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10: return 

        # 计算合成图像
        if self.source_image:
            # 使用蒙版合成：mask为白色(255)的地方使用source，黑色(0)的地方使用target
            display_content = Image.composite(self.source_image, self.target_image, self.mask_image)
        else:
            # 如果没有原图，显示镂空图 + 红色半透明涂抹痕迹
            # 创建一个红色的覆盖层
            red_overlay = Image.new("RGBA", self.target_image.size, (255, 0, 0, 100))
            # 仅在mask区域显示红色
            overlay = Image.new("RGBA", self.target_image.size, (0, 0, 0, 0))
            overlay.paste(red_overlay, (0, 0), self.mask_image)
            
            display_content = Image.alpha_composite(self.target_image, overlay)

        # 计算缩放比例以适应窗口
        iw, ih = display_content.size
        ratio = min(cw/iw, ch/ih)
        self.scale_ratio = ratio
        
        new_size = (int(iw * ratio), int(ih * ratio))
        resized_img = display_content.resize(new_size, Image.Resampling.LANCZOS)
        
        self.display_image = ImageTk.PhotoImage(resized_img)
        
        # 清除背景并居中显示
        self.canvas.delete("bg_image")
        x = (cw - new_size[0]) // 2
        y = (ch - new_size[1]) // 2
        self.image_pos = (x, y)
        
        self.canvas.create_image(x, y, anchor=tk.NW, image=self.display_image, tags="bg_image")
        self.canvas.tag_lower("bg_image") # 确保图片在笔刷痕迹下方

    def paint(self, event):
        if self.target_image is None or self.draw is None:
            return
        
        # 1. 在Canvas上绘制临时痕迹 (为了流畅性)
        r_screen = self.brush_size * self.scale_ratio / 2
        x, y = event.x, event.y
        
        # 颜色：如果有原图，用绿色表示"恢复中"，否则用红色表示"选区"
        color = 'green' if self.source_image else 'red'
        self.canvas.create_oval(x-r_screen, y-r_screen, x+r_screen, y+r_screen, 
                                fill=color, outline=color, tags="brush_stroke")
        
        # 2. 在蒙版Image上绘制 (逻辑数据)
        # 将屏幕坐标转换为图片坐标
        ix = (event.x - self.image_pos[0]) / self.scale_ratio
        iy = (event.y - self.image_pos[1]) / self.scale_ratio
        
        r_real = self.brush_size / 2
        self.draw.ellipse((ix-r_real, iy-r_real, ix+r_real, iy+r_real), fill=255, outline=255)

    def on_paint_release(self, event):
        # 鼠标松开时，清除临时痕迹，刷新整体显示
        self.canvas.delete("brush_stroke")
        self.update_display()

    def save_image(self):
        if self.target_image is None or self.source_image is None:
            messagebox.showwarning("提示", "请确保已加载镂空图和原图")
            return
            
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("JPG", "*.jpg")])
        if path:
            try:
                res = Image.composite(self.source_image, self.target_image, self.mask_image)
                res.save(path)
                messagebox.showinfo("成功", f"图片已保存至:\n{path}")
            except Exception as e:
                messagebox.showerror("错误", f"保存失败: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    # 尝试设置高DPI支持 (Windows)
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass
        
    app = ImageRestorerApp(root)
    root.mainloop()
