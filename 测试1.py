# -*- coding: utf-8 -*-
# @Time    : 2026/2/17 19:24
# @Author  : cy1026
# @File    : 测试1.py
# @Software: PyCharm

import cv2
import numpy as np
import os
import json
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk

# ==========================================
# 核心逻辑类 (Core Logic)
# ==========================================
class ImageProcessor:
    """
    负责图像处理的核心算法，与UI解耦。
    """
    @staticmethod
    def cv_imread(file_path):
        try:
            # 支持中文路径读取
            return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None

    @staticmethod
    def cv_imwrite(file_path, img):
        try:
            # 支持中文路径保存
            is_success, im_buf = cv2.imencode(".png", img)
            if is_success:
                im_buf.tofile(file_path)
                return True
        except Exception as e:
            print(f"Error writing file {file_path}: {e}")
        return False

    @staticmethod
    def get_mask_color(img, hue_tol, sat_min, val_min):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, w, _ = img.shape
        # 取四个角作为背景色参考
        corners_hsv = [hsv[0,0], hsv[0, w-1], hsv[h-1, 0], hsv[h-1, w-1]]
        bg_hsv = np.median(corners_hsv, axis=0)
        bg_h = bg_hsv[0]

        min_h, max_h = bg_h - hue_tol, bg_h + hue_tol
        
        # 处理色相环绕 (0-180)
        if min_h < 0:
            lower1, upper1 = np.array([min_h + 180, sat_min, val_min]), np.array([179, 255, 255])
            lower2, upper2 = np.array([0, sat_min, val_min]), np.array([max_h, 255, 255])
            mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2))
        elif max_h > 179:
            lower1, upper1 = np.array([min_h, sat_min, val_min]), np.array([179, 255, 255])
            lower2, upper2 = np.array([0, sat_min, val_min]), np.array([max_h - 180, 255, 255])
            mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2))
        else:
            lower, upper = np.array([min_h, sat_min, val_min]), np.array([max_h, 255, 255])
            mask = cv2.inRange(hsv, lower, upper)

        return cv2.bitwise_not(mask) # 反转，保留前景

    @staticmethod
    def get_mask_yellow(img, h_center, h_tol, s_min, v_min):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        min_h, max_h = h_center - h_tol, h_center + h_tol

        if min_h < 0:
            lower1, upper1 = np.array([min_h + 180, s_min, v_min]), np.array([179, 255, 255])
            lower2, upper2 = np.array([0, s_min, v_min]), np.array([max_h, 255, 255])
            mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2))
        elif max_h > 179:
            lower1, upper1 = np.array([min_h, s_min, v_min]), np.array([179, 255, 255])
            lower2, upper2 = np.array([0, s_min, v_min]), np.array([max_h - 180, 255, 255])
            mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2))
        else:
            lower, upper = np.array([min_h, s_min, v_min]), np.array([max_h, 255, 255])
            mask = cv2.inRange(hsv, lower, upper)
        
        return mask

    @staticmethod
    def get_mask_gray(img, thresh_val, bg_type):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        type_flag = cv2.THRESH_BINARY if bg_type == "black" else cv2.THRESH_BINARY_INV
        
        if thresh_val == 0:
            type_flag += cv2.THRESH_OTSU
        
        _, mask = cv2.threshold(blurred, thresh_val, 255, type_flag)
        return mask

    @staticmethod
    def apply_morphology(mask, clean_k, connect_k, iters):
        if clean_k > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (clean_k, clean_k)))
        if connect_k > 0 and iters > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (connect_k, connect_k)), iterations=iters)
        return mask

# ==========================================
# 用户界面类 (UI)
# ==========================================
class ImageCutterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI图片处理工厂 - 交互式切分工具")
        self.root.geometry("1400x900")

        # 配置文件路径
        self.config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app_config.json")

        # 初始化变量
        self.input_path = ""
        self.output_path = ""
        self.files = []
        self.current_index = 0
        self.raw_image = None
        self.current_image = None
        self.current_filename = ""
        self.processed_mask = None
        self.display_image = None
        
        # 标志位：是否正在进行自动检测
        self.is_auto_detecting = False

        # 手动修补层 (Manual Mask Layers)
        self.manual_draw_layer = None  # 记录画笔 (Positive)
        self.manual_erase_layer = None # 记录橡皮 (Negative)

        # 编辑状态
        self.is_editing_mask = False
        self.drawing = False
        self.brush_size = 20
        self.edit_mode = "draw" # "draw" or "erase"
        self.last_mouse_pos = None
        self.brush_cursor_id = None

        # 缩放和平移状态
        self.zoom_scale = 1.0
        self.pan_offset_x = 0
        self.pan_offset_y = 0

        # 构建界面
        self.setup_ui()
        
        # 启动时尝试加载配置，如果失败则引导用户
        self.root.after(100, self.init_directories)

    def init_directories(self):
        """初始化目录：优先加载配置，否则询问用户"""
        if self.load_settings():
            self.refresh_file_list()
            print(f"已自动加载配置：输入={self.input_path}, 输出={self.output_path}")
        else:
            self.ask_directories()

    def load_settings(self):
        """读取配置文件"""
        if not os.path.exists(self.config_file):
            return False
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                input_path = config.get("input_path", "")
                output_path = config.get("output_path", "")
                
                # 验证路径是否存在
                if input_path and os.path.exists(input_path) and output_path and os.path.exists(output_path):
                    self.input_path = input_path
                    self.output_path = output_path
                    return True
        except Exception as e:
            print(f"读取配置文件失败: {e}")
        
        return False

    def save_settings(self):
        """保存配置文件"""
        config = {
            "input_path": self.input_path,
            "output_path": self.output_path
        }
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
            print("配置已保存")
        except Exception as e:
            print(f"保存配置文件失败: {e}")

    def ask_directories(self):
        """引导用户选择输入和输出目录"""
        # 1. 选择输入目录
        messagebox.showinfo("欢迎", "欢迎使用图片切分工具！\n\n第一步：请选择包含图片的【输入文件夹】。")
        in_dir = filedialog.askdirectory(title="选择输入图片文件夹")
        if not in_dir:
            if messagebox.askretrycancel("提示", "未选择输入文件夹，程序无法运行。\n是否重试？"):
                self.ask_directories()
                return
            else:
                self.root.destroy()
                return
        
        self.input_path = in_dir

        # 2. 选择输出目录
        messagebox.showinfo("下一步", "第二步：请选择切分后图片的【保存文件夹】。\n(如果文件夹不存在，您可以新建一个)")
        out_dir = filedialog.askdirectory(title="选择保存结果的文件夹")
        if not out_dir:
            # 如果用户取消，默认在输入目录下创建 output
            out_dir = os.path.join(self.input_path, "output_crops")
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            messagebox.showinfo("提示", f"您未选择保存路径，已默认设置为：\n{out_dir}")
        
        self.output_path = out_dir
        
        # 保存配置
        self.save_settings()
        
        # 加载文件列表
        self.refresh_file_list()

    def refresh_file_list(self):
        if not self.input_path: return
        
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        self.files = [f for f in os.listdir(self.input_path) if f.lower().endswith(valid_exts)]
        
        if not self.files:
            messagebox.showwarning("警告", f"在文件夹:\n{self.input_path}\n中没有找到图片文件！")
            return

        self.current_index = 0
        # 加载第一张图片，并强制执行一次自动检测
        self.load_image(0, force_auto_detect=True)

    def change_input_directory(self):
        directory = filedialog.askdirectory(title="更改输入图片文件夹")
        if directory:
            self.input_path = directory
            self.save_settings() # 保存更改
            self.refresh_file_list()

    def change_output_directory(self):
        directory = filedialog.askdirectory(title="更改保存文件夹")
        if directory:
            self.output_path = directory
            self.save_settings() # 保存更改
            messagebox.showinfo("提示", f"保存路径已更新为：\n{self.output_path}")

    def setup_ui(self):
        # 顶部菜单栏
        top_bar = ttk.Frame(self.root, padding="5")
        top_bar.pack(fill=tk.X)
        
        ttk.Button(top_bar, text="📂 更改输入文件夹", command=self.change_input_directory).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_bar, text="💾 更改保存文件夹", command=self.change_output_directory).pack(side=tk.LEFT, padx=5)
        ttk.Label(top_bar, text=" | ").pack(side=tk.LEFT)
        self.status_label = ttk.Label(top_bar, text="准备就绪", foreground="gray")
        self.status_label.pack(side=tk.LEFT, padx=5)

        # 主分割窗口
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True)

        # 左侧控制面板
        self.control_frame = ttk.Frame(main_paned, padding="10", width=320)
        main_paned.add(self.control_frame, weight=0)

        # --- 主控制区 ---
        self.main_controls_frame = ttk.Frame(self.control_frame)
        self.main_controls_frame.pack(fill=tk.X)

        ttk.Label(self.main_controls_frame, text="参数设置", font=("Arial", 14, "bold")).pack(pady=10)

        # 自动应用参数复选框 (默认关闭！)
        self.auto_apply_var = tk.BooleanVar(value=False)
        auto_apply_check = ttk.Checkbutton(self.main_controls_frame, text="自动检测并应用参数", variable=self.auto_apply_var)
        auto_apply_check.pack(anchor=tk.W, pady=(0, 5))

        # 保留手动修补复选框
        self.keep_manual_mask_var = tk.BooleanVar(value=False)
        keep_manual_check = ttk.Checkbutton(self.main_controls_frame, text="保留手动修补 (画笔/橡皮)", variable=self.keep_manual_mask_var)
        keep_manual_check.pack(anchor=tk.W, pady=(0, 10))

        # 模式选择
        self.mode_var = tk.StringVar(value="color")
        ttk.Label(self.main_controls_frame, text="处理模式:").pack(anchor=tk.W, pady=(5, 0))

        mode_frame = ttk.Frame(self.main_controls_frame)
        mode_frame.pack(fill=tk.X)
        ttk.Radiobutton(mode_frame, text="彩色背景", variable=self.mode_var, value="color", command=self.on_mode_change).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(mode_frame, text="黑白背景", variable=self.mode_var, value="gray", command=self.on_mode_change).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(mode_frame, text="特定黄", variable=self.mode_var, value="yellow", command=self.on_mode_change).pack(side=tk.LEFT, padx=2)

        # 滑块存储
        self.sliders = {}

        # 参数容器
        self.param_container = ttk.Frame(self.main_controls_frame)
        self.param_container.pack(fill=tk.X, pady=10)

        # 彩色模式参数
        self.color_frame = ttk.LabelFrame(self.param_container, text="彩色模式参数 (HSV)", padding="5")
        self.add_slider(self.color_frame, "hue_tol", "色相容差 (Hue)", 0, 90, 15)
        self.add_slider(self.color_frame, "sat_min", "最小饱和度 (Sat)", 0, 255, 40)
        self.add_slider(self.color_frame, "val_min", "最小亮度 (Val)", 0, 255, 40)

        # 黑白模式参数
        self.gray_frame = ttk.LabelFrame(self.param_container, text="黑白模式参数", padding="5")
        self.add_slider(self.gray_frame, "gray_thresh", "灰度阈值 (0=Auto)", 0, 255, 0)
        bg_type_frame = ttk.Frame(self.gray_frame)
        bg_type_frame.pack(fill=tk.X)
        self.bg_type_var = tk.StringVar(value="black")
        ttk.Radiobutton(bg_type_frame, text="黑背景 (物体亮)", variable=self.bg_type_var, value="black", command=self.update_preview).pack(side=tk.LEFT)
        ttk.Radiobutton(bg_type_frame, text="白背景 (物体暗)", variable=self.bg_type_var, value="white", command=self.update_preview).pack(side=tk.LEFT)

        # 黄色模式参数
        self.yellow_frame = ttk.LabelFrame(self.param_container, text="白底黄主体参数", padding="5")
        self.add_slider(self.yellow_frame, "yellow_h_center", "黄色中心 (Hue)", 0, 180, 30)
        self.add_slider(self.yellow_frame, "yellow_h_tol", "色相容差", 0, 90, 15)
        self.add_slider(self.yellow_frame, "yellow_s_min", "最小饱和度", 0, 255, 40)
        self.add_slider(self.yellow_frame, "yellow_v_min", "最小亮度", 0, 255, 40)

        # 形态学参数
        self.morph_frame = ttk.LabelFrame(self.main_controls_frame, text="形态学处理 (连接主体)", padding="5")
        self.morph_frame.pack(fill=tk.X, pady=10)
        self.add_slider(self.morph_frame, "clean_kernel", "去噪核大小 (Open)", 0, 10, 3)
        self.add_slider(self.morph_frame, "connect_kernel", "连接核大小 (Close)", 0, 20, 5)
        self.add_slider(self.morph_frame, "connect_iters", "连接迭代次数", 0, 10, 2)

        # 操作按钮
        action_frame = ttk.Frame(self.main_controls_frame)
        action_frame.pack(fill=tk.X, pady=10)
        self.edit_mask_button = ttk.Button(action_frame, text="✏️ 编辑蒙版 (E)", command=self.toggle_mask_editing)
        self.edit_mask_button.pack(fill=tk.X, pady=(0, 5))
        
        save_btn = ttk.Button(action_frame, text="✂️ 执行拆分并保存 (S)", command=self.save_crops, style="Accent.TButton")
        save_btn.pack(fill=tk.X)

        # 导航按钮
        nav_frame = ttk.Frame(self.main_controls_frame)
        nav_frame.pack(fill=tk.X, pady=20)
        ttk.Button(nav_frame, text="<< 上一张 (P)", command=self.prev_image).pack(side=tk.LEFT, padx=5, expand=True)
        ttk.Button(nav_frame, text="下一张 (N)", command=self.next_image).pack(side=tk.LEFT, padx=5, expand=True)

        # 信息标签
        self.info_label = ttk.Label(self.main_controls_frame, text="", foreground="blue", wraplength=280)
        self.info_label.pack(pady=5)

        # --- 蒙版编辑控制区 (初始隐藏) ---
        self.edit_controls_frame = ttk.Frame(self.control_frame)

        ttk.Label(self.edit_controls_frame, text="蒙版编辑模式", font=("Arial", 14, "bold")).pack(pady=10)
        
        edit_mode_frame = ttk.Frame(self.edit_controls_frame)
        edit_mode_frame.pack(fill=tk.X, pady=5)
        self.edit_mode_var = tk.StringVar(value="draw")
        ttk.Radiobutton(edit_mode_frame, text="画笔 (B)", variable=self.edit_mode_var, value="draw", command=self._set_edit_mode).pack(side=tk.LEFT)
        ttk.Radiobutton(edit_mode_frame, text="橡皮 (E)", variable=self.edit_mode_var, value="erase", command=self._set_edit_mode).pack(side=tk.LEFT, padx=10)

        ttk.Label(self.edit_controls_frame, text="笔刷大小:").pack(anchor=tk.W, pady=(10,0))
        self.size_slider = ttk.Scale(self.edit_controls_frame, from_=1, to=100, value=self.brush_size, orient=tk.HORIZONTAL, command=self._update_brush_size)
        self.size_slider.pack(fill=tk.X)

        ttk.Button(self.edit_controls_frame, text="✅ 完成编辑 (M)", command=self.toggle_mask_editing, style="Accent.TButton").pack(fill=tk.X, pady=20)

        # 右侧预览区
        preview_frame = ttk.Frame(main_paned, padding="10")
        main_paned.add(preview_frame, weight=1)

        self.canvas = tk.Canvas(preview_frame, bg="#2b2b2b")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 绑定事件
        self.canvas.bind("<Configure>", lambda e: self.update_display())
        self.root.bind("<s>", lambda e: self.save_crops())
        self.root.bind("<m>", lambda e: self.toggle_mask_editing())
        self.root.bind("<p>", lambda e: self.prev_image())
        self.root.bind("<n>", lambda e: self.next_image())
        self.root.bind("<b>", lambda e: self.edit_mode_var.set("draw") or self._set_edit_mode())
        self.root.bind("<e>", lambda e: self.edit_mode_var.set("erase") or self._set_edit_mode())
        self.root.bind("[", self.decrease_brush)
        self.root.bind("]", self.increase_brush)

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonPress-1>", self.start_paint)
        self.canvas.bind("<ButtonRelease-1>", self.stop_paint)
        self.canvas.bind("<Motion>", self.update_brush_cursor)
        self.canvas.bind("<Leave>", self.hide_brush_cursor)

        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        # FIX: 绑定方向键为平移
        self.root.bind("<Up>", lambda e: self.pan_image(0, 50))
        self.root.bind("<Down>", lambda e: self.pan_image(0, -50))
        self.root.bind("<Left>", lambda e: self.pan_image(50, 0))
        self.root.bind("<Right>", lambda e: self.pan_image(-50, 0))
        
        # 初始化显示
        self.on_mode_change()

    # --- 逻辑方法 ---

    def toggle_mask_editing(self):
        self.is_editing_mask = not self.is_editing_mask
        if self.is_editing_mask:
            self.main_controls_frame.pack_forget()
            self.edit_controls_frame.pack(fill=tk.X)
            self.edit_mask_button.config(text="完成编辑 (M)")
            self.canvas.config(cursor="none")
            self.update_display()
        else:
            self.edit_controls_frame.pack_forget()
            self.main_controls_frame.pack(fill=tk.X)
            self.edit_mask_button.config(text="编辑蒙版 (M)")
            self.canvas.config(cursor="")
            self.hide_brush_cursor()
            self.update_display()

    def _set_edit_mode(self):
        self.edit_mode = self.edit_mode_var.get()

    def _update_brush_size(self, val):
        self.brush_size = int(float(val))
        self.update_brush_cursor(self.last_mouse_pos)

    def decrease_brush(self, event=None):
        if not self.is_editing_mask: return
        new_size = max(1, self.brush_size - 5)
        self.size_slider.set(new_size)
        self._update_brush_size(new_size)

    def increase_brush(self, event=None):
        if not self.is_editing_mask: return
        new_size = min(100, self.brush_size + 5)
        self.size_slider.set(new_size)
        self._update_brush_size(new_size)

    def start_paint(self, event):
        if not self.is_editing_mask: return
        self.drawing = True
        self.paint(event)

    def stop_paint(self, event):
        if not self.is_editing_mask: return
        self.drawing = False

    def paint(self, event):
        if not self.drawing or not self.is_editing_mask: return
        if self.manual_draw_layer is None or self.manual_erase_layer is None: return

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        h, w = self.current_image.shape[:2]

        base_scale = min(canvas_w / w, canvas_h / h)
        current_scale = base_scale * self.zoom_scale

        new_w, new_h = int(w * current_scale), int(h * current_scale)
        base_offset_x = (canvas_w - new_w) // 2
        base_offset_y = (canvas_h - new_h) // 2
        total_offset_x = base_offset_x + self.pan_offset_x
        total_offset_y = base_offset_y + self.pan_offset_y

        x = int((event.x - total_offset_x) / current_scale)
        y = int((event.y - total_offset_y) / current_scale)

        real_brush_radius = int((self.brush_size / 2) / current_scale)
        if real_brush_radius < 1: real_brush_radius = 1

        # 根据模式在不同的层上绘制
        if self.edit_mode == "draw":
            cv2.circle(self.manual_draw_layer, (x, y), real_brush_radius, 255, -1)
            # 如果在画笔层画了，橡皮层对应位置应该清除，避免冲突
            cv2.circle(self.manual_erase_layer, (x, y), real_brush_radius, 0, -1)
        else:
            cv2.circle(self.manual_erase_layer, (x, y), real_brush_radius, 255, -1)
            # 如果在橡皮层画了，画笔层对应位置应该清除
            cv2.circle(self.manual_draw_layer, (x, y), real_brush_radius, 0, -1)

        self.update_preview()
        # FIX 3: 在绘画后立即更新光标位置
        self.update_brush_cursor(event)

    def update_brush_cursor(self, event):
        self.last_mouse_pos = event
        if not self.is_editing_mask or event is None: return

        if self.brush_cursor_id:
            self.canvas.delete(self.brush_cursor_id)

        x, y = event.x, event.y
        r = self.brush_size / 2
        color = "white" if self.edit_mode == "draw" else "black"
        self.brush_cursor_id = self.canvas.create_oval(x-r, y-r, x+r, y+r, fill=color, outline="gray")

    def hide_brush_cursor(self, event=None):
        if self.brush_cursor_id:
            self.canvas.delete(self.brush_cursor_id)
            self.brush_cursor_id = None

    def on_mouse_wheel(self, event):
        if event.delta > 0:
            self.zoom_scale *= 1.1
        else:
            self.zoom_scale /= 1.1
            if self.zoom_scale < 0.1: self.zoom_scale = 0.1

        self.update_display()
        if self.is_editing_mask:
            self.update_brush_cursor(self.last_mouse_pos)

    def pan_image(self, dx, dy):
        self.pan_offset_x += dx
        self.pan_offset_y += dy
        self.update_display()
        if self.is_editing_mask:
            self.update_brush_cursor(self.last_mouse_pos)

    def on_mode_change(self):
        # 如果不是程序自动检测触发的，说明是用户手动点击了模式
        if not self.is_auto_detecting:
            self.auto_apply_var.set(False)

        mode = self.mode_var.get()
        self.color_frame.pack_forget()
        self.gray_frame.pack_forget()
        self.yellow_frame.pack_forget()

        if mode == "color":
            self.color_frame.pack(fill=tk.X, expand=True)
        elif mode == "gray":
            self.gray_frame.pack(fill=tk.X, expand=True)
        elif mode == "yellow":
            self.yellow_frame.pack(fill=tk.X, expand=True)

        self.update_preview()

    def add_slider(self, parent, name, label, min_val, max_val, default):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)
        lbl = ttk.Label(frame, text=f"{label}: {default}")
        lbl.pack(anchor=tk.W)
        var = tk.IntVar(value=default)
        def on_change(val):
            # 如果不是程序自动检测触发的，说明是用户手动拖动了滑块
            if not self.is_auto_detecting:
                self.auto_apply_var.set(False)
                
            lbl.config(text=f"{label}: {int(float(val))}")
            if not self.is_editing_mask:
                self.update_preview()
        slider = ttk.Scale(frame, from_=min_val, to=max_val, variable=var, orient=tk.HORIZONTAL, command=on_change)
        slider.pack(fill=tk.X)
        self.sliders[name] = var

    def load_image(self, index, force_auto_detect=False):
        if not self.files: return

        if self.is_editing_mask:
            self.toggle_mask_editing()

        if index < 0: index = 0
        if index >= len(self.files): index = len(self.files) - 1

        self.current_index = index
        self.current_filename = self.files[index]
        file_path = os.path.join(self.input_path, self.current_filename)

        img = ImageProcessor.cv_imread(file_path)
        if img is None:
            messagebox.showerror("错误", f"无法读取图片: {self.current_filename}")
            return

        self.raw_image = img

        if len(img.shape) == 3 and img.shape[2] == 4:
            self.current_image = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif len(img.shape) == 2:
            self.current_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            self.current_image = img

        self.root.title(f"AI图片处理工厂 - {self.current_filename} ({index+1}/{len(self.files)})")
        self.status_label.config(text=f"当前文件: {self.current_filename}")

        self.zoom_scale = 1.0
        self.pan_offset_x = 0
        self.pan_offset_y = 0

        # 初始化或调整手动修补层
        h, w = self.current_image.shape[:2]
        
        should_reset_manual = True
        if self.keep_manual_mask_var.get() and self.manual_draw_layer is not None:
            # 如果勾选了保留，且之前有层，则尝试调整大小保留
            try:
                if self.manual_draw_layer.shape != (h, w):
                    self.manual_draw_layer = cv2.resize(self.manual_draw_layer, (w, h), interpolation=cv2.INTER_NEAREST)
                    self.manual_erase_layer = cv2.resize(self.manual_erase_layer, (w, h), interpolation=cv2.INTER_NEAREST)
                should_reset_manual = False
            except:
                should_reset_manual = True
        
        if should_reset_manual:
            self.manual_draw_layer = np.zeros((h, w), dtype=np.uint8)
            self.manual_erase_layer = np.zeros((h, w), dtype=np.uint8)

        # 核心逻辑修改：
        # 1. 如果是强制自动检测（比如刚打开文件夹），则执行检测。
        # 2. 如果复选框被勾选，则执行检测。
        # 3. 否则，沿用参数。
        if force_auto_detect or self.auto_apply_var.get():
            self.auto_detect_params()
        else:
            self.update_preview() # 使用现有参数更新预览

    def auto_detect_params(self):
        # 开启标志位，防止触发用户手动修改的逻辑
        self.is_auto_detecting = True
        
        img = self.current_image
        h, w, _ = img.shape
        corners = np.array([img[0,0], img[0, w-1], img[h-1, 0], img[h-1, w-1]])
        bg_color = np.median(corners, axis=0)
        mean_val = np.mean(bg_color)

        if mean_val < 50:
            self.mode_var.set("gray")
            self.bg_type_var.set("black")
            self.sliders["gray_thresh"].set(0)
        elif mean_val > 200:
            self.mode_var.set("gray")
            self.bg_type_var.set("white")
            self.sliders["gray_thresh"].set(0)
        else:
            self.mode_var.set("color")
            self.sliders["hue_tol"].set(15)
            self.sliders["sat_min"].set(40)
            self.sliders["val_min"].set(40)
        
        self.on_mode_change()
        
        # 关闭标志位
        self.is_auto_detecting = False

    def get_mask(self):
        if self.current_image is None: return None
        img = self.current_image
        mode = self.mode_var.get()
        mask = None

        # 1. 计算算法蒙版
        if mode == "color":
            mask = ImageProcessor.get_mask_color(img, self.sliders["hue_tol"].get(), self.sliders["sat_min"].get(), self.sliders["val_min"].get())
        elif mode == "yellow":
            mask = ImageProcessor.get_mask_yellow(img, self.sliders["yellow_h_center"].get(), self.sliders["yellow_h_tol"].get(), self.sliders["yellow_s_min"].get(), self.sliders["yellow_v_min"].get())
        else:
            mask = ImageProcessor.get_mask_gray(img, self.sliders["gray_thresh"].get(), self.bg_type_var.get())

        mask = ImageProcessor.apply_morphology(mask, self.sliders["clean_kernel"].get(), self.sliders["connect_kernel"].get(), self.sliders["connect_iters"].get())

        # 2. 叠加手动修补层
        if self.manual_draw_layer is not None and self.manual_erase_layer is not None:
            # 添加画笔内容 (OR)
            mask = cv2.bitwise_or(mask, self.manual_draw_layer)
            # 移除橡皮内容 (AND NOT)
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(self.manual_erase_layer))

        return mask

    def update_preview(self):
        # FIX 1: 总是重新计算最终蒙版
        self.processed_mask = self.get_mask()
        self.update_display()

    def update_display(self):
        if self.current_image is None or self.processed_mask is None: return

        img_rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)

        if self.is_editing_mask:
            overlay = np.zeros_like(img_rgb)
            # 显示最终蒙版区域
            overlay[self.processed_mask == 255] = [255, 0, 0] 
            
            # FIX 2: 增加蒙版对比度，让红色更显眼
            final_vis = cv2.addWeighted(img_rgb, 0.4, overlay, 0.6, 0)
        else:
            mask_inv = cv2.bitwise_not(self.processed_mask)
            mask_3c = cv2.cvtColor(self.processed_mask, cv2.COLOR_GRAY2RGB)
            mask_inv_3c = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2RGB)
            fg = cv2.bitwise_and(img_rgb, mask_3c)
            bg = cv2.bitwise_and(img_rgb, mask_inv_3c)
            bg = (bg * 0.3).astype(np.uint8)
            bg[:, :, 0] = np.clip(bg[:, :, 0] + 50, 0, 255)
            final_vis = cv2.add(fg, bg)
            contours, _ = cv2.findContours(self.processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(final_vis, contours, -1, (0, 255, 0), 2)
            self.info_label.config(text=f"检测到 {len(contours)} 个对象")

        canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if canvas_w < 10 or canvas_h < 10: return

        h, w, _ = final_vis.shape
        base_scale = min(canvas_w / w, canvas_h / h)
        current_scale = base_scale * self.zoom_scale
        new_w, new_h = int(w * current_scale), int(h * current_scale)

        final_vis_resized = cv2.resize(final_vis, (new_w, new_h))
        img_pil = Image.fromarray(final_vis_resized)
        self.display_image = ImageTk.PhotoImage(img_pil)

        self.canvas.delete("all")
        x = (canvas_w - new_w) // 2 + self.pan_offset_x
        y = (canvas_h - new_h) // 2 + self.pan_offset_y
        self.canvas.create_image(x, y, anchor=tk.NW, image=self.display_image)

    def prev_image(self, event=None):
        if self.current_index > 0:
            self.load_image(self.current_index - 1)

    def next_image(self, event=None):
        if self.current_index < len(self.files) - 1:
            self.load_image(self.current_index + 1)
        else:
            messagebox.showinfo("完成", "已经是最后一张图片了。")

    def save_crops(self):
        if self.processed_mask is None or self.raw_image is None: return

        if len(self.raw_image.shape) == 3 and self.raw_image.shape[2] == 4:
            b, g, r, original_a = cv2.split(self.raw_image)
        else:
            bgr_image = cv2.cvtColor(self.raw_image, cv2.COLOR_GRAY2BGR) if len(self.raw_image.shape) == 2 else self.raw_image
            b, g, r = cv2.split(bgr_image)
            original_a = np.full(self.raw_image.shape[:2], 255, dtype=np.uint8)

        final_alpha = cv2.bitwise_and(original_a, self.processed_mask)
        final_full_image = cv2.merge([b, g, r, final_alpha])

        contours, _ = cv2.findContours(self.processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        base_name = os.path.splitext(self.current_filename)[0]
        count = 0

        if not contours:
            self.info_label.config(text="未检测到可保存的对象！")
            return

        for i, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 10 or h < 10: continue
            crop = final_full_image[y:y+h, x:x+w]
            save_name = f"{base_name}_{count}.png"
            save_path = os.path.join(self.output_path, save_name)
            
            if ImageProcessor.cv_imwrite(save_path, crop):
                count += 1

        self.info_label.config(text=f"已保存 {count} 个切片！")
        self.root.after(1000, self.next_image)

if __name__ == "__main__":
    root = tk.Tk()
    # 尝试加载主题，如果失败则忽略
    try:
        root.tk.call("source", "azure.tcl")
        root.tk.call("set_theme", "dark")
    except:
        pass

    app = ImageCutterApp(root)
    root.mainloop()
