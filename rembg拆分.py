# -*- coding: utf-8 -*-
# @Time    : 2026/2/22 11:53
# @Author  : cy1026
# @File    : rembg拆分.py
# @Software: PyCharm
import os
import sys

# ==========================================
# 环境配置 (Environment Setup)
# ==========================================
if getattr(sys, 'frozen', False):
    base_path = os.path.dirname(sys.executable)
else:
    base_path = os.path.dirname(os.path.abspath(__file__))

models_dir = os.path.join(base_path, "models")
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

os.environ["U2NET_HOME"] = models_dir

import cv2
import numpy as np
import json
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
from rembg import remove, new_session
import threading
import queue


# ==========================================
# 核心逻辑类 (Core Logic)
# ==========================================
class ImageProcessor:
    """
    负责图像处理的核心算法，与UI解耦。
    """
    _sessions = {}

    @classmethod
    def get_session(cls, model_name):
        if model_name not in cls._sessions:
            print(f"正在加载模型: {model_name} ...")
            try:
                cls._sessions[model_name] = new_session(model_name)
            except Exception as e:
                print(f"模型加载失败: {e}")
                return None
        return cls._sessions[model_name]

    @staticmethod
    def cv_imread(file_path):
        try:
            return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None

    @staticmethod
    def cv_imwrite(file_path, img):
        try:
            is_success, im_buf = cv2.imencode(".png", img)
            if is_success:
                im_buf.tofile(file_path)
                return True
        except Exception as e:
            print(f"Error writing file {file_path}: {e}")
        return False

    @staticmethod
    def get_mask_rgba_range(raw_image, r_min, r_max, g_min, g_max, b_min, b_max, a_min, a_max, invert=True):
        # 确保图像是4通道BGRA
        if len(raw_image.shape) == 2:
            image_bgra = cv2.cvtColor(raw_image, cv2.COLOR_GRAY2BGRA)
        elif raw_image.shape[2] == 3:
            image_bgra = cv2.cvtColor(raw_image, cv2.COLOR_BGR2BGRA)
        else:
            image_bgra = raw_image

        # OpenCV 使用 BGR 顺序
        lower_bound = np.array([b_min, g_min, r_min, a_min], dtype=np.uint8)
        upper_bound = np.array([b_max, g_max, r_max, a_max], dtype=np.uint8)
        
        # 创建一个蒙版，其中在范围内的像素为白色（255）
        mask = cv2.inRange(image_bgra, lower_bound, upper_bound)
        
        # 统计匹配像素比例 (用于调试)
        match_ratio = np.count_nonzero(mask) / (mask.shape[0] * mask.shape[1])
        
        # 如果 invert=True (默认)，则反转蒙版
        if invert:
            return cv2.bitwise_not(mask), match_ratio
        else:
            return mask, match_ratio

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
    def get_mask_rembg(img, model_name="u2net", alpha_matting=False, am_fg_thresh=240, am_bg_thresh=10, am_erode=10):
        try:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            
            session = ImageProcessor.get_session(model_name)
            if session is None:
                return np.zeros(img.shape[:2], dtype=np.uint8)

            output = remove(
                pil_img,
                session=session,
                alpha_matting=alpha_matting,
                alpha_matting_foreground_threshold=am_fg_thresh,
                alpha_matting_background_threshold=am_bg_thresh,
                alpha_matting_erode_size=am_erode
            )
            
            output_np = np.array(output)
            if output_np.shape[2] == 4:
                return output_np[:, :, 3]
            return np.zeros(img.shape[:2], dtype=np.uint8)
        except Exception as e:
            print(f"Rembg error: {e}")
            return np.zeros(img.shape[:2], dtype=np.uint8)

    @staticmethod
    def shift_mask(mask, dx, dy):
        if dx == 0 and dy == 0:
            return mask
        h, w = mask.shape
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(mask, M, (w, h))

    @staticmethod
    def apply_morphology(mask, clean_k, connect_k, iters):
        if clean_k > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (clean_k, clean_k)))
        if connect_k > 0 and iters > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                    cv2.getStructuringElement(cv2.MORPH_RECT, (connect_k, connect_k)), iterations=iters)
        return mask


# ==========================================
# 用户界面类 (UI)
# ==========================================
class ImageCutterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI图片处理工厂 - 交互式切分工具")
        self.root.geometry("1400x900")

        self.config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app_config.json")

        self.input_path = ""
        self.output_path = ""
        self.files = []
        self.current_index = 0
        self.raw_image = None
        self.current_image = None
        self.current_filename = ""
        self.processed_mask = None
        self.display_image = None

        self.is_auto_detecting = False
        self.is_processing = False
        self.debounce_job = None
        self.is_picking_color = False

        self.manual_draw_layer = None
        self.manual_erase_layer = None

        self.is_editing_mask = False
        self.drawing = False
        self.brush_size = 20
        self.edit_mode = "draw"
        self.last_mouse_pos = None
        self.brush_cursor_id = None

        self.zoom_scale = 1.0
        self.pan_offset_x = 0
        self.pan_offset_y = 0

        self.processing_request_queue = queue.Queue(maxsize=1)
        self.processing_result_queue = queue.Queue()

        self.setup_ui()

        threading.Thread(target=self._processing_worker, daemon=True).start()
        self.root.after(100, self._check_result_queue)
        self.root.after(150, self.init_directories)

    def init_directories(self):
        if self.load_settings():
            self.refresh_file_list()
            print(f"已自动加载配置：输入={self.input_path}, 输出={self.output_path}")
        else:
            self.ask_directories()

    def load_settings(self):
        if not os.path.exists(self.config_file):
            return False
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                input_path = config.get("input_path", "")
                output_path = config.get("output_path", "")
                if input_path and os.path.exists(input_path) and output_path and os.path.exists(output_path):
                    self.input_path = input_path
                    self.output_path = output_path
                    return True
        except Exception as e:
            print(f"读取配置文件失败: {e}")
        return False

    def save_settings(self):
        config = {"input_path": self.input_path, "output_path": self.output_path}
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
            print("配置已保存")
        except Exception as e:
            print(f"保存配置文件失败: {e}")

    def ask_directories(self):
        messagebox.showinfo("欢迎", "欢迎使用图片切分工具！\n\n第一步：请选择包含图片的【输入文件夹】。")
        in_dir = filedialog.askdirectory(title="选择输入图片文件夹")
        if not in_dir:
            if messagebox.askretrycancel("提示", "未选择输入文件夹，程序无法运行。\n是否重试？"):
                self.ask_directories()
            else:
                self.root.destroy()
            return
        self.input_path = in_dir

        messagebox.showinfo("下一步", "第二步：请选择切分后图片的【保存文件夹】。")
        out_dir = filedialog.askdirectory(title="选择保存结果的文件夹")
        if not out_dir:
            out_dir = os.path.join(self.input_path, "output_crops")
            os.makedirs(out_dir, exist_ok=True)
            messagebox.showinfo("提示", f"您未选择保存路径，已默认设置为：\n{out_dir}")
        self.output_path = out_dir

        self.save_settings()
        self.refresh_file_list()

    def refresh_file_list(self):
        if not self.input_path: return
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        self.files = [f for f in os.listdir(self.input_path) if f.lower().endswith(valid_exts)]
        if not self.files:
            messagebox.showwarning("警告", f"在文件夹:\n{self.input_path}\n中没有找到图片文件！")
            return
        self.current_index = 0
        self.load_image(0, force_auto_detect=True)

    def change_input_directory(self):
        directory = filedialog.askdirectory(title="更改输入图片文件夹")
        if directory:
            self.input_path = directory
            self.save_settings()
            self.refresh_file_list()

    def change_output_directory(self):
        directory = filedialog.askdirectory(title="更改保存文件夹")
        if directory:
            self.output_path = directory
            self.save_settings()
            messagebox.showinfo("提示", f"保存路径已更新为：\n{self.output_path}")

    def setup_ui(self):
        top_bar = ttk.Frame(self.root, padding="5")
        top_bar.pack(fill=tk.X)

        ttk.Button(top_bar, text="📂 更改输入文件夹", command=self.change_input_directory).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_bar, text="💾 更改保存文件夹", command=self.change_output_directory).pack(side=tk.LEFT, padx=5)
        ttk.Label(top_bar, text=" | ").pack(side=tk.LEFT)
        self.status_label = ttk.Label(top_bar, text="准备就绪", foreground="gray")
        self.status_label.pack(side=tk.LEFT, padx=5)

        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True)

        # --- 左侧控制面板 (带滚动条) ---
        control_container = ttk.Frame(main_paned, width=340)
        main_paned.add(control_container, weight=0)
        
        canvas = tk.Canvas(control_container, bg=self.root.cget("bg"), highlightthickness=0)
        scrollbar = ttk.Scrollbar(control_container, orient="vertical", command=canvas.yview)
        
        self.scrollable_frame = ttk.Frame(canvas)
        self.scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        canvas_frame = canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.bind('<Configure>', lambda e: canvas.itemconfig(canvas_frame, width=e.width))
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _on_mousewheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

        self.control_frame = self.scrollable_frame

        # --- 主控制区内容 ---
        self.main_controls_frame = ttk.Frame(self.control_frame, padding="10")
        self.main_controls_frame.pack(fill=tk.X)

        ttk.Label(self.main_controls_frame, text="参数设置", font=("Arial", 14, "bold")).pack(pady=10)

        self.auto_apply_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.main_controls_frame, text="自动检测并应用参数", variable=self.auto_apply_var).pack(anchor=tk.W, pady=(0, 5))
        self.keep_manual_mask_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.main_controls_frame, text="保留手动修补 (画笔/橡皮)", variable=self.keep_manual_mask_var).pack(anchor=tk.W, pady=(0, 10))

        self.mode_var = tk.StringVar(value="color")
        ttk.Label(self.main_controls_frame, text="处理模式:").pack(anchor=tk.W, pady=(5, 0))
        mode_frame = ttk.Frame(self.main_controls_frame)
        mode_frame.pack(fill=tk.X)
        ttk.Radiobutton(mode_frame, text="彩色背景", variable=self.mode_var, value="color", command=self.on_mode_change).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(mode_frame, text="黑白背景", variable=self.mode_var, value="gray", command=self.on_mode_change).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(mode_frame, text="特定黄", variable=self.mode_var, value="yellow", command=self.on_mode_change).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(mode_frame, text="Rembg AI", variable=self.mode_var, value="rembg", command=self.on_mode_change).pack(side=tk.LEFT, padx=2)

        self.sliders = {}
        self.param_container = ttk.Frame(self.main_controls_frame)
        self.param_container.pack(fill=tk.X, pady=10)

        # --- 彩色模式 (RGBA) ---
        self.color_frame = ttk.LabelFrame(self.param_container, text="彩色模式参数 (RGBA)", padding="5")
        picker_frame = ttk.Frame(self.color_frame)
        picker_frame.pack(fill=tk.X, pady=5)
        ttk.Button(picker_frame, text="🎨 从图中吸取颜色", command=self.start_color_picking).pack(side=tk.LEFT)
        
        self.color_invert_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(picker_frame, text="反转选择 (选中背景)", variable=self.color_invert_var, command=self.schedule_update).pack(side=tk.LEFT, padx=10)

        self.color_picker_tolerance_label = ttk.Label(picker_frame, text="容差: 20")
        self.color_picker_tolerance_label.pack(side=tk.LEFT, padx=10)
        self.color_picker_tolerance = tk.IntVar(value=20)
        tolerance_slider = ttk.Scale(picker_frame, from_=0, to=100, variable=self.color_picker_tolerance, orient=tk.HORIZONTAL, command=lambda v: self.color_picker_tolerance_label.config(text=f"容差: {int(float(v))}"))
        tolerance_slider.pack(fill=tk.X, expand=True)
        
        slider_grid = ttk.Frame(self.color_frame)
        slider_grid.pack(fill=tk.X)
        slider_grid.columnconfigure(2, weight=1)
        slider_grid.columnconfigure(4, weight=1)
        self.add_channel_sliders(slider_grid, "R", "color_r", 0, 0, 255)
        self.add_channel_sliders(slider_grid, "G", "color_g", 1, 0, 255)
        self.add_channel_sliders(slider_grid, "B", "color_b", 2, 0, 255)
        self.add_channel_sliders(slider_grid, "A", "color_a", 3, 0, 255) # Alpha 默认全选

        self.gray_frame = ttk.LabelFrame(self.param_container, text="黑白模式参数", padding="5")
        self.add_slider(self.gray_frame, "gray_thresh", "灰度阈值 (0=Auto)", 0, 255, 0)
        bg_type_frame = ttk.Frame(self.gray_frame)
        bg_type_frame.pack(fill=tk.X)
        self.bg_type_var = tk.StringVar(value="black")
        ttk.Radiobutton(bg_type_frame, text="黑背景 (物体亮)", variable=self.bg_type_var, value="black", command=self.schedule_update).pack(side=tk.LEFT)
        ttk.Radiobutton(bg_type_frame, text="白背景 (物体暗)", variable=self.bg_type_var, value="white", command=self.schedule_update).pack(side=tk.LEFT)

        self.yellow_frame = ttk.LabelFrame(self.param_container, text="白底黄主体参数", padding="5")
        self.add_slider(self.yellow_frame, "yellow_h_center", "黄色中心 (Hue)", 0, 180, 30)
        self.add_slider(self.yellow_frame, "yellow_h_tol", "色相容差", 0, 90, 15)
        self.add_slider(self.yellow_frame, "yellow_s_min", "最小饱和度", 0, 255, 40)
        self.add_slider(self.yellow_frame, "yellow_v_min", "最小亮度", 0, 255, 40)

        self.rembg_frame = ttk.LabelFrame(self.param_container, text="Rembg AI 模式", padding="5")
        
        model_select_frame = ttk.Frame(self.rembg_frame)
        model_select_frame.pack(fill=tk.X, pady=5)
        ttk.Label(model_select_frame, text="模型:").pack(side=tk.LEFT)
        self.rembg_model_var = tk.StringVar(value="u2net")
        model_combo = ttk.Combobox(model_select_frame, textvariable=self.rembg_model_var, state="readonly", width=15)
        model_combo['values'] = ('u2net', 'u2netp', 'u2net_human_seg', 'isnet-general-use', 'isnet-anime')
        model_combo.pack(side=tk.LEFT, padx=5)
        model_combo.bind("<<ComboboxSelected>>", lambda e: self.schedule_update())

        self.rembg_alpha_matting_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.rembg_frame, text="启用 Alpha Matting (边缘优化)", 
                        variable=self.rembg_alpha_matting_var, 
                        command=self.schedule_update).pack(anchor=tk.W, pady=5)
        
        self.add_slider(self.rembg_frame, "rembg_shift_x", "位置微调 X", -50, 50, 0)
        self.add_slider(self.rembg_frame, "rembg_shift_y", "位置微调 Y", -50, 50, 0)
        self.add_slider(self.rembg_frame, "rembg_mask_thresh", "蒙版显示阈值", 1, 255, 127)
        self.add_slider(self.rembg_frame, "rembg_fg_thresh", "前景阈值", 0, 255, 240)
        self.add_slider(self.rembg_frame, "rembg_bg_thresh", "背景阈值", 0, 255, 10)
        self.add_slider(self.rembg_frame, "rembg_erode", "腐蚀大小", 0, 40, 10)

        self.morph_frame = ttk.LabelFrame(self.main_controls_frame, text="形态学处理 (连接主体)", padding="5")
        self.morph_frame.pack(fill=tk.X, pady=10)
        self.add_slider(self.morph_frame, "clean_kernel", "去噪核大小 (Open)", 0, 10, 3)
        self.add_slider(self.morph_frame, "connect_kernel", "连接核大小 (Close)", 0, 20, 5)
        self.add_slider(self.morph_frame, "connect_iters", "连接迭代次数", 0, 10, 2)

        action_frame = ttk.Frame(self.main_controls_frame)
        action_frame.pack(fill=tk.X, pady=10)
        
        self.apply_mask_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(action_frame, text="裁剪时应用蒙版去背", variable=self.apply_mask_var).pack(anchor=tk.W, pady=(0, 5))
        
        self.edit_mask_button = ttk.Button(action_frame, text="✏️ 编辑蒙版 (E)", command=self.toggle_mask_editing)
        self.edit_mask_button.pack(fill=tk.X, pady=(0, 5))
        ttk.Button(action_frame, text="✂️ 执行拆分并保存 (S)", command=self.save_crops, style="Accent.TButton").pack(fill=tk.X)

        nav_frame = ttk.Frame(self.main_controls_frame)
        nav_frame.pack(fill=tk.X, pady=20)
        ttk.Button(nav_frame, text="<< 上一张 (P)", command=self.prev_image).pack(side=tk.LEFT, padx=5, expand=True)
        ttk.Button(nav_frame, text="下一张 (N)", command=self.next_image).pack(side=tk.LEFT, padx=5, expand=True)

        self.info_label = ttk.Label(self.main_controls_frame, text="", foreground="blue", wraplength=280)
        self.info_label.pack(pady=5)

        self.edit_controls_frame = ttk.Frame(self.control_frame, padding="10")
        ttk.Label(self.edit_controls_frame, text="蒙版编辑模式", font=("Arial", 14, "bold")).pack(pady=10)
        edit_mode_frame = ttk.Frame(self.edit_controls_frame)
        edit_mode_frame.pack(fill=tk.X, pady=5)
        self.edit_mode_var = tk.StringVar(value="draw")
        ttk.Radiobutton(edit_mode_frame, text="画笔 (B)", variable=self.edit_mode_var, value="draw", command=self._set_edit_mode).pack(side=tk.LEFT)
        ttk.Radiobutton(edit_mode_frame, text="橡皮 (E)", variable=self.edit_mode_var, value="erase", command=self._set_edit_mode).pack(side=tk.LEFT, padx=10)
        ttk.Label(self.edit_controls_frame, text="笔刷大小:").pack(anchor=tk.W, pady=(10, 0))
        self.size_slider = ttk.Scale(self.edit_controls_frame, from_=1, to=100, value=self.brush_size, orient=tk.HORIZONTAL, command=self._update_brush_size)
        self.size_slider.pack(fill=tk.X)
        ttk.Button(self.edit_controls_frame, text="✅ 完成编辑 (M)", command=self.toggle_mask_editing, style="Accent.TButton").pack(fill=tk.X, pady=20)

        preview_frame = ttk.Frame(main_paned, padding="10")
        main_paned.add(preview_frame, weight=1)
        self.canvas = tk.Canvas(preview_frame, bg="#2b2b2b")
        self.canvas.pack(fill=tk.BOTH, expand=True)

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
        self.root.bind("<Up>", lambda e: self.pan_image(0, 50))
        self.root.bind("<Down>", lambda e: self.pan_image(0, -50))
        self.root.bind("<Left>", lambda e: self.pan_image(50, 0))
        self.root.bind("<Right>", lambda e: self.pan_image(-50, 0))

        self.on_mode_change()

    # --- 异步与防抖 ---
    def _processing_worker(self):
        rembg_cache = {}
        while True:
            try:
                params, raw_image, manual_draw, manual_erase, image_id = self.processing_request_queue.get()
                
                mode = params['mode']
                base_mask = None
                match_ratio = 0
                
                if mode == 'rembg':
                    cache_key = (image_id, params.get("rembg_model"), params.get("rembg_alpha_matting"), params.get("rembg_fg_thresh"), params.get("rembg_bg_thresh"), params.get("rembg_erode"))
                    if cache_key in rembg_cache:
                        base_mask = rembg_cache[cache_key].copy()
                    else:
                        if len(raw_image.shape) == 2: bgr_image = cv2.cvtColor(raw_image, cv2.COLOR_GRAY2BGR)
                        elif raw_image.shape[2] == 4: bgr_image = cv2.cvtColor(raw_image, cv2.COLOR_BGRA2BGR)
                        else: bgr_image = raw_image
                        base_mask = ImageProcessor.get_mask_rembg(bgr_image, model_name=params.get("rembg_model", "u2net"), alpha_matting=params.get("rembg_alpha_matting", False), am_fg_thresh=params.get("rembg_fg_thresh", 240), am_bg_thresh=params.get("rembg_bg_thresh", 10), am_erode=params.get("rembg_erode", 10))
                        if len(rembg_cache) > 5: rembg_cache.clear()
                        rembg_cache[cache_key] = base_mask.copy()
                elif mode == 'color':
                    base_mask, match_ratio = ImageProcessor.get_mask_rgba_range(
                        raw_image, 
                        params["color_r_min"], params["color_r_max"], 
                        params["color_g_min"], params["color_g_max"], 
                        params["color_b_min"], params["color_b_max"], 
                        params["color_a_min"], params["color_a_max"],
                        invert=params.get("color_invert", True)
                    )
                else:
                    if len(raw_image.shape) == 2: bgr_image = cv2.cvtColor(raw_image, cv2.COLOR_GRAY2BGR)
                    elif raw_image.shape[2] == 4: bgr_image = cv2.cvtColor(raw_image, cv2.COLOR_BGRA2BGR)
                    else: bgr_image = raw_image
                    if mode == 'gray':
                        base_mask = ImageProcessor.get_mask_gray(bgr_image, params["gray_thresh"], params["bg_type"])
                    elif mode == 'yellow':
                        base_mask = ImageProcessor.get_mask_yellow(bgr_image, params["yellow_h_center"], params["yellow_h_tol"], params["yellow_s_min"], params["yellow_v_min"])

                if base_mask is None:
                    h, w = raw_image.shape[:2]
                    base_mask = np.zeros((h, w), dtype=np.uint8)

                final_mask = self._apply_post_processing(base_mask, params, manual_draw, manual_erase)
                self.processing_result_queue.put((final_mask, match_ratio))

            except Exception as e:
                print(f"后台处理错误: {e}")

    def _apply_post_processing(self, mask, params, manual_draw_layer, manual_erase_layer):
        if params['mode'] == "rembg":
            sx, sy = params.get("rembg_shift_x", 0), params.get("rembg_shift_y", 0)
            if sx != 0 or sy != 0:
                mask = ImageProcessor.shift_mask(mask, sx, sy)
        
        mask = ImageProcessor.apply_morphology(mask, params["clean_kernel"], params["connect_kernel"], params["connect_iters"])
        
        if manual_draw_layer is not None:
            mask = cv2.bitwise_or(mask, manual_draw_layer)
        if manual_erase_layer is not None:
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(manual_erase_layer))
            
        return mask

    def _check_result_queue(self):
        try:
            result_mask, match_ratio = self.processing_result_queue.get_nowait()
            self.processed_mask = result_mask
            self.is_processing = False
            
            # 更新状态栏信息
            status_text = f"当前文件: {self.current_filename}"
            if self.mode_var.get() == "color":
                status_text += f" | 颜色匹配率: {match_ratio:.1%}"
            self.status_label.config(text=status_text)
            
            self.update_display()
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self._check_result_queue)

    def schedule_update(self):
        if self.debounce_job:
            self.root.after_cancel(self.debounce_job)
        self.debounce_job = self.root.after(250, self.update_preview)

    def update_preview(self):
        if self.current_image is None: return
        
        params = {'mode': self.mode_var.get(), 'bg_type': self.bg_type_var.get()}
        params['rembg_model'] = self.rembg_model_var.get()
        params['rembg_alpha_matting'] = self.rembg_alpha_matting_var.get()
        params['color_invert'] = self.color_invert_var.get()
        for name, var in self.sliders.items():
            params[name] = var.get()

        if not self.processing_request_queue.empty():
            try:
                self.processing_request_queue.get_nowait()
            except queue.Empty:
                pass
        
        request_data = (params, self.raw_image.copy(), self.manual_draw_layer.copy(), self.manual_erase_layer.copy(), self.current_filename)
        self.processing_request_queue.put(request_data)

        if not self.is_processing:
            self.is_processing = True
            self.status_label.config(text="处理中...")

    # --- 逻辑方法 ---
    def toggle_mask_editing(self):
        self.is_editing_mask = not self.is_editing_mask
        if self.is_editing_mask:
            self.main_controls_frame.pack_forget()
            self.edit_controls_frame.pack(fill=tk.X)
            self.edit_mask_button.config(text="完成编辑 (M)")
            self.canvas.config(cursor="none")
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

    def increase_brush(self, event=None):
        if not self.is_editing_mask: return
        new_size = min(100, self.brush_size + 5)
        self.size_slider.set(new_size)

    def start_paint(self, event):
        if self.is_picking_color:
            self.pick_color(event)
            return
        if not self.is_editing_mask: return
        self.drawing = True
        self.paint(event)

    def stop_paint(self, event):
        if not self.is_editing_mask: return
        self.drawing = False

    def paint(self, event):
        if not self.drawing or not self.is_editing_mask: return
        if self.manual_draw_layer is None or self.manual_erase_layer is None: return

        canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()
        h, w = self.current_image.shape[:2]
        base_scale = min(canvas_w / w, canvas_h / h)
        current_scale = base_scale * self.zoom_scale
        new_w, new_h = int(w * current_scale), int(h * current_scale)
        total_offset_x = (canvas_w - new_w) // 2 + self.pan_offset_x
        total_offset_y = (canvas_h - new_h) // 2 + self.pan_offset_y
        x = int((event.x - total_offset_x) / current_scale)
        y = int((event.y - total_offset_y) / current_scale)
        real_brush_radius = max(1, int((self.brush_size / 2) / current_scale))

        layer_to_draw = self.manual_draw_layer if self.edit_mode == "draw" else self.manual_erase_layer
        layer_to_clear = self.manual_erase_layer if self.edit_mode == "draw" else self.manual_draw_layer
        cv2.circle(layer_to_draw, (x, y), real_brush_radius, 255, -1)
        cv2.circle(layer_to_clear, (x, y), real_brush_radius, 0, -1)

        self.update_preview() # For painting, we want immediate feedback
        self.update_brush_cursor(event)

    def update_brush_cursor(self, event):
        self.last_mouse_pos = event
        if not self.is_editing_mask or event is None: return
        if self.brush_cursor_id: self.canvas.delete(self.brush_cursor_id)
        x, y = event.x, event.y
        r = self.brush_size / 2
        color = "white" if self.edit_mode == "draw" else "black"
        self.brush_cursor_id = self.canvas.create_oval(x - r, y - r, x + r, y + r, fill=color, outline="gray")

    def hide_brush_cursor(self, event=None):
        if self.brush_cursor_id:
            self.canvas.delete(self.brush_cursor_id)
            self.brush_cursor_id = None

    def on_mouse_wheel(self, event):
        self.zoom_scale *= 1.1 if event.delta > 0 else 1 / 1.1
        self.zoom_scale = max(0.1, self.zoom_scale)
        self.update_display()
        if self.is_editing_mask: self.update_brush_cursor(self.last_mouse_pos)

    def pan_image(self, dx, dy):
        self.pan_offset_x += dx
        self.pan_offset_y += dy
        self.update_display()
        if self.is_editing_mask: self.update_brush_cursor(self.last_mouse_pos)

    def on_mode_change(self):
        if not self.is_auto_detecting: self.auto_apply_var.set(False)
        mode = self.mode_var.get()
        self.color_frame.pack_forget()
        self.gray_frame.pack_forget()
        self.yellow_frame.pack_forget()
        self.rembg_frame.pack_forget()

        if mode == "color": self.color_frame.pack(fill=tk.X, expand=True)
        elif mode == "gray": self.gray_frame.pack(fill=tk.X, expand=True)
        elif mode == "yellow": self.yellow_frame.pack(fill=tk.X, expand=True)
        elif mode == "rembg": self.rembg_frame.pack(fill=tk.X, expand=True)
        
        self.schedule_update()

    def add_slider(self, parent, name, label, min_val, max_val, default):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)
        lbl = ttk.Label(frame, text=f"{label}: {default}")
        lbl.pack(anchor=tk.W)
        var = tk.IntVar(value=default)
        def on_change(val):
            if not self.is_auto_detecting: self.auto_apply_var.set(False)
            lbl.config(text=f"{label}: {int(float(val))}")
            if not self.is_editing_mask: self.schedule_update()
        slider = ttk.Scale(frame, from_=min_val, to=max_val, variable=var, orient=tk.HORIZONTAL, command=on_change)
        slider.pack(fill=tk.X)
        self.sliders[name] = var

    def add_channel_sliders(self, parent, label, name_prefix, row_idx, default_min, default_max):
        ttk.Label(parent, text=f"{label}:").grid(row=row_idx, column=0, sticky='w')
        
        var_min = tk.IntVar(value=default_min)
        lbl_min = ttk.Label(parent, text=str(default_min), width=4)
        lbl_min.grid(row=row_idx, column=1)
        slider_min = ttk.Scale(parent, from_=0, to=255, variable=var_min, orient=tk.HORIZONTAL, 
                               command=lambda v, l=lbl_min: (l.config(text=str(int(float(v)))), self.schedule_update()))
        slider_min.grid(row=row_idx, column=2, sticky='ew', padx=(0,5))
        self.sliders[f"{name_prefix}_min"] = var_min

        var_max = tk.IntVar(value=default_max)
        lbl_max = ttk.Label(parent, text=str(default_max), width=4)
        lbl_max.grid(row=row_idx, column=3)
        slider_max = ttk.Scale(parent, from_=0, to=255, variable=var_max, orient=tk.HORIZONTAL,
                               command=lambda v, l=lbl_max: (l.config(text=str(int(float(v)))), self.schedule_update()))
        slider_max.grid(row=row_idx, column=4, sticky='ew', padx=(0,5))
        self.sliders[f"{name_prefix}_max"] = var_max

    def load_image(self, index, force_auto_detect=False):
        if not self.files: return
        if self.is_editing_mask: self.toggle_mask_editing()

        self.current_index = max(0, min(index, len(self.files) - 1))
        self.current_filename = self.files[self.current_index]
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

        self.root.title(f"AI图片处理工厂 - {self.current_filename} ({self.current_index + 1}/{len(self.files)})")
        self.status_label.config(text=f"当前文件: {self.current_filename}")
        self.zoom_scale, self.pan_offset_x, self.pan_offset_y = 1.0, 0, 0
        h, w = self.current_image.shape[:2]

        should_reset_manual = not self.keep_manual_mask_var.get() or self.manual_draw_layer is None
        if not should_reset_manual:
            try:
                if self.manual_draw_layer.shape != (h, w):
                    self.manual_draw_layer = cv2.resize(self.manual_draw_layer, (w, h), interpolation=cv2.INTER_NEAREST)
                    self.manual_erase_layer = cv2.resize(self.manual_erase_layer, (w, h), interpolation=cv2.INTER_NEAREST)
            except:
                should_reset_manual = True
        
        if should_reset_manual:
            self.manual_draw_layer = np.zeros((h, w), dtype=np.uint8)
            self.manual_erase_layer = np.zeros((h, w), dtype=np.uint8)

        if force_auto_detect or self.auto_apply_var.get():
            self.auto_detect_params()
        else:
            self.update_preview()

    def auto_detect_params(self):
        self.is_auto_detecting = True
        img = self.current_image
        h, w, _ = img.shape
        corners = np.array([img[0, 0], img[0, w - 1], img[h - 1, 0], img[h - 1, w - 1]])
        mean_val = np.mean(np.median(corners, axis=0))

        if mean_val < 50:
            self.mode_var.set("gray")
            self.bg_type_var.set("black")
            self.sliders["gray_thresh"].set(0)
        elif mean_val > 200:
            self.mode_var.set("gray")
            self.bg_type_var.set("white")
            self.sliders["gray_thresh"].set(0)
        else:
            self.mode_var.set("yellow") # Fallback to yellow for other colors
            self.sliders["yellow_h_center"].set(30)
            self.sliders["yellow_h_tol"].set(15)
            self.sliders["yellow_s_min"].set(40)
            self.sliders["yellow_v_min"].set(40)
        
        self.on_mode_change()
        self.is_auto_detecting = False

    def update_display(self):
        if self.current_image is None or self.processed_mask is None: return

        img_rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        final_vis = None

        thresh_val = self.sliders.get("rembg_mask_thresh", tk.IntVar(value=127)).get()
        _, display_mask = cv2.threshold(self.processed_mask, thresh_val, 255, cv2.THRESH_BINARY)

        if self.is_editing_mask:
            overlay = np.zeros_like(img_rgb)
            overlay[display_mask == 255] = [255, 0, 0]
            final_vis = cv2.addWeighted(img_rgb, 0.4, overlay, 0.6, 0)
        else:
            mask_3c = cv2.cvtColor(display_mask, cv2.COLOR_GRAY2RGB)
            fg = cv2.bitwise_and(img_rgb, mask_3c)
            bg = cv2.bitwise_and(img_rgb, cv2.bitwise_not(mask_3c))
            bg = (bg * 0.3).astype(np.uint8)
            bg[:, :, 0] = np.clip(bg[:, :, 0] + 50, 0, 255)
            final_vis = cv2.add(fg, bg)
            contours, _ = cv2.findContours(display_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(final_vis, contours, -1, (0, 255, 0), 2)
            self.info_label.config(text=f"检测到 {len(contours)} 个对象")

        canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if canvas_w < 10 or canvas_h < 10: return

        h, w, _ = final_vis.shape
        current_scale = min(canvas_w / w, canvas_h / h) * self.zoom_scale
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

        thresh_val = self.sliders.get("rembg_mask_thresh", tk.IntVar(value=127)).get()
        _, save_mask = cv2.threshold(self.processed_mask, thresh_val, 255, cv2.THRESH_BINARY)
        
        if self.apply_mask_var.get():
            final_alpha = cv2.bitwise_and(original_a, save_mask)
        else:
            final_alpha = original_a
            
        final_full_image = cv2.merge([b, g, r, final_alpha])

        contours, _ = cv2.findContours(save_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        base_name = os.path.splitext(self.current_filename)[0]
        count = 0

        if not contours:
            self.info_label.config(text="未检测到可保存的对象！")
            return

        for i, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 10 or h < 10: continue
            crop = final_full_image[y:y + h, x:x + w]
            save_name = f"{base_name}_{count}.png"
            save_path = os.path.join(self.output_path, save_name)
            if ImageProcessor.cv_imwrite(save_path, crop):
                count += 1

        self.info_label.config(text=f"已保存 {count} 个切片！")
        self.root.after(1000, self.next_image)

    def start_color_picking(self):
        if self.is_editing_mask:
            messagebox.showwarning("提示", "请先完成蒙版编辑。")
            return
        self.is_picking_color = True
        self.canvas.config(cursor="crosshair")
        self.status_label.config(text="请在图片上点击以吸取颜色...")

    def _get_bgra_from_raw(self, raw_img):
        if len(raw_img.shape) == 2:
            return cv2.cvtColor(raw_img, cv2.COLOR_GRAY2BGRA)
        if raw_img.shape[2] == 3:
            return cv2.cvtColor(raw_img, cv2.COLOR_BGR2BGRA)
        return raw_img

    def pick_color(self, event):
        if self.raw_image is None: return

        image_bgra = self._get_bgra_from_raw(self.raw_image)
        h, w = image_bgra.shape[:2]

        canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()
        base_scale = min(canvas_w / w, canvas_h / h)
        current_scale = base_scale * self.zoom_scale
        new_w, new_h = int(w * current_scale), int(h * current_scale)
        total_offset_x = (canvas_w - new_w) // 2 + self.pan_offset_x
        total_offset_y = (canvas_h - new_h) // 2 + self.pan_offset_y
        
        img_x = int((event.x - total_offset_x) / current_scale)
        img_y = int((event.y - total_offset_y) / current_scale)

        if 0 <= img_x < w and 0 <= img_y < h:
            # 区域采样：取 9x9 区域
            x_start = max(0, img_x - 4)
            x_end = min(w, img_x + 5)
            y_start = max(0, img_y - 4)
            y_end = min(h, img_y + 5)
            
            roi = image_bgra[y_start:y_end, x_start:x_end]
            
            # 计算区域内的最小值和最大值
            min_vals = np.min(roi, axis=(0, 1))
            max_vals = np.max(roi, axis=(0, 1))
            
            b_min, g_min, r_min, a_min = min_vals
            b_max, g_max, r_max, a_max = max_vals
            
            tol = self.color_picker_tolerance.get()
            
            self.sliders["color_r_min"].set(max(0, int(r_min) - tol))
            self.sliders["color_r_max"].set(min(255, int(r_max) + tol))
            self.sliders["color_g_min"].set(max(0, int(g_min) - tol))
            self.sliders["color_g_max"].set(min(255, int(g_max) + tol))
            self.sliders["color_b_min"].set(max(0, int(b_min) - tol))
            self.sliders["color_b_max"].set(min(255, int(b_max) + tol))
            # Alpha 通道通常不需要太严格，除非用户明确想选半透明
            self.sliders["color_a_min"].set(max(0, int(a_min) - tol))
            self.sliders["color_a_max"].set(min(255, int(a_max) + tol))

        self.is_picking_color = False
        self.canvas.config(cursor="")
        self.status_label.config(text=f"当前文件: {self.current_filename}")
        self.schedule_update()


if __name__ == "__main__":
    root = tk.Tk()
    try:
        root.tk.call("source", "azure.tcl")
        root.tk.call("set_theme", "dark")
    except:
        pass
    app = ImageCutterApp(root)
    root.mainloop()
