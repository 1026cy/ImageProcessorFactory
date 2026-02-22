# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import cv2
import os
import threading
import time

# ==========================================
# 核心逻辑模块 (Core Logic Module)
# ==========================================
class VideoProcessor:
    """
    负责视频处理的核心业务逻辑，不包含任何界面代码。
    方便后期移植或作为独立库使用。
    """
    def __init__(self):
        self.is_running = False

    def extract_frames(self, video_path, output_dir, frame_interval, log_callback):
        """
        执行拆帧操作
        :param video_path: 视频源路径
        :param output_dir: 保存目录
        :param frame_interval: 帧间隔 (每隔多少帧保存一次)
        :param log_callback: 用于向界面发送日志的回调函数
        """
        if not os.path.exists(video_path):
            log_callback("错误：找不到视频文件！")
            return

        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
                log_callback(f"提示：创建输出目录 {output_dir}")
            except Exception as e:
                log_callback(f"错误：无法创建目录 - {str(e)}")
                return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            log_callback("错误：无法打开视频文件，请检查格式。")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        log_callback(f"视频加载成功。总帧数: {total_frames}, FPS: {fps:.2f}")
        log_callback(f"开始处理... 每 {frame_interval} 帧保存一张。")

        count = 0
        saved_count = 0
        self.is_running = True

        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                break

            if count % frame_interval == 0:
                frame_name = f"frame_{count:06d}.jpg"
                save_path = os.path.join(output_dir, frame_name)
                try:
                    cv2.imwrite(save_path, frame)
                    saved_count += 1
                    if saved_count % 10 == 0 or saved_count == 1:
                        log_callback(f"已保存: {frame_name} (进度: {count}/{total_frames})")
                except Exception as e:
                    log_callback(f"保存失败: {frame_name} - {e}")

            count += 1

        cap.release()
        self.is_running = False
        log_callback("============================")
        log_callback(f"处理完成！共保存 {saved_count} 张图片。")
        log_callback(f"文件保存在: {output_dir}")
        log_callback("============================")

# ==========================================
# 用户界面模块 (UI Module)
# ==========================================
class AppUI:
    """
    负责图形界面的展示和交互。
    """
    def __init__(self, root):
        self.root = root
        self.root.title("视频拆帧工具 - 傻瓜版")
        self.root.geometry("600x550") # 稍微加高一点窗口
        
        self.processor = VideoProcessor()
        self.create_widgets()

    def create_widgets(self):
        # 1. 视频选择区域
        frame_input = tk.LabelFrame(self.root, text="第一步：选择视频文件", padx=10, pady=10)
        frame_input.pack(fill="x", padx=10, pady=5)

        self.entry_video = tk.Entry(frame_input, width=50)
        self.entry_video.pack(side="left", padx=5, expand=True)
        self.entry_video.bind("<KeyRelease>", self.update_estimated_frames)
        
        btn_video = tk.Button(frame_input, text="浏览...", command=self.select_video)
        btn_video.pack(side="left")

        # 2. 输出目录区域
        frame_output = tk.LabelFrame(self.root, text="第二步：选择保存位置", padx=10, pady=10)
        frame_output.pack(fill="x", padx=10, pady=5)

        self.entry_output = tk.Entry(frame_output, width=50)
        self.entry_output.pack(side="left", padx=5, expand=True)

        btn_output = tk.Button(frame_output, text="浏览...", command=self.select_output)
        btn_output.pack(side="left")

        # 3. 设置区域
        frame_settings = tk.LabelFrame(self.root, text="第三步：设置拆分频率", padx=10, pady=10)
        frame_settings.pack(fill="x", padx=10, pady=5)

        row1 = tk.Frame(frame_settings)
        row1.pack(fill="x")
        tk.Label(row1, text="每隔").pack(side="left")
        self.entry_interval = tk.Entry(row1, width=5)
        self.entry_interval.insert(0, "30")
        self.entry_interval.pack(side="left", padx=5)
        self.entry_interval.bind("<KeyRelease>", self.update_estimated_frames)
        tk.Label(row1, text="帧保存一张图片。").pack(side="left")

        row2 = tk.Frame(frame_settings)
        row2.pack(fill="x")
        tk.Label(row2, text="(提示：填1则保存所有帧，填30约等于1秒一张)", fg="gray").pack(side="left")

        row3 = tk.Frame(frame_settings)
        row3.pack(fill="x", pady=(5,0))
        self.label_estimate = tk.Label(row3, text="预计保存: N/A", fg="blue", font=("Arial", 10, "bold"))
        self.label_estimate.pack(side="left")

        # 4. 操作区域
        frame_action = tk.Frame(self.root, pady=10)
        frame_action.pack(fill="x", padx=10)

        self.btn_start = tk.Button(frame_action, text="开始拆解视频", bg="#4CAF50", fg="white", font=("Arial", 12, "bold"), command=self.start_processing)
        self.btn_start.pack(fill="x")

        # 5. 日志显示区域
        frame_log = tk.LabelFrame(self.root, text="运行日志", padx=10, pady=10)
        frame_log.pack(fill="both", expand=True, padx=10, pady=5)

        self.log_area = scrolledtext.ScrolledText(frame_log, height=10)
        self.log_area.pack(fill="both", expand=True)

    def select_video(self):
        filename = filedialog.askopenfilename(title="选择视频文件", filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv"), ("所有文件", "*.*")])
        if filename:
            self.entry_video.delete(0, tk.END)
            self.entry_video.insert(0, filename)
            default_out = os.path.join(os.path.dirname(filename), "output_frames")
            if not self.entry_output.get():
                self.entry_output.delete(0, tk.END)
                self.entry_output.insert(0, default_out)
            self.update_estimated_frames()

    def select_output(self):
        directory = filedialog.askdirectory(title="选择保存图片的文件夹")
        if directory:
            self.entry_output.delete(0, tk.END)
            self.entry_output.insert(0, directory)

    def log(self, message):
        self.log_area.insert(tk.END, message + "\n")
        self.log_area.see(tk.END)

    def update_estimated_frames(self, event=None):
        video_path = self.entry_video.get()
        interval_str = self.entry_interval.get()

        if not os.path.exists(video_path) or not interval_str.isdigit() or int(interval_str) < 1:
            self.label_estimate.config(text="预计保存: N/A")
            return

        interval = int(interval_str)
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.label_estimate.config(text="预计保存: 无法打开视频")
                return

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            if total_frames > 0:
                estimated_count = (total_frames - 1) // interval + 1
                self.label_estimate.config(text=f"预计保存: {estimated_count} 张图片")
            else:
                self.label_estimate.config(text="预计保存: 无法读取帧数")
        except Exception:
            self.label_estimate.config(text="预计保存: 计算出错")

    def start_processing(self):
        video_path = self.entry_video.get()
        output_dir = self.entry_output.get()
        interval_str = self.entry_interval.get()

        if not video_path:
            messagebox.showwarning("提示", "请先选择视频文件！")
            return
        if not output_dir:
            messagebox.showwarning("提示", "请选择保存路径！")
            return
        if not interval_str.isdigit() or int(interval_str) < 1:
            messagebox.showwarning("提示", "间隔帧数必须是大于0的整数！")
            return

        self.btn_start.config(state="disabled", text="正在处理中...")
        self.log_area.delete(1.0, tk.END)

        thread = threading.Thread(target=self.run_thread, args=(video_path, output_dir, int(interval_str)))
        thread.daemon = True
        thread.start()

    def run_thread(self, video_path, output_dir, interval):
        self.processor.extract_frames(video_path, output_dir, interval, self.log)
        self.root.after(0, lambda: self.btn_start.config(state="normal", text="开始拆解视频"))
        self.root.after(0, lambda: messagebox.showinfo("完成", "视频拆解任务已完成！"))

# ==========================================
# 程序入口
# ==========================================
if __name__ == "__main__":
    # 安装依赖: pip install opencv-python
    root = tk.Tk()
    app = AppUI(root)
    root.mainloop()
