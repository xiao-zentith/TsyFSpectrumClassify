import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import tkinter as tk
from tkinter import filedialog, messagebox, colorchooser
import matplotlib.colors as mcolors


def read_matrix_from_txt(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        horizontal_coords = [float(value) for value in lines[0].strip().split()]
        vertical_coords = [float(line.strip().split()[0]) for line in lines[1:]]
        matrix_data = []
        for line in lines[1:]:
            row = [float(value) for value in line.strip().split()[1:]]
            matrix_data.append(row)

        # Reverse the horizontal coordinates and corresponding data rows
        # horizontal_coords.reverse()
        # matrix_data.reverse()

        return np.array(matrix_data), horizontal_coords, vertical_coords
    except Exception as e:
        log_text.insert(tk.END, f"Error reading file {file_path}: {str(e)}\n")
        return None, None, None


# 绘制并保存热力图
def plot_and_save_contour(input_file_path, output_dir, title, xlabel, ylabel, levels, cmap, alpha):
    try:
        spectral_data, horizontal_coords, vertical_coords = read_matrix_from_txt(input_file_path)
        if spectral_data is None:
            return

        X, Y = np.meshgrid(horizontal_coords, vertical_coords)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        contour = ax.contour3D(X, Y, spectral_data, levels=levels, cmap=cmap, alpha=alpha)
        fig.colorbar(contour, ax=ax, shrink=0.5, aspect=5)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel('Value')
        filename = os.path.basename(input_file_path).replace('.txt', '.png')
        relative_path = os.path.relpath(os.path.dirname(input_file_path), input_dir.get())
        output_subdir = os.path.join(output_dir, relative_path)
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)
        output_file_path = os.path.join(output_subdir, filename)
        plt.savefig(output_file_path)
        plt.close(fig)
        log_text.insert(tk.END, f"File saved: {output_file_path}\n")
    except Exception as e:
        log_text.insert(tk.END, f"Error processing file: {str(e)}\n")


# 遍历文件夹及其子文件夹中的所有TXT文件
def process_all_txt_files(input_dir_str, output_dir_str):
    for root, dirs, files in os.walk(input_dir_str):
        for file in files:
            if file.endswith('.txt'):
                input_file_path = os.path.join(root, file)
                cmap = cmap_var.get() if mode_var.get() == "preset" else custom_cmap
                plot_and_save_contour(input_file_path, output_dir_str, title.get(), xlabel.get(), ylabel.get(),
                                      int(levels.get()), cmap, float(alpha.get()))


# UI界面
def browse_input_dir():
    input_dir.set(filedialog.askdirectory(title="Select Input Directory"))
    validate_input_dir()


def browse_output_dir():
    output_dir.set(filedialog.askdirectory(title="Select Output Directory"))
    validate_output_dir()


def start_processing():
    input_path = input_dir.get()
    output_path = output_dir.get()
    if input_path and output_path:
        process_all_txt_files(input_path, output_path)
    else:
        messagebox.showerror("Error", "Both input and output directories must be specified.")


def validate_input_dir():
    if not input_dir.get():
        messagebox.showerror("Error", "Input directory must not be empty.")
    elif not os.path.isdir(input_dir.get()):
        messagebox.showerror("Error", "Input directory does not exist.")


def validate_output_dir():
    if not output_dir.get():
        messagebox.showerror("Error", "Output directory must not be empty.")
    elif not os.path.isdir(output_dir.get()):
        messagebox.showerror("Error", "Output directory does not exist.")


def clear_log():
    log_text.delete(1.0, tk.END)


def toggle_mode(mode):
    global custom_cmap
    if mode == "preset":
        cmap_frame.grid(row=1, column=0, columnspan=4, sticky='ew', padx=10, pady=10)
        color_range_frame.grid_remove()
    elif mode == "custom":
        cmap_frame.grid_remove()
        color_range_frame.grid(row=1, column=0, columnspan=4, sticky='ew', padx=10, pady=10)
        custom_cmap = generate_custom_colormap(min_color.get(), max_color.get())


def choose_color(var):
    color_code = colorchooser.askcolor(title="Choose color")[1]
    var.set(color_code)
    update_color_display(var)


def update_color_display(var):
    color_label = min_color_label if var == min_color else max_color_label
    color_label.config(bg=var.get())
    toggle_mode(mode_var.get())  # Update colormap when color changes


def generate_custom_colormap(min_color, max_color):
    colors = [(0, min_color), (1, max_color)]
    cmap_name = 'custom'
    return plt.cm.colors.LinearSegmentedColormap.from_list(cmap_name, colors, N=256)


# 创建主窗口
root = tk.Tk()
root.title("TXT File Processor")
root.geometry('800x800')

# 模式选择
mode_var = tk.StringVar(value="preset")
preset_radio = tk.Radiobutton(root, text="Preset Colormap", variable=mode_var, value="preset",
                              command=lambda: toggle_mode("preset"))
preset_radio.grid(row=0, column=0, sticky='w', padx=10, pady=10)
custom_radio = tk.Radiobutton(root, text="Custom Color Range", variable=mode_var, value="custom",
                              command=lambda: toggle_mode("custom"))
custom_radio.grid(row=0, column=1, sticky='w', padx=10, pady=10)

# 输入目录区域
input_dir_frame = tk.Frame(root)
input_dir_frame.grid(row=2, column=0, columnspan=4, sticky='ew', padx=10, pady=10)

tk.Label(input_dir_frame, text="Input Directory:").grid(row=0, column=0, sticky='e')
input_dir = tk.StringVar()
input_entry = tk.Entry(input_dir_frame, textvariable=input_dir, width=50)
input_entry.grid(row=0, column=1, sticky='ew')
tk.Button(input_dir_frame, text="Browse", command=browse_input_dir).grid(row=0, column=2, sticky='ew')
tk.Button(input_dir_frame, text="Reset", command=lambda: input_dir.set("")).grid(row=0, column=3, sticky='ew')

# 输出目录区域
output_dir_frame = tk.Frame(root)
output_dir_frame.grid(row=3, column=0, columnspan=4, sticky='ew', padx=10, pady=10)

tk.Label(output_dir_frame, text="Output Directory:").grid(row=0, column=0, sticky='e')
output_dir = tk.StringVar()
output_entry = tk.Entry(output_dir_frame, textvariable=output_dir, width=50)
output_entry.grid(row=0, column=1, sticky='ew')
tk.Button(output_dir_frame, text="Browse", command=browse_output_dir).grid(row=0, column=2, sticky='ew')
tk.Button(output_dir_frame, text="Reset", command=lambda: output_dir.set("")).grid(row=0, column=3, sticky='ew')

# 等高线密度滑动模块
levels_frame = tk.Frame(root)
levels_frame.grid(row=4, column=0, columnspan=4, sticky='ew', padx=10, pady=10)

tk.Label(levels_frame, text="Contour Levels:").grid(row=0, column=0, sticky='e')
levels = tk.IntVar()
levels_slider = tk.Scale(levels_frame, from_=10, to=1000, orient='horizontal', label='Density', length=500,
                         variable=levels)
levels_slider.grid(row=0, column=1, sticky='ew')

# 透明度滑动模块
alpha_frame = tk.Frame(root)
alpha_frame.grid(row=5, column=0, columnspan=4, sticky='ew', padx=10, pady=10)

tk.Label(alpha_frame, text="Transparency:").grid(row=0, column=0, sticky='e')
alpha = tk.DoubleVar(value=1.0)  # Default fully opaque
alpha_slider = tk.Scale(alpha_frame, from_=0.0, to=1.0, orient='horizontal', resolution=0.01, label='Alpha', length=500,
                        variable=alpha)
alpha_slider.grid(row=0, column=1, sticky='ew')

# 自定义颜色范围选择框
color_range_frame = tk.Frame(root)
color_range_frame.grid(row=1, column=0, columnspan=4, sticky='ew', padx=10, pady=10)

tk.Label(color_range_frame, text="Color Range Min:").grid(row=0, column=0, sticky='e')
min_color = tk.StringVar(value="#0000FF")  # Default blue
min_color_button = tk.Button(color_range_frame, text="Choose Color", command=lambda: choose_color(min_color))
min_color_button.grid(row=0, column=1, sticky='ew')
min_color_label = tk.Label(color_range_frame, bg=min_color.get(), width=10)
min_color_label.grid(row=0, column=2, sticky='ew')

tk.Label(color_range_frame, text="Color Range Max:").grid(row=0, column=3, sticky='e')
max_color = tk.StringVar(value="#FF0000")  # Default red
max_color_button = tk.Button(color_range_frame, text="Choose Color", command=lambda: choose_color(max_color))
max_color_button.grid(row=0, column=4, sticky='ew')
max_color_label = tk.Label(color_range_frame, bg=max_color.get(), width=10)
max_color_label.grid(row=0, column=5, sticky='ew')

# 标题和轴标签输入框
title_frame = tk.Frame(root)
title_frame.grid(row=6, column=0, columnspan=4, sticky='ew', padx=10, pady=10)

tk.Label(title_frame, text="Title:").grid(row=0, column=0, sticky='e')
title = tk.StringVar(value="3D Contour")
title_entry = tk.Entry(title_frame, textvariable=title, width=50)
title_entry.grid(row=0, column=1, sticky='ew')

tk.Label(title_frame, text="X-axis Label:").grid(row=1, column=0, sticky='e')
xlabel = tk.StringVar(value="Excitation Spectrum/nm")
xlabel_entry = tk.Entry(title_frame, textvariable=xlabel, width=50)
xlabel_entry.grid(row=1, column=1, sticky='ew')

tk.Label(title_frame, text="Y-axis Label:").grid(row=2, column=0, sticky='e')
ylabel = tk.StringVar(value="Emission Spectrum/nm")
ylabel_entry = tk.Entry(title_frame, textvariable=ylabel, width=50)
ylabel_entry.grid(row=2, column=1, sticky='ew')

# 颜色映射选择
cmap_frame = tk.Frame(root)
cmap_frame.grid(row=7, column=0, columnspan=4, sticky='ew', padx=10, pady=10)

tk.Label(cmap_frame, text="Color Map:").grid(row=0, column=0, sticky='e')
cmap_var = tk.StringVar(value="viridis")  # Default colormap
cmap_options = ["viridis", "plasma", "inferno", "magma", "cividis"]
cmap_menu = tk.OptionMenu(cmap_frame, cmap_var, *cmap_options)
cmap_menu.grid(row=0, column=1, sticky='ew')

# 日志文本框
log_frame = tk.Frame(root)
log_frame.grid(row=8, column=0, columnspan=4, sticky='nsew', padx=10, pady=10)

scrollbar = tk.Scrollbar(log_frame)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
log_text = tk.Text(log_frame, width=60, height=20, yscrollcommand=scrollbar.set)
log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
scrollbar.config(command=log_text.yview)

# 重置日志按钮
tk.Button(root, text="Reset Log", command=clear_log).grid(row=9, column=0, sticky='ew', padx=10)

# 开始处理按钮
tk.Button(root, text="Start Processing", command=start_processing).grid(row=9, column=2, sticky='ew', padx=10)

# 初始状态
toggle_mode("preset")

# 运行UI循环
root.mainloop()






