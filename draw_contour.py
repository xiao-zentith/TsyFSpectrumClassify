import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


# 读取TXT文件中的二维矩阵数据
def read_matrix_from_txt(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 第一行包含横坐标
    horizontal_coords = [float(value) for value in lines[0].strip().split()]

    # 第二行开始的每一行的第一个元素是纵坐标
    vertical_coords = [float(line.strip().split()[0]) for line in lines[1:]]

    # 读取矩阵数据，跳过每行的第一个元素
    matrix_data = []
    for line in lines[1:]:
        row = [float(value) for value in line.strip().split()[1:]]  # 从第二个元素开始取
        matrix_data.append(row)

    return np.array(matrix_data), horizontal_coords, vertical_coords


# 绘制并保存热力图
def plot_and_save_contour(input_file_path, output_dir):
    spectral_data, horizontal_coords, vertical_coords = read_matrix_from_txt(input_file_path)

    # 创建X和Y坐标网格
    X, Y = np.meshgrid(horizontal_coords, vertical_coords)

    # 创建一个图形和一个子图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制等高线图，增加levels参数的值以增加等高线数量
    contour = ax.contour3D(X, Y, spectral_data, levels=500, cmap='viridis')

    # 添加颜色条
    fig.colorbar(contour, ax=ax, shrink=0.5, aspect=5)

    # # 添加原光谱的标题和坐标轴标签
    # ax.set_title('3D Contour Plot of Spectral Data')
    # ax.set_xlabel('Excitation Spectrum')
    # ax.set_ylabel('Emission Spectrum')
    # ax.set_zlabel('Value')

    # # 添加TsyF光谱的标题和坐标轴标签
    # ax.set_title('3D Contour Plot of TsyF Data')
    # ax.set_xlabel('Excitation Spectrum')
    # ax.set_ylabel('Interval')
    # ax.set_zlabel('Value')

    # 添加原光谱的标题和坐标轴标签
    ax.set_title('3D Contour Plot of Norm Data')
    ax.set_xlabel('Excitation Spectrum')
    ax.set_ylabel('Emission Spectrum')
    ax.set_zlabel('Value')

    # 保存图形
    filename = os.path.basename(input_file_path)
    output_file_path = os.path.join(output_dir, filename.replace('.txt', '.png'))
    plt.savefig(output_file_path)
    plt.close(fig)


# 遍历文件夹及其子文件夹中的所有TXT文件
def process_all_txt_files(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.txt'):
                input_file_path = os.path.join(root, file)
                plot_and_save_contour(input_file_path, output_dir)


# # 指定输入和输出文件夹路径
# input_dir = r'C:\Users\xiao\Desktop\论文汇总\data\dataset_after_extract'
# output_dir = r'C:\Users\xiao\Desktop\论文汇总\data\contour_spectrum'

# # 指定输入和输出文件夹路径
# input_dir = r'C:\Users\xiao\Desktop\论文汇总\data\dataset_to_TsyF'
# output_dir = r'C:\Users\xiao\Desktop\论文汇总\data\contour_TsyF'

# 指定输入和输出文件夹路径
input_dir = r'C:\Users\xiao\Desktop\论文汇总\data\dataset_after_norm'
output_dir = r'C:\Users\xiao\Desktop\论文汇总\data\contour_norm'

# 处理所有TXT文件
process_all_txt_files(input_dir, output_dir)