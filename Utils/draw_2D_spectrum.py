import os
import matplotlib.pyplot as plt
from cycler import cycler


def read_spectrum_data(file_path):
    x = []
    y = []
    with open(file_path, 'r') as file:
        for line in file:
            data = line.strip().split()
            if len(data) == 2:
                x.append(float(data[0]))
                y.append(float(data[1]))
    return x, y


def plot_spectra(folder_path):
    colors = plt.cm.tab20.colors  # tab20 包含 20 种颜色
    plt.figure(figsize=(10, 6))
    plt.rc('axes', prop_cycle=cycler(color=colors))
    plt.tick_params(axis='both', which='major', labelsize=20)
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            x, y = read_spectrum_data(file_path)
            plt.plot(x, y, label=filename.split('.')[0])  # 使用文件名去掉扩展名作为标签

    plt.xlabel('Emission / nm', fontsize=20)
    plt.ylabel('Intensity / nm', fontsize=20)
    plt.title('Emission Spectrum', fontsize=24)
    plt.rcParams.update({'font.size': 12})
    plt.legend()
    plt.grid(False)
    plt.show()


# 使用示例：
folder_path = r'C:\Users\xiao\Desktop\Draw-flatbread\data\em_spectrum'  # 替换为你的文件夹路径
plot_spectra(folder_path)



