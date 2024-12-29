import numpy as np
import os
import shutil


class NormProcessor:
    def __init__(self, input_folder, output_folder):
        self.input_folder = input_folder
        self.output_folder = output_folder
        pass

    # 读取TXT文件
    def read_data(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # 提取横坐标（第一行）
        x_coords = [float(val) for val in lines[0].split()]

        # 提取纵坐标（除第一行外的第一列）
        y_coords = [float(line.split()[0]) for line in lines[1:]]

        # 提取矩阵实际值部分（第2行开始到最后一行且第二列开始到最后一列）
        matrix_values = []
        for line in lines[1:]:
            row_values = [float(val) for val in line.split()[1:]]  # 从第二列开始到最后一列
            matrix_values.append(row_values)

        # 将矩阵实际值部分转换为numpy数组
        matrix = np.array(matrix_values)

        return x_coords, y_coords, matrix

    # 对矩阵进行Z-score标准化
    def z_score_normalization(self, matrix):
        return (matrix - np.mean(matrix, axis=0)) / np.std(matrix, axis=0)

    # 将标准化后的矩阵值放回原位置，并保存到新文件
    def replace_matrix_values(self, x_coords, y_coords, normalized_matrix, file_path):
        with open(file_path, 'w') as file:
            # 写入横坐标
            file.write(' '.join(f"{val:.4f}" for val in x_coords) + '\n')

            # 写入纵坐标和标准化后的矩阵值
            for i, y_coord in enumerate(y_coords):
                # 将纵坐标和标准化后的矩阵值写入每一行
                row_str = f"{y_coord:.4f} " + ' '.join(f"{val:.4f}" for val in normalized_matrix[i]) + '\n'
                file.write(row_str)

    # 处理单个文件
    def process_file(self, file_path, input_dir, output_dir):
        relative_path = os.path.relpath(file_path, start=input_dir)
        new_file_path = os.path.join(output_dir, relative_path)

        # 确保新文件的目录存在
        os.makedirs(os.path.dirname(new_file_path), exist_ok=True)

        x_coords, y_coords, matrix = self.read_data(file_path)
        normalized_matrix = self.z_score_normalization(matrix)
        self.replace_matrix_values(x_coords, y_coords, normalized_matrix, new_file_path)

    # 批量处理文件夹
    def batch_process(self, input_dir, output_dir):
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 遍历输入目录及其子目录
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    self.process_file(file_path, input_dir, output_dir)

    # 对外提供的接口
    def process_folders(self):
        self.batch_process(self.input_folder, self.output_folder)


# 示例调用
if __name__ == '__main__':
    input_folder = r'C:\Users\xiao\Desktop\画大饼环节\data\dataset_EEM\EEM_noise'  # 替换为你的输入文件夹路径
    output_folder = r'C:\Users\xiao\Desktop\画大饼环节\data\dataset_EEM\EEM_norm'  # 替换为你的输出文件夹路径
    processor = NormProcessor(input_folder, output_folder)
    processor.process_folders()



