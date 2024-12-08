import numpy as np
import os


def process_data(file_path):
    # 读取文件
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 提取横坐标
    horizontal_coords = [float(line.strip()) for line in lines[0].split()]

    # 初始化纵坐标和矩阵值
    vertical_coords = []
    matrix_values = []

    # 遍历文件中的数据
    for i in range(1, len(lines)):
        line_values = [float(val) for val in lines[i].split()]
        vertical_coords.append(line_values[0])
        # 将横纵坐标对应的值添加到矩阵值中
        matrix_values.append(line_values[1:])

    # 将矩阵值转换为numpy数组
    matrix_values = np.array(matrix_values)

    # 创建一个新的23*22矩阵，初始值为0
    result_matrix = np.zeros((23, 22))

    # 遍历横纵坐标和矩阵值，只保留符合条件的数据
    for j in range(len(horizontal_coords)):
        row_index = -1
        for i in range(len(vertical_coords)):
            if vertical_coords[i] >= horizontal_coords[j] + 10 and vertical_coords[i] <= horizontal_coords[j] + 120:
                # 计算在23*21矩阵中的位置
                row_index += 1
                col_index = j
                result_matrix[row_index, col_index + 1] = matrix_values[i, j]  # 减1是因为第一列是纵坐标

    for i in range(23):
        result_matrix[i, 0] = (i + 2) * 5
    return result_matrix, horizontal_coords


def process_folder(folder_path, output_folder_path):
    # 遍历文件夹
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.txt'):
                file_path = os.path.join(root, file)
                result_matrix, horizontal_coords = process_data(file_path)

                # 创建输出文件的路径
                relative_path = os.path.relpath(root, folder_path)
                output_file_path = os.path.join(output_folder_path, relative_path, file)
                output_dir = os.path.dirname(output_file_path)

                # 如果输出目录不存在，则创建它
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                # 将结果写入新的TXT文件
                with open(output_file_path, 'w') as output_file:
                    output_file.write(' '.join(map(str, horizontal_coords)) + '\n')
                    for row in result_matrix:
                        output_file.write(' '.join(map(str, row)) + '\n')


# 假设你的TXT文件所在的文件夹路径是'data_folder'
input_folder_path = r'C:\Users\xiao\Desktop\论文汇总\data\dataset\dataset_extract'
output_folder_path = r'C:\Users\xiao\Desktop\论文汇总\data\dataset\dataset_TsyF'
process_folder(input_folder_path, output_folder_path)