import numpy as np


def read_matrix_from_file(file_path):
    """
    从TXT文件中读取数据并构建矩阵
    :param file_path: 文件路径
    :return: 横坐标数组, 纵坐标数组, 数据矩阵
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 提取横坐标
    x_coords = list(map(float, lines[0].strip().split()))

    # 提取纵坐标和数据矩阵
    y_coords = []
    data_matrix = []
    for line in lines[1:]:
        parts = list(map(float, line.strip().split()))
        y_coords.append(parts[0])
        data_matrix.append(parts[1:])

    return np.array(x_coords), np.array(y_coords), np.array(data_matrix)