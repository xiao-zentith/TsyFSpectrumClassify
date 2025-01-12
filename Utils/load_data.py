import os
import numpy as np
from Utils.read_matrix import read_matrix_from_file


def load_data(input_folder):
    """
    加载所有数据并返回特征和标签
    :param input_folder: 输入文件夹路径
    :return: 特征数组, 标签数组, 标签映射字典, 文件路径列表
    """
    features = []
    labels = []
    label_map = {}
    file_paths = []

    for label_idx, label in enumerate(os.listdir(input_folder)):
        label_map[label] = label_idx
        label_folder = os.path.join(input_folder, label)

        if not os.path.isdir(label_folder):
            continue

        for txt_file in os.listdir(label_folder):
            if not txt_file.endswith('.txt'):
                continue

            file_path = os.path.join(label_folder, txt_file)
            x_coords, y_coords, matrix = read_matrix_from_file(file_path)

            # 展平矩阵为特征向量
            feature_vector = matrix.flatten()

            # 将横坐标和纵坐标展平并添加到特征向量中
            feature_vector = np.concatenate((x_coords, y_coords, feature_vector))

            features.append(feature_vector)
            labels.append(label_idx)
            file_paths.append(file_path)

    # 将列表转换为numpy数组
    X = np.array(features)
    y = np.array(labels)

    return X, y, label_map, file_paths