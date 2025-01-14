import os
import numpy as np
from Utils.read_matrix import read_matrix_from_file


def load_data(data_list, label_map):
    """
    加载数据并返回特征和标签
    :param data_list: 包含文件路径和类别的列表
    :return: 特征数组, 标签数组, 标签映射字典
    """
    features = []
    labels = []

    for item in data_list:
        label = item['category']

        file_path = item['input']
        x_coords, y_coords, matrix = read_matrix_from_file(file_path)

        # 展平矩阵为特征向量
        feature_vector = matrix.flatten()

        # 将横坐标和纵坐标展平并添加到特征向量中
        feature_vector = np.concatenate((x_coords, y_coords, feature_vector))

        features.append(feature_vector)
        labels.append(label_map[label])

    # 将列表转换为numpy数组
    X = np.array(features)
    y = np.array(labels)

    return X, y