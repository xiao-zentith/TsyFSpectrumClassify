import numpy as np


def calculate_relative_error(actual_matrix, predicted_matrix):
    """
    计算两个二维矩阵之间的相对误差。

    参数:
        predicted_matrix (numpy.ndarray): 预测值矩阵。
        actual_matrix (numpy.ndarray): 实际值矩阵。

    返回:
        relative_error_matrix (numpy.ndarray): 每个元素对应的相对误差矩阵。
        mean_relative_error (float): 平均相对误差。
    """
    # 确保输入是NumPy数组且形状相同
    if not isinstance(predicted_matrix, np.ndarray):
        predicted_matrix = np.array(predicted_matrix)
    if not isinstance(actual_matrix, np.ndarray):
        actual_matrix = np.array(actual_matrix)

    if predicted_matrix.shape != actual_matrix.shape:
        raise ValueError("两个矩阵的形状必须相同")

    # 避免除以零的情况
    mask = actual_matrix != 0

    # 初始化相对误差矩阵
    relative_error_matrix = np.zeros_like(actual_matrix, dtype=float)

    # 计算相对误差
    relative_error_matrix[mask] = np.abs(predicted_matrix[mask] - actual_matrix[mask]) / np.abs(actual_matrix[mask])

    # 对于实际值为0的位置，可以考虑设定相对误差为无穷大或特定值，这里设为NaN
    relative_error_matrix[~mask] = np.nan

    # 计算平均相对误差，忽略NaN值
    mean_relative_error = np.nanmean(relative_error_matrix)

    return relative_error_matrix, mean_relative_error


