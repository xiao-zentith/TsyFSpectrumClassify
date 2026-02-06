import numpy as np

def pearson_corr_matrix(matrix1, matrix2):
    # 确保输入是二维数组
    matrix1 = np.atleast_2d(matrix1)
    matrix2 = np.atleast_2d(matrix2)

    # 检查两个矩阵是否有相同的形状
    if matrix1.shape != matrix2.shape:
        raise ValueError("两个矩阵的形状必须相同")

    # 计算每个矩阵各列的均值
    mean_matrix1 = np.mean(matrix1, axis=0)
    mean_matrix2 = np.mean(matrix2, axis=0)

    # 计算去均值后的矩阵
    demeaned_matrix1 = matrix1 - mean_matrix1
    demeaned_matrix2 = matrix2 - mean_matrix2

    # 计算协方差矩阵
    covariance = np.sum(demeaned_matrix1 * demeaned_matrix2, axis=0) / (matrix1.shape[0] - 1)

    # 计算标准差
    std_matrix1 = np.std(matrix1, axis=0, ddof=1)
    std_matrix2 = np.std(matrix2, axis=0, ddof=1)

    # 避免除以零的情况
    std_matrix1[std_matrix1 == 0] = 1
    std_matrix2[std_matrix2 == 0] = 1

    # 计算皮尔逊相关系数
    pearson_corr = covariance / (std_matrix1 * std_matrix2)

    return pearson_corr

# 示例数据
matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[5, 6], [7, 8]])

# 调用函数计算皮尔逊相关系数
corr = pearson_corr_matrix(matrix1, matrix2)
print("皮尔逊相关系数:", corr)