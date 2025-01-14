import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.preprocessing import StandardScaler
#
# from Utils.read_npz import read_npz_file
#
#
# def read_spectra_matrix(file_path):
#     with open(file_path, 'r') as file:
#         lines = file.readlines()
#
#     # Extract x coordinates from the first line
#     x_coords = list(map(float, lines[0].strip().split()))
#
#     # Extract y coordinates and matrix values
#     data = []
#     for line in lines[1:]:
#         row = list(map(float, line.strip().split()))
#         data.append(row)
#
#     y_coords = [row[0] for row in data]
#     matrix_values = [row[1:] for row in data]
#
#     return x_coords, y_coords, np.array(matrix_values)


def calculate_cosine_similarity(matrix1, matrix2):
    """
    计算两个矩阵按行的余弦相似度

    :param matrix1: 第一个矩阵 (numpy array)
    :param matrix2: 第二个矩阵 (numpy array)
    :return: 包含每对行之间余弦相似度的一维数组
    """
    # 确保输入矩阵具有相同的形状
    if matrix1.shape != matrix2.shape:
        raise ValueError("两个矩阵必须具有相同的形状")

    # 初始化存储结果的列表
    similarities = []

    # 逐行计算余弦相似度
    for row1, row2 in zip(matrix1, matrix2):
        similarity = cosine_similarity([row1], [row2])
        similarities.append(similarity[0][0])

    return np.array(similarities)


# # 示例使用
# if __name__ == "__main__":
#     # 定义文件路径
#     file_path1 = r'C:\Users\xiao\Desktop\画大饼环节\data\dataset_K\dataset_mixup\C6\mixed_C15_extracted.txt_with_C25_extracted.txt.txt'
#     file_path2 = r'C:\Users\xiao\Desktop\画大饼环节\data\dataset_K\dataset_mixup\C6\C20_extracted.txt'
#
#     # # 读取矩阵数据
#     # x_coords1, y_coords1, matrix1 = read_spectra_matrix(file_path1)
#     # x_coords2, y_coords2, matrix2 = read_spectra_matrix(file_path2)
#     #
#     # # 检查横坐标和纵坐标是否相同
#     # if not np.allclose(x_coords1, x_coords2) or not np.allclose(y_coords1, y_coords2):
#     #     raise ValueError("两个矩阵的横坐标或纵坐标不匹配")
#
#     matrix1, matrix2 = read_npz_file(r'/dataset/dataset_result/C6 + FITC/fold_2\test_sample_1_data.npz')
#
#     # 标准化矩阵
#     scaler = StandardScaler()
#     matrix1_standardized = scaler.fit_transform(matrix1)
#     matrix2_standardized = scaler.transform(matrix2)
#
#     # 计算余弦相似度
#     result = calculate_cosine_similarity(matrix1_standardized, matrix2_standardized)
#
#     # 计算并打印余弦相似度的平均值
#     average_similarity = np.mean(result)
#     print(f"余弦相似度的平均值: {average_similarity}")
#


