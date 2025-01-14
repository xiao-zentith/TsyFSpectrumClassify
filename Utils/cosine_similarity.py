import numpy as np


def cosine_similarity(matrix_a, matrix_b):
    # 将矩阵展平成向量
    vector_a = matrix_a.flatten()
    vector_b = matrix_b.flatten()

    # 计算点积
    dot_product = np.dot(vector_a, vector_b)

    # 计算向量的模
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    # 计算余弦相似度
    if norm_a == 0 or norm_b == 0:
        return 0.0  # 如果其中一个向量为零，则相似度为0

    similarity = dot_product / (norm_a * norm_b)
    return similarity

