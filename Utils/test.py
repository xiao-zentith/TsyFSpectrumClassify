import numpy as np

def read_spectra_matrix(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Extract x coordinates from the first line
    x_coords = list(map(float, lines[0].strip().split()))

    # Extract y coordinates and matrix values
    data = []
    for line in lines[1:]:
        row = list(map(float, line.strip().split()))
        data.append(row)

    y_coords = [row[0] for row in data]
    matrix_values = [row[1:] for row in data]

    return x_coords, y_coords, np.array(matrix_values)


def apply_mixup(matrix1, matrix2, lamb=0.5):
    """
    使用Mixup方法生成新的矩阵

    :param matrix1: 第一个矩阵 (numpy array)
    :param matrix2: 第二个矩阵 (numpy array)
    :param lamb: Mixup系数 (默认为0.5)
    :return: 新生成的矩阵 (numpy array)
    """
    # 确保输入矩阵具有相同的形状
    if matrix1.shape != matrix2.shape:
        raise ValueError("两个矩阵必须具有相同的形状")

    # 应用Mixup方法
    mixed_matrix = lamb * matrix1 + (1 - lamb) * matrix2
    return mixed_matrix


# 示例使用
if __name__ == "__main__":
    # 定义文件路径
    file_path1 = r'C:\Users\xiao\Desktop\画大饼环节\data\dataset_K\dataset_mixup\C6\C10_extracted.txt'
    file_path2 = r'C:\Users\xiao\Desktop\画大饼环节\data\dataset_K\dataset_mixup\C6\C40_extracted.txt'

    # 读取矩阵数据
    x_coords1, y_coords1, matrix1 = read_spectra_matrix(file_path1)
    x_coords2, y_coords2, matrix2 = read_spectra_matrix(file_path2)

    # 检查横坐标和纵坐标是否相同
    if not np.allclose(x_coords1, x_coords2) or not np.allclose(y_coords1, y_coords2):
        raise ValueError("两个矩阵的横坐标或纵坐标不匹配")

    # 应用Mixup方法生成新的矩阵
    new_matrix = apply_mixup(matrix1, matrix2, lamb=0.5)

    # 打印新的矩阵
    print("新的混合矩阵:")
    print(new_matrix)

    # 如果需要保存新的矩阵到文件
    output_file_path = '1.txt'
    with open(output_file_path, 'w') as file:
        # 写入横坐标
        file.write(' '.join(map(str, x_coords1)) + '\n')
        # 写入纵坐标和矩阵值
        for y_coord, row in zip(y_coords1, new_matrix):
            file.write(f'{y_coord} ' + ' '.join(map(str, row)) + '\n')

    print(f"新的混合矩阵已保存到 {output_file_path}")



