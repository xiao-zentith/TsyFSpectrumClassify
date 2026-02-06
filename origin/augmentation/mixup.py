import os
import numpy as np
import shutil


class MatrixMixupProcessor:
    def __init__(self, input_folder, output_folder, random_seed=None):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.random_seed = random_seed
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

    @staticmethod
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

    @staticmethod
    def mixup_matrices(matrix_a, matrix_b, lam=None):
        """
        通过mixup方法生成一个新的矩阵
        :param matrix_a: 第一个原始矩阵
        :param matrix_b: 第二个原始矩阵
        :param lam: 混合系数，默认为None时会随机生成一个在0到1之间的数
        :return: 新生成的混合矩阵
        """
        if lam is None:
            lam = np.random.uniform(0, 1)

        mixed_matrix = lam * matrix_a + (1 - lam) * matrix_b
        return np.round(mixed_matrix, 2), lam

    @staticmethod
    def save_mixed_matrix_to_file(x_coords, y_coords, mixed_matrix, output_file_path):
        """
        将混合矩阵保存到TXT文件中
        :param x_coords: 横坐标数组
        :param y_coords: 纵坐标数组
        :param mixed_matrix: 混合矩阵
        :param output_file_path: 输出文件路径
        """
        with open(output_file_path, 'w') as file:
            # 写入横坐标
            file.write(' '.join(map(str, x_coords)) + '\n')

            # 写入纵坐标和数据矩阵
            for y, row in zip(y_coords, mixed_matrix):
                file.write(f'{y} {" ".join(map(str, row))}\n')

    @staticmethod
    def copy_original_files(input_label_folder, output_label_folder):
        """
        将原始文件复制到输出文件夹
        :param input_label_folder: 输入标签文件夹路径
        :param output_label_folder: 输出标签文件夹路径
        """
        txt_files = [f for f in os.listdir(input_label_folder) if f.endswith('.txt')]

        for txt_file in txt_files:
            src_file_path = os.path.join(input_label_folder, txt_file)
            dst_file_path = os.path.join(output_label_folder, txt_file)
            shutil.copy(src_file_path, dst_file_path)
            # print(f"原始文件 {src_file_path} 已复制到 {dst_file_path}")

    def process_label_folder(self, input_label_folder, output_label_folder):
        """
        处理单个标签文件夹中的数据
        :param input_label_folder: 输入标签文件夹路径
        :param output_label_folder: 输出标签文件夹路径
        """
        txt_files = [f for f in os.listdir(input_label_folder) if f.endswith('.txt')]

        if len(txt_files) < 2:
            print(f"警告: 标签文件夹 {input_label_folder} 中少于两个TXT文件，跳过处理")
            return

        # 创建输出文件夹
        os.makedirs(output_label_folder, exist_ok=True)

        # 复制原始文件
        self.copy_original_files(input_label_folder, output_label_folder)

        # 获取所有文件路径
        file_paths = [os.path.join(input_label_folder, f) for f in txt_files]

        # 记录已处理的组合
        processed_pairs = set()

        for i in range(len(file_paths)):
            for j in range(i + 1, len(file_paths)):
                if (i, j) not in processed_pairs and (j, i) not in processed_pairs:
                    file_path_a = file_paths[i]
                    file_path_b = file_paths[j]

                    # 读取两个矩阵数据
                    x_coords_a, y_coords_a, matrix_a = self.read_matrix_from_file(file_path_a)
                    x_coords_b, y_coords_b, matrix_b = self.read_matrix_from_file(file_path_b)

                    # 检查两个矩阵是否具有相同的维度和横纵坐标
                    if not np.array_equal(x_coords_a, x_coords_b) or not np.array_equal(y_coords_a, y_coords_b):
                        print(f"警告: 文件 {file_path_a} 和 {file_path_b} 的横纵坐标或维度不同，跳过处理")
                        continue

                    # 进行mixup处理
                    mixed_matrix, lam = self.mixup_matrices(matrix_a, matrix_b)

                    # 设置输出文件路径
                    output_file_name = f'mixed_{os.path.basename(file_path_a)}_with_{os.path.basename(file_path_b)}.txt'
                    output_file_path = os.path.join(output_label_folder, output_file_name)

                    # 保存混合矩阵到文件
                    self.save_mixed_matrix_to_file(x_coords_a, y_coords_a, mixed_matrix, output_file_path)

                    # 记录已处理的组合
                    processed_pairs.add((i, j))
                    # print(f"混合矩阵已保存到 {output_file_path}，使用的λ值为 {lam}")

    def main(self):
        """
        主函数，处理整个输入文件夹
        """
        # 创建输出文件夹
        os.makedirs(self.output_folder, exist_ok=True)

        # 遍历输入文件夹中的每个子文件夹
        for label in os.listdir(self.input_folder):
            input_label_folder = os.path.join(self.input_folder, label)
            output_label_folder = os.path.join(self.output_folder, label)

            if os.path.isdir(input_label_folder):
                self.process_label_folder(input_label_folder, output_label_folder)


class ExternalClass:
    def __init__(self, input_folder, output_folder, random_seed=None):
        self.processor = MatrixMixupProcessor(input_folder, output_folder, random_seed)

    def run_processing(self):
        self.processor.main()


# 示例调用
if __name__ == "__main__":
    input_folder = r'C:\Users\xiao\Desktop\画大饼环节\data\dataset_EEM\dataset_EEM'
    output_folder = r'C:\Users\xiao\Desktop\画大饼环节\data\dataset_EEM\EEM_mixup'
    random_seed = 42  # 设置随机种子以保证结果可重复
    external_instance = ExternalClass(input_folder, output_folder, random_seed)
    external_instance.run_processing()



