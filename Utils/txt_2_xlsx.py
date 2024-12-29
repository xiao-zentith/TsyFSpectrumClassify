import os
import pandas as pd
import openpyxl

def read_tsf_to_xlsx(source_folder, target_folder):
    # 确保目标文件夹存在
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 遍历源文件夹及其子文件夹中的所有文件
    for root, dirs, files in os.walk(source_folder):
        # 计算相对路径
        relative_path = os.path.relpath(root, source_folder)
        current_target_folder = os.path.join(target_folder, relative_path)

        # 确保当前目标文件夹存在
        if not os.path.exists(current_target_folder):
            os.makedirs(current_target_folder)

        for filename in files:
            if filename.endswith('.txt'):
                file_path = os.path.join(root, filename)

                with open(file_path, 'r') as txt_file:
                    lines = txt_file.readlines()

                # 解析第一行作为横坐标
                x_coords = lines[0].strip().split()[0:]
                x_coords = [float(x) for x in x_coords]

                # 初始化纵坐标列表和数据矩阵
                y_coords = []
                data_matrix = []

                # 解析剩余的行作为纵坐标和数据矩阵
                for line in lines[1:]:
                    parts = line.strip().split()
                    y_coords.append(parts[0])
                    data_matrix.append(parts[1:])

                y_coords = [float(y) for y in y_coords]

                # 创建DataFrame
                df = pd.DataFrame(data_matrix, columns=x_coords, index=y_coords)

                # 将所有列转换为数值类型，无法转换的设置为NaN
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                # 创建对应的xlsx文件名
                xlsx_filename = os.path.splitext(filename)[0] + '.xlsx'
                xlsx_file_path = os.path.join(current_target_folder, xlsx_filename)

                # 写入xlsx文件
                df.to_excel(xlsx_file_path, engine='openpyxl')

# 示例调用
source_folder = r'C:\Users\xiao\Desktop\Draw-flatbread\data\dataset_EEM\dataset_EEM'
target_folder = r'C:\Users\xiao\Desktop\Draw-flatbread\data\dataset_EEM\dataset'
read_tsf_to_xlsx(source_folder, target_folder)