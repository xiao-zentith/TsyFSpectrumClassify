import os
import pandas as pd


def process_excel(input_file, output_folder_c1, output_folder_c2, output_folder_c3, output_folder_c4):
    # 创建输出文件夹
    if not os.path.exists(output_folder_c1):
        os.makedirs(output_folder_c1)

    if not os.path.exists(output_folder_c2):
        os.makedirs(output_folder_c2)

    if not os.path.exists(output_folder_c3):
        os.makedirs(output_folder_c3)

    if not os.path.exists(output_folder_c3):
        os.makedirs(output_folder_c3)

    # 读取输入文件中的表格
    xls = pd.ExcelFile(input_file)
    fmax_df = pd.read_excel(xls, 'FMax')
    c1_df = pd.read_excel(xls, 'C1')
    c2_df = pd.read_excel(xls, 'C2')
    c3_df = pd.read_excel(xls, 'C3')
    c4_df = pd.read_excel(xls, 'C4')

    # 获取C1和C2的最大值（不包括第一列）
    c1_max_value = c1_df.iloc[:, 1:].max().max()
    c2_max_value = c2_df.iloc[:, 1:].max().max()
    c3_max_value = c3_df.iloc[:, 1:].max().max()
    c4_max_value = c4_df.iloc[:, 1:].max().max()

    # 处理每一行数据
    for index, row in fmax_df.iterrows():
        file_name = row.iloc[0]
        fmax1 = row['FMax1']
        fmax2 = row['FMax2']
        fmax3 = row['FMax3']
        fmax4 = row['FMax4']

        weight1 = fmax1 / c1_max_value
        weight2 = fmax2 / c2_max_value
        weight3 = fmax3 / c3_max_value
        weight4 = fmax4 / c4_max_value

        # 计算新的矩阵（不包括第一列）
        new_c1_matrix_values = weight1 * c1_df.iloc[:, 1:]
        new_c2_matrix_values = weight2 * c2_df.iloc[:, 1:]
        new_c3_matrix_values = weight3 * c3_df.iloc[:, 1:]
        new_c4_matrix_values = weight4 * c4_df.iloc[:, 1:]

        # 将第一列重新添加到新的矩阵中
        new_c1_matrix = pd.concat([c1_df.iloc[:, 0], new_c1_matrix_values], axis=1)
        new_c2_matrix = pd.concat([c2_df.iloc[:, 0], new_c2_matrix_values], axis=1)
        new_c3_matrix = pd.concat([c3_df.iloc[:, 0], new_c3_matrix_values], axis=1)
        new_c4_matrix = pd.concat([c4_df.iloc[:, 0], new_c4_matrix_values], axis=1)

        # 写入新的Excel文件
        with pd.ExcelWriter(os.path.join(output_folder_c1, f'{file_name}.xlsx')) as writer:
            new_c1_matrix.to_excel(writer, sheet_name='Sheet1', index=False)

        with pd.ExcelWriter(os.path.join(output_folder_c2, f'{file_name}.xlsx')) as writer:
            new_c2_matrix.to_excel(writer, sheet_name='Sheet1', index=False)

        with pd.ExcelWriter(os.path.join(output_folder_c3, f'{file_name}.xlsx')) as writer:
            new_c3_matrix.to_excel(writer, sheet_name='Sheet1', index=False)

        with pd.ExcelWriter(os.path.join(output_folder_c4, f'{file_name}.xlsx')) as writer:
            new_c4_matrix.to_excel(writer, sheet_name='Sheet1', index=False)


# 使用示例
input_file = r'C:\Users\xiao\Desktop\Draw-flatbread\data\open_dataset\fish\EEM\2025-05-24 4-component result.xlsx'  # 替换为你的输入文件名
output_folder_c1 = r'C:\Users\xiao\Desktop\Draw-flatbread\data\open_dataset\fish\EEM\target\target1'  # 替换为你想要保存Weighted_C1文件的文件夹名
output_folder_c2 = r'C:\Users\xiao\Desktop\Draw-flatbread\data\open_dataset\fish\EEM\target\target2'  # 替换为你想要保存Weighted_C2文件的文件夹名
output_folder_c3 = r'C:\Users\xiao\Desktop\Draw-flatbread\data\open_dataset\fish\EEM\target\target3'
output_folder_c4 = r'C:\Users\xiao\Desktop\Draw-flatbread\data\open_dataset\fish\EEM\target\target4'
process_excel(input_file, output_folder_c1, output_folder_c2, output_folder_c3, output_folder_c4)



