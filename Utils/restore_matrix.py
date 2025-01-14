import os
import pandas as pd


def process_excel(input_file, output_folder_c1, output_folder_c2):
    # 创建输出文件夹
    if not os.path.exists(output_folder_c1):
        os.makedirs(output_folder_c1)

    if not os.path.exists(output_folder_c2):
        os.makedirs(output_folder_c2)

    # 读取输入文件中的表格
    xls = pd.ExcelFile(input_file)
    fmax_df = pd.read_excel(xls, 'FMax')
    c1_df = pd.read_excel(xls, 'C1')
    c2_df = pd.read_excel(xls, 'C2')

    # 获取C1和C2的最大值（不包括第一列）
    c1_max_value = c1_df.iloc[:, 1:].max().max()
    c2_max_value = c2_df.iloc[:, 1:].max().max()

    # 处理每一行数据
    for index, row in fmax_df.iterrows():
        file_name = row.iloc[0]
        fmax1 = row['FMax1']
        fmax2 = row['FMax2']

        weight1 = fmax1 / c1_max_value
        weight2 = fmax2 / c2_max_value

        # 计算新的矩阵（不包括第一列）
        new_c1_matrix_values = weight1 * c1_df.iloc[:, 1:]
        new_c2_matrix_values = weight2 * c2_df.iloc[:, 1:]

        # 将第一列重新添加到新的矩阵中
        new_c1_matrix = pd.concat([c1_df.iloc[:, 0], new_c1_matrix_values], axis=1)
        new_c2_matrix = pd.concat([c2_df.iloc[:, 0], new_c2_matrix_values], axis=1)

        # 写入新的Excel文件
        with pd.ExcelWriter(os.path.join(output_folder_c1, f'{file_name}.xlsx')) as writer:
            new_c1_matrix.to_excel(writer, sheet_name='Sheet1', index=False)

        with pd.ExcelWriter(os.path.join(output_folder_c2, f'{file_name}.xlsx')) as writer:
            new_c2_matrix.to_excel(writer, sheet_name='Sheet1', index=False)


# 使用示例
input_file = r'C:\Users\xiao\Desktop\Draw-flatbread\data\dataset_EEM\EEM_xlsx\FITC + hpts_Component\2025-01-12 2-component result.xlsx'  # 替换为你的输入文件名
output_folder_c1 = r'C:\Users\xiao\Desktop\Draw-flatbread\data\dataset_EEM\EEM_xlsx\FITC + hpts_Component\Component1'  # 替换为你想要保存Weighted_C1文件的文件夹名
output_folder_c2 = r'C:\Users\xiao\Desktop\Draw-flatbread\data\dataset_EEM\EEM_xlsx\FITC + hpts_Component\Component2'  # 替换为你想要保存Weighted_C2文件的文件夹名
process_excel(input_file, output_folder_c1, output_folder_c2)



