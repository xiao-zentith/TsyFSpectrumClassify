import os
import pandas as pd


def modify_first_column_name_in_xlsx_files(directory):
    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        if filename.endswith('.xlsx'):
            file_path = os.path.join(directory, filename)

            # 读取Excel文件
            df = pd.read_excel(file_path)

            # 检查DataFrame是否为空
            if not df.empty:
                # 获取当前的第一列名
                current_first_column_name = df.columns[0]

                # 创建一个新的列名列表，将第一列名替换为0
                new_columns = list(df.columns)
                new_columns[0] = 0

                # 设置新的列名
                df.columns = new_columns

                # 将修改后的内容写回Excel文件
                df.to_excel(file_path, index=False)
                print(f'Modified: {file_path}')
            else:
                print(f'Skipped empty file: {file_path}')


# 使用示例：请将'/path/to/your/directory'替换为实际的目录路径
modify_first_column_name_in_xlsx_files(r'C:\Users\xiao\Desktop\Draw-flatbread\data\dataset_EEM\EEM_xlsx\FITC + hpts_Component\Component2')



