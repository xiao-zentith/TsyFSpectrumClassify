import os
import pandas as pd


def check_nan_in_excel_files(folder_path):
    # 获取文件夹中的所有文件名
    file_names = os.listdir(folder_path)

    # 过滤出所有的.xlsx文件
    excel_files = [file for file in file_names if file.endswith('.xlsx')]

    # 遍历每一个excel文件
    for excel_file in excel_files:
        file_path = os.path.join(folder_path, excel_file)

        try:
            # 读取excel文件
            df = pd.read_excel(file_path)

            # 检查是否有nan值
            has_nan = df.isnull().values.any()

            if has_nan:
                nan_positions = df[df.isna()].stack().index.tolist()
                print(f"File: {excel_file}, Contains NaN: True")
                print("NaN Positions:")
                for pos in nan_positions:
                    row_idx, col_name = pos
                    print(f"Row: {row_idx + 1}, Column: {col_name}")
            else:
                print(f"File: {excel_file}, Contains NaN: False")

        except Exception as e:
            print(f"Error reading file {excel_file}: {e}")


# 使用示例
folder_path = '/home/asus515/PycharmProjects/TsyFSpectrumClassify_remote/dataset/dataset_target/Fish/Component4'  # 替换为你的文件夹路径
check_nan_in_excel_files(folder_path)



