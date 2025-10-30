import pandas as pd
import os

# 设置文件夹路径和输出文件夹
input_folder = r'C:\Users\xiao\Desktop\academic_papers\data\dataset_EEM\dataset\hpts'  # 替换为你实际的输入文件夹路径
output_folder = r'C:\Users\xiao\Desktop\academic_papers\data\dataset_EEM\dataset_460\HPTS'     # 输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 遍历文件夹下所有xlsx文件
for filename in os.listdir(input_folder):
    if filename.endswith('.xlsx'):
        file_path = os.path.join(input_folder, filename)

        # 读取Excel文件（不设置表头）
        df = pd.read_excel(file_path, header=None)

        # 提取实际光谱矩阵（去除首行首列）
        matrix = df.iloc[1:, 1:]

        # 提取倒数第五列作为Ex=460数据
        ex460_column = matrix.iloc[:, -5]

        # 发射波长在第一列
        emission_wavelengths = df.iloc[1:, 0]

        # 组合为DataFrame
        result = pd.DataFrame({
            'Emission_Wavelength': emission_wavelengths,
            'Intensity_at_Ex460': ex460_column
        })

        # 构建输出文件路径并保存
        output_file_path = os.path.join(output_folder, f'{os.path.splitext(filename)[0]}_Ex460.xlsx')
        result.to_excel(output_file_path, index=False)

        print(f"Processed and saved: {output_file_path}")