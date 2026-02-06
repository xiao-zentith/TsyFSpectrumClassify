import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from pathlib import Path


def read_eem_from_excel(file_path):
    """读取EEM Excel文件"""
    df = pd.read_excel(file_path, header=None)
    excitation_orig = df.iloc[0, 1:].values.astype(float)
    emission = df.iloc[1:, 0].values.astype(float)
    original_data = df.iloc[1:, 1:].values.T.astype(float)  # 转置为激发×发射
    return excitation_orig, emission, original_data


def interpolate_excitation(data_orig, excitation_orig, new_size=63):
    """执行激发波长维度插值"""
    excitation_new = np.linspace(excitation_orig.min(), excitation_orig.max(), new_size)
    interpolated = np.zeros((new_size, data_orig.shape[1]))

    for j in range(data_orig.shape[1]):
        f = interp1d(excitation_orig, data_orig[:, j], kind='linear',
                     fill_value="extrapolate")
        interpolated[:, j] = f(excitation_new)

    return excitation_new, interpolated


def process_single_file(input_path, output_dir):
    """处理单个文件"""
    try:
        # 读取数据
        ex_orig, em_orig, data_orig = read_eem_from_excel(input_path)

        # 执行插值
        ex_new, data_interp = interpolate_excitation(data_orig, ex_orig)

        # 构建输出DataFrame
        output_df = pd.DataFrame(data_interp.T,  # 转置回发射×激发
                                 columns=ex_new,
                                 index=em_orig)
        output_df.reset_index(inplace=True)
        output_df.rename(columns={'index': 'Em/Ex'}, inplace=True)

        # 设置输出路径
        output_path = Path(output_dir) / Path(input_path).name

        # 保存文件
        with pd.ExcelWriter(output_path) as writer:
            # 写入标题行
            pd.DataFrame([[''] + list(ex_new)]).to_excel(
                writer,
                header=False,
                index=False
            )
            # 写入数据（从第二行开始）
            output_df.to_excel(
                writer,
                startrow=1,
                index=False,
                header=False
            )

        print(f"成功处理：{Path(input_path).name}")

    except Exception as e:
        print(f"处理失败：{Path(input_path).name} - 错误信息：{str(e)}")


def batch_process(input_dir, output_dir):
    """批量处理文件夹"""
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 遍历所有xlsx文件
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.xlsx'):
            input_path = Path(input_dir) / file_name
            process_single_file(input_path, output_dir)


if __name__ == "__main__":
    # 配置路径
    input_directory = '../dataset/dataset_preprocess/C6 + FITC'
    output_directory = "../dataset/dataset_resized/C6 + FITC"

    # 执行批处理
    batch_process(input_directory, output_directory)
    print("\n处理完成！请检查输出文件夹。")