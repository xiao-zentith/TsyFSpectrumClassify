import shutil
from Utils.MixUp import ExternalClass
from Utils.ZScore_norm import NormProcessor
from Utils.add_noise import NoiseProcessor


class AugmentData:
    def __init__(self, input_folder, output_folder, random_seed):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.random_seed = random_seed

    def process_data(self):
        # 第一步：MixUp 处理
        mixup_output_folder = f"{self.output_folder}/mixup"
        mixup_processor = ExternalClass(self.input_folder, mixup_output_folder, self.random_seed)
        mixup_processor.run_processing()

        # 第二步：添加噪声
        noise_output_folder = f"{self.output_folder}"
        noise_adder = NoiseProcessor(mixup_output_folder, noise_output_folder, self.random_seed)
        noise_adder.process_files_in_directory()

        # 删除 MixUp 的输出文件夹
        shutil.rmtree(mixup_output_folder)

        # 第三步：Z-Score 归一化
        zscore_normalizer = NormProcessor(noise_output_folder, self.output_folder)
        zscore_normalizer.process_folders()

        # 删除 噪声 添加的输出文件夹
        shutil.rmtree(noise_output_folder)


# 示例使用
if __name__ == "__main__":
    input_folder = r'C:\Users\xiao\Desktop\画大饼环节\data\dataset_K\dataset_TsyF'
    output_folder = r'C:\Users\xiao\Desktop\画大饼环节\data\dataset_K\1'
    random_seed = 42
    augmenter = AugmentData(input_folder, output_folder, random_seed)
    augmenter.process_data()



