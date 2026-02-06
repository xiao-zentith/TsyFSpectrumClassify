import os
import json
import sys
from pathlib import Path
from itertools import combinations
from sklearn.model_selection import train_test_split, KFold

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from src.utils.path_manager import PathManager

class DatasetInfoGenerator:
    """数据集信息生成器 - 集成路径管理器的优化版本"""
    
    def __init__(self, config_path=None):
        """初始化生成器"""
        self.path_manager = PathManager(config_path)
    
    def list_xlsx_files(self, folder_path):
        """列出文件夹中的Excel文件"""
        folder_path = Path(folder_path)
        if not folder_path.exists():
            return []
        return [f.name for f in folder_path.glob("*.xlsx")]
    
    def generate_mixup_filenames(self, file_pairs):
        """生成mixup增强文件名"""
        augmented_names = []
        for file1, file2 in file_pairs:
            base_name1 = os.path.splitext(file1)[0]
            base_name2 = os.path.splitext(file2)[0]
            mixed_name = f"mixed_{base_name1}_with_{base_name2}.xlsx"
            noise_mixed_name = f"noise_mixed_{base_name1}_with_{base_name2}.xlsx"
            augmented_names.append(mixed_name)
            augmented_names.append(noise_mixed_name)
        return augmented_names
    
    def simple_data_split(self, xlsx_files, data_split):
        """简单数据分割"""
        train_val_files, test_files = train_test_split(xlsx_files, test_size=data_split["test"], random_state=42)
        train_files, val_files = train_test_split(train_val_files,
                                                  test_size=(data_split["val"] / (data_split["train"] + data_split["val"])),
                                                  random_state=42)
        return train_files, val_files, test_files
    
    def nested_cross_validation(self, is_mixup, xlsx_files, processed_folder, target_folders):
        """生成嵌套交叉验证的数据集"""
        outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
        dataset_info_list = []

        for fold_idx, (train_val_indices, test_indices) in enumerate(outer_cv.split(xlsx_files)):
            train_val_files = [xlsx_files[i] for i in train_val_indices]
            test_files = [xlsx_files[i] for i in test_indices]

            inner_cv = KFold(n_splits=4, shuffle=True, random_state=42)

            for inner_fold_idx, (train_indices, val_indices) in enumerate(inner_cv.split(train_val_files)):
                train_files = [train_val_files[i] for i in train_indices]
                val_files = [train_val_files[i] for i in val_indices]

                # Generate mixup pairs for training set
                train_file_pairs = list(combinations(train_files, 2))
                augmented_files = self.generate_mixup_filenames(train_file_pairs)

                # Include original train files in the augmented files
                if is_mixup:
                    all_train_files = train_files + augmented_files
                else:
                    all_train_files = train_files

                # Record the paths along with their corresponding target files
                dataset_info = {
                    "fold": fold_idx,
                    "inner_fold": inner_fold_idx,
                    "train": [{"input": os.path.join(processed_folder, tf),
                               "targets": [os.path.join(tfolder, os.path.basename(os.path.splitext(tf)[0] + ".xlsx")) for
                                           tfolder in target_folders]} for tf in all_train_files],
                    "validation": [{"input": os.path.join(processed_folder, vf),
                                    "targets": [os.path.join(tfolder, os.path.basename(vf)) for tfolder in target_folders]}
                                   for vf in val_files],
                    "test": [{"input": os.path.join(processed_folder, tef),
                              "targets": [os.path.join(tfolder, os.path.basename(tef)) for tfolder in target_folders]} for
                             tef in test_files]
                }

                dataset_info_list.append(dataset_info)

        return dataset_info_list
    
    def generate_from_config(self, config_file, output_file=None):
        """从配置文件生成数据集信息"""
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 使用路径管理器获取路径
        try:
            input_folder = config.get("dataset_raw") or str(self.path_manager.get_data_path("raw"))
            processed_folder = config.get("dataset_processed") or str(self.path_manager.get_data_path("processed"))
            
            # 处理目标文件夹
            target_folders = []
            for i in range(1, 5):  # 支持最多4个目标文件夹
                target_key = f"dataset_target{i}"
                if target_key in config:
                    target_folders.append(config[target_key])
            
            if not target_folders:
                target_folders = [str(self.path_manager.get_data_path("target"))]
                
        except Exception as e:
            print(f"路径配置错误: {e}")
            return False

        data_split = config.get("data_split", {"train": 0.7, "val": 0.15, "test": 0.15})
        is_cross_validation = config.get("is_cross_validation", True)
        is_mixup = config.get("is_mixup", False)
        
        if output_file is None:
            output_file = "dataset_info.json"
        
        xlsx_files = self.list_xlsx_files(input_folder)
        
        if not xlsx_files:
            print(f"在 {input_folder} 中未找到Excel文件")
            return False

        if is_cross_validation:
            dataset_info_list = self.nested_cross_validation(is_mixup, xlsx_files, processed_folder, target_folders)
        else:
            train_files, val_files, test_files = self.simple_data_split(xlsx_files, data_split)

            # Generate mixup pairs for training set
            train_file_pairs = list(combinations(train_files, 2))
            augmented_files = self.generate_mixup_filenames(train_file_pairs)

            # Include original train files in the augmented files
            if is_mixup:
                all_train_files = train_files + augmented_files
            else:
                all_train_files = train_files

            # Record the paths along with their corresponding target files
            dataset_info = {
                "train": [{"input": os.path.join(processed_folder, tf),
                           "targets": [os.path.join(tfolder, os.path.basename(os.path.splitext(tf)[0] + ".xlsx")) for
                                       tfolder in target_folders]} for tf in all_train_files],
                "validation": [
                    {"input": os.path.join(processed_folder, vf),
                     "targets": [os.path.join(tfolder, os.path.basename(vf)) for tfolder in target_folders]}
                    for vf in val_files],
                "test": [
                    {"input": os.path.join(processed_folder, tef),
                     "targets": [os.path.join(tfolder, os.path.basename(tef)) for tfolder in target_folders]}
                    for tef in test_files]
            }

            dataset_info_list = [dataset_info]

        # Write to JSON file
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(dataset_info_list, f, indent=4, ensure_ascii=False)
            print(f"成功生成数据集信息文件: {output_file}")
            return True
        except Exception as e:
            print(f"写入文件时出错: {e}")
            return False


# 保持向后兼容的函数
def list_xlsx_files(folder_path):
    return [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]

def generate_mixup_filenames(file_pairs):
    augmented_names = []
    for file1, file2 in file_pairs:
        base_name1 = os.path.splitext(file1)[0]
        base_name2 = os.path.splitext(file2)[0]
        mixed_name = f"mixed_{base_name1}_with_{base_name2}.xlsx"
        noise_mixed_name = f"noise_mixed_{base_name1}_with_{base_name2}.xlsx"
        mixed_path = mixed_name
        noise_mixed_path = noise_mixed_name
        augmented_names.append(mixed_path)
        augmented_names.append(noise_mixed_path)
    return augmented_names


def simple_data_split(xlsx_files, data_split):
    train_val_files, test_files = train_test_split(xlsx_files, test_size=data_split["test"], random_state=42)
    train_files, val_files = train_test_split(train_val_files,
                                              test_size=(data_split["val"] / (data_split["train"] + data_split["val"])),
                                              random_state=42)
    return train_files, val_files, test_files


#生成嵌套交叉验证的数据集
def nested_cross_validation(is_mixup, xlsx_files, processed_folder, target_folders):
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    dataset_info_list = []

    for fold_idx, (train_val_indices, test_indices) in enumerate(outer_cv.split(xlsx_files)):
        train_val_files = [xlsx_files[i] for i in train_val_indices]
        test_files = [xlsx_files[i] for i in test_indices]

        inner_cv = KFold(n_splits=4, shuffle=True, random_state=42)

        for inner_fold_idx, (train_indices, val_indices) in enumerate(inner_cv.split(train_val_files)):
            train_files = [train_val_files[i] for i in train_indices]
            val_files = [train_val_files[i] for i in val_indices]

            # Generate mixup pairs for training set
            train_file_pairs = list(combinations(train_files, 2))
            augmented_files = generate_mixup_filenames(train_file_pairs)

            # Include original train files in the augmented files
            if is_mixup:
                all_train_files = train_files + augmented_files
            else:
                all_train_files = train_files

            # Record the paths along with their corresponding target files
            dataset_info = {
                "fold": fold_idx,
                "inner_fold": inner_fold_idx,
                "train": [{"input": os.path.join(processed_folder, tf),
                           "targets": [os.path.join(tfolder, os.path.basename(os.path.splitext(tf)[0] + ".xlsx")) for
                                       tfolder in target_folders]} for tf in all_train_files],
                "validation": [{"input": os.path.join(processed_folder, vf),
                                "targets": [os.path.join(tfolder, os.path.basename(vf)) for tfolder in target_folders]}
                               for vf in val_files],
                "test": [{"input": os.path.join(processed_folder, tef),
                          "targets": [os.path.join(tfolder, os.path.basename(tef)) for tfolder in target_folders]} for
                         tef in test_files]
            }

            dataset_info_list.append(dataset_info)

    return dataset_info_list


def main(config_file=None):
    """主函数 - 使用优化后的生成器"""
    try:
        generator = DatasetInfoGenerator()
        
        if config_file is None:
            # 尝试使用默认配置文件
            config_path = generator.path_manager.get_path("configs", "main")
            config_file = config_path / "config.json"
            
            if not config_file.exists():
                print("未找到配置文件，请指定配置文件路径")
                return False
        
        return generator.generate_from_config(config_file)
        
    except Exception as e:
        print(f"生成过程中出错: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="数据集信息生成器")
    parser.add_argument("--config", "-c", help="配置文件路径")
    parser.add_argument("--output", "-o", help="输出文件路径")
    
    args = parser.parse_args()
    
    if args.config:
        main(args.config)
    else:
        main()



