import os

def remove_txt_from_filenames(directory):
    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        if filename.endswith('.xlsx') and '.txt' in filename:
            # 构造新的文件名，去掉'txt'
            new_filename = filename.replace('.txt', '')
            old_file_path = os.path.join(directory, filename)
            new_file_path = os.path.join(directory, new_filename)

            # 重命名文件
            os.rename(old_file_path, new_file_path)
            print(f'Renamed: {old_file_path} -> {new_file_path}')


# 使用示例：请将'/path/to/your/directory'替换为实际的目录路径
remove_txt_from_filenames(get_data_path("raw"))
