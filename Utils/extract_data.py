import os

# 定义一个函数来处理单个文件
def extract_nm_data(input_file, output_file):
    with open(input_file, 'r') as infile:
        lines = infile.readlines()

    nm_data_index = None
    for i, line in enumerate(lines):
        if "Data Points" in line:
            nm_data_index = i + 1
            break

    if nm_data_index:
        with open(output_file, 'w') as outfile:
            # 在第一行前添加一个空格
            # outfile.write('                ')
            for line in lines[nm_data_index:]:
                outfile.write(line.strip() + '\n')

# 定义一个函数来批量处理文件夹中的所有.txt文件
def process_folder(folder_path, output_folder):
    # 创建输出文件夹如果它不存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历文件夹中的所有文件和子文件夹
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):  # 如果是文件夹，递归处理
            new_output_folder = os.path.join(output_folder, item)
            if not os.path.exists(new_output_folder):
                os.makedirs(new_output_folder)
            process_folder(item_path, new_output_folder)
        elif item.endswith('.TXT'):  # 检查文件扩展名
            input_file = item_path
            output_file = os.path.join(output_folder, os.path.splitext(item)[0] + '_extracted.txt')
            extract_nm_data(input_file, output_file)
            print(f"Processed {item} into {output_file}")

# 调用函数，传入文件夹路径和输出文件夹路径
folder_path = r'C:\Users\xiao\Desktop\论文汇总\data\dataset\dataset_origin'  # 替换为你的文件夹路径
output_folder = r'C:\Users\xiao\Desktop\论文汇总\data\dataset\dataset_extract'  # 替换为输出文件夹的路径
process_folder(folder_path, output_folder)