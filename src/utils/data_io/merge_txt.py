import os


def merge_txt_files(folder_path, output_file):
    # 获取文件夹下所有的txt文件
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

    # 检查是否有txt文件
    if not txt_files:
        print("No .txt files found in the folder.")
        return

    # 用于存储所有文件的数据
    data_dicts = []

    # 读取每个文件并将数据存储在字典中
    for file_name in txt_files:
        file_path = os.path.join(folder_path, file_name)
        data_dict = {}
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line_number, line in enumerate(lines, start=1):
                stripped_line = line.strip()
                if not stripped_line:  # 跳过空行
                    continue
                parts = stripped_line.split()
                if len(parts) != 2:  # 检查是否恰好有两个值
                    print(
                        f"Warning: Line {line_number} in {file_name} does not contain exactly two values. Skipping this line.")
                    continue
                coordinate, intensity = parts
                data_dict[coordinate] = intensity
        data_dicts.append(data_dict)

    # 找出所有文件中共有的坐标
    common_coordinates = set(data_dicts[0].keys())
    for data_dict in data_dicts[1:]:
        common_coordinates.intersection_update(data_dict.keys())

    # 将数据写入输出文件
    with open(output_file, 'w') as out_file:
        # 写入表头
        header = "Coordinate\t" + "\t".join(txt_files) + "\n"
        out_file.write(header)

        # 写入数据行
        for coordinate in sorted(common_coordinates):
            row_parts = [coordinate]
            for data_dict in data_dicts:
                row_parts.append(data_dict[coordinate])
            row = "\t".join(row_parts) + "\n"
            out_file.write(row)


# 使用示例
folder_path = r'C:\Users\xiao\Desktop\academic_papers\data\em_spectrum'  # 替换为你的文件夹路径
output_file = 'merged_output.txt'  # 合并后文件名
merge_txt_files(folder_path, output_file)



