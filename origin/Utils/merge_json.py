import json

def convert_to_dict(data_list):
    """将输入的列表转换为以(fold, inner_fold)为键的字典"""
    result = {}
    for item in data_list:
        key = (item['fold'], item['inner_fold'])
        result[key] = item
    return result

def merge_json_data(data1, data2):
    """合并两个 JSON 数据"""
    dict1 = convert_to_dict(data1)
    dict2 = convert_to_dict(data2)
    merged_dict = {}

    # 获取所有唯一的 (fold, inner_fold) 组合
    all_keys = dict1.keys() | dict2.keys()

    for key in all_keys:
        merged_item = {}
        if key in dict1 and key in dict2:
            item1 = dict1[key]
            item2 = dict2[key]
            merged_item['fold'] = item1['fold']
            merged_item['inner_fold'] = item1['inner_fold']
            merged_item['train'] = item1['train'] + item2['train']
            merged_item['validation'] = item1['validation'] + item2['validation']
            merged_item['test'] = item1['test'] + item2['test']
        elif key in dict1:
            merged_item = dict1[key].copy()
        else:
            merged_item = dict2[key].copy()
        merged_dict[key] = merged_item

    # 将字典转回列表，并按 fold 和 inner_fold 排序
    merged_list = list(merged_dict.values())
    merged_list.sort(key=lambda x: (x['fold'], x['inner_fold']))
    return merged_list

if __name__ == "__main__":
    # 读取两个 JSON 文件
    with open('merged_output.json', 'r') as f1:
        data1 = json.load(f1)
    with open('get_data_path("raw")', 'r') as f2:
        data2 = json.load(f2)

    # 合并数据
    merged_data = merge_json_data(data1, data2)

    # 写入合并后的数据
    with open('merged_output.json', 'w') as f:
        json.dump(merged_data, f, indent=4)