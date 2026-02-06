#!/usr/bin/env python3
"""
批量重命名预处理文件，去掉文件名中的.txt部分
将 xxx.txt.xlsx 重命名为 xxx.xlsx
"""

import os
import sys
from pathlib import Path

def rename_preprocess_files(base_dir):
    """批量重命名预处理文件"""
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"错误: 目录不存在: {base_path}")
        return False
    
    # 查找所有需要重命名的文件
    files_to_rename = list(base_path.rglob("*.txt.xlsx"))
    
    if not files_to_rename:
        print("没有找到需要重命名的文件")
        return True
    
    print(f"找到 {len(files_to_rename)} 个需要重命名的文件")
    
    success_count = 0
    error_count = 0
    
    for old_file in files_to_rename:
        try:
            # 生成新文件名：去掉 .txt 部分
            old_name = old_file.name
            new_name = old_name.replace('.txt.xlsx', '.xlsx')
            new_file = old_file.parent / new_name
            
            # 检查新文件名是否已存在
            if new_file.exists():
                print(f"警告: 目标文件已存在，跳过: {new_file}")
                continue
            
            # 重命名文件
            old_file.rename(new_file)
            print(f"重命名: {old_name} -> {new_name}")
            success_count += 1
            
        except Exception as e:
            print(f"错误: 重命名失败 {old_file}: {e}")
            error_count += 1
    
    print(f"\n重命名完成:")
    print(f"  成功: {success_count} 个文件")
    print(f"  失败: {error_count} 个文件")
    
    return error_count == 0

def main():
    # 预处理数据目录
    preprocess_dir = "/home/asus515/PycharmProjects/TsyFSpectrumClassify_remote/data/dataset_regression/preprocess"
    
    print("开始批量重命名预处理文件...")
    print(f"目标目录: {preprocess_dir}")
    
    success = rename_preprocess_files(preprocess_dir)
    
    if success:
        print("\n✅ 所有文件重命名成功!")
    else:
        print("\n❌ 部分文件重命名失败，请检查错误信息")
        sys.exit(1)

if __name__ == "__main__":
    main()