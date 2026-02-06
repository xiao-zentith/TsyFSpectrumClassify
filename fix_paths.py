#!/usr/bin/env python3
"""
修复路径配置脚本
正确处理之前重构中的路径替换问题
"""

import json
import os
import re
from pathlib import Path


def fix_config_json():
    """修复config.json中的路径配置"""
    config_path = Path("config.json")
    if not config_path.exists():
        return
    
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 修复错误的路径替换
    content = content.replace('get_data_path("raw")', 'data/raw')
    content = content.replace('get_data_path("processed")', 'data/processed')
    content = content.replace('get_data_path("target")', 'data/target')
    
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ 修复了 {config_path}")


def fix_merge_json():
    """修复merge_json.py中的路径"""
    file_path = Path("src/utils/data_io/merge_json.py")
    if not file_path.exists():
        return
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 修复错误的路径替换
    content = content.replace("with open('get_data_path(\"raw\")', 'r') as f2:", 
                             "with open('data/raw/config.json', 'r') as f2:")
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ 修复了 {file_path}")


def create_path_usage_example():
    """创建路径管理器使用示例"""
    example_content = '''"""
路径管理器使用示例
"""

from src.utils.path_manager import PathManager

# 初始化路径管理器
pm = PathManager()

# 获取各种路径
data_raw = pm.get_path('data', 'raw')
data_processed = pm.get_path('data', 'processed')
models_classification = pm.get_path('models', 'classification')
results_regression = pm.get_path('results', 'regression')

print(f"原始数据路径: {data_raw}")
print(f"处理后数据路径: {data_processed}")
print(f"分类模型路径: {models_classification}")
print(f"回归结果路径: {results_regression}")

# 在代码中使用
import json

# 读取配置文件
config_path = pm.get_path('configs', 'main') / 'config.json'
with open(config_path, 'r') as f:
    config = json.load(f)

# 保存模型
model_path = pm.get_path('models', 'classification') / 'best_model.pth'
# torch.save(model.state_dict(), model_path)
'''
    
    example_path = Path("examples/path_manager_usage.py")
    example_path.parent.mkdir(exist_ok=True)
    
    with open(example_path, 'w', encoding='utf-8') as f:
        f.write(example_content)
    
    print(f"✅ 创建了使用示例: {example_path}")


def main():
    """主函数"""
    print("🔧 修复路径配置问题...")
    
    fix_config_json()
    fix_merge_json()
    create_path_usage_example()
    
    print("\n🎉 路径修复完成!")
    print("\n📖 使用建议:")
    print("1. 在新代码中使用 PathManager 来管理路径")
    print("2. 参考 examples/path_manager_usage.py 了解用法")
    print("3. 避免在代码中硬编码路径")


if __name__ == "__main__":
    main()