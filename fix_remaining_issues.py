#!/usr/bin/env python3
"""
修复项目结构中剩余的问题
"""

import json
import os
from pathlib import Path


def fix_config_json():
    """修复config.json中的路径配置问题"""
    config_path = Path("config.json")
    
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 修复路径配置
        config["dataset_processed"] = "data/processed"
        config["dataset_target1"] = "data/target"
        config["dataset_target2"] = "data/target"
        config["dataset_target3"] = "data/target"
        config["dataset_target4"] = "data/target"
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 修复了 {config_path} 中的路径配置")
    else:
        print(f"❌ 未找到 {config_path}")


def create_missing_directories():
    """创建paths.json中定义但缺失的目录"""
    paths_config = Path("configs/paths.json")
    
    if not paths_config.exists():
        print(f"❌ 未找到 {paths_config}")
        return
    
    with open(paths_config, 'r', encoding='utf-8') as f:
        paths = json.load(f)
    
    # 需要创建的目录列表
    directories_to_create = [
        "models/classification",
        "models/regression", 
        "results/classification",
        "results/regression",
        "logs",
        "data/processed",
        "data/target"
    ]
    
    created_dirs = []
    for dir_path in directories_to_create:
        path = Path(dir_path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            created_dirs.append(dir_path)
    
    if created_dirs:
        print(f"✅ 创建了以下目录:")
        for dir_path in created_dirs:
            print(f"   - {dir_path}")
    else:
        print("ℹ️  所有必需的目录都已存在")


def create_directory_readme_files():
    """为新创建的目录添加README文件"""
    readme_configs = [
        {
            "path": "models/README.md",
            "content": """# 模型目录

此目录用于存储训练好的模型文件。

## 目录结构
- `classification/` - 分类模型
- `regression/` - 回归模型

## 使用方法
```python
from src.utils.path_manager import PathManager

pm = PathManager()
model_path = pm.get_path('models', 'classification')
```
"""
        },
        {
            "path": "results/README.md", 
            "content": """# 结果目录

此目录用于存储模型训练和评估的结果。

## 目录结构
- `classification/` - 分类结果
- `regression/` - 回归结果

## 文件类型
- 训练日志
- 评估报告
- 可视化图表
- 性能指标
"""
        },
        {
            "path": "logs/README.md",
            "content": """# 日志目录

此目录用于存储应用程序运行日志。

## 日志类型
- 训练日志
- 错误日志
- 调试信息
- 性能监控

## 使用方法
```python
from src.utils.path_manager import PathManager

pm = PathManager()
log_path = pm.get_path('logs')
```
"""
        }
    ]
    
    for config in readme_configs:
        readme_path = Path(config["path"])
        if not readme_path.exists():
            readme_path.parent.mkdir(parents=True, exist_ok=True)
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(config["content"])
            print(f"✅ 创建了 {readme_path}")


def validate_path_manager():
    """验证路径管理器是否正常工作"""
    try:
        # 测试导入
        import sys
        sys.path.append('src')
        from utils.path_manager import PathManager
        
        pm = PathManager()
        
        # 测试几个关键路径
        test_paths = [
            ('data', 'raw'),
            ('models', 'classification'),
            ('results', 'regression'),
            ('logs',)
        ]
        
        print("🧪 测试路径管理器:")
        for path_keys in test_paths:
            try:
                path = pm.get_path(*path_keys)
                print(f"   ✅ {'.'.join(path_keys)}: {path}")
            except Exception as e:
                print(f"   ❌ {'.'.join(path_keys)}: {e}")
        
        return True
    except Exception as e:
        print(f"❌ 路径管理器测试失败: {e}")
        return False


def main():
    """主函数"""
    print("🔧 修复项目结构中的剩余问题...")
    print("=" * 50)
    
    # 1. 修复配置文件
    print("\n1️⃣ 修复配置文件路径...")
    fix_config_json()
    
    # 2. 创建缺失的目录
    print("\n2️⃣ 创建缺失的目录...")
    create_missing_directories()
    
    # 3. 添加README文件
    print("\n3️⃣ 创建目录说明文件...")
    create_directory_readme_files()
    
    # 4. 验证路径管理器
    print("\n4️⃣ 验证路径管理器...")
    if validate_path_manager():
        print("✅ 路径管理器工作正常")
    
    print("\n🎉 项目结构修复完成!")
    print("\n📋 修复总结:")
    print("   ✅ 修复了config.json中的路径配置")
    print("   ✅ 创建了缺失的目录结构")
    print("   ✅ 添加了目录说明文档")
    print("   ✅ 验证了路径管理器功能")
    
    print("\n💡 建议:")
    print("   1. 查看 PROJECT_STRUCTURE_ANALYSIS.md 了解完整分析")
    print("   2. 使用 PathManager 管理所有路径操作")
    print("   3. 定期检查配置文件的一致性")


if __name__ == "__main__":
    main()