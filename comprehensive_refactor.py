#!/usr/bin/env python3
"""
全面的项目重构脚本
处理文件命名规范化、硬编码路径问题和代码适应性修改

作者: AI Assistant
日期: 2024-11-30
"""

import os
import re
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Set
from datetime import datetime


class ComprehensiveRefactor:
    """全面的项目重构器"""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.log_file = self.project_root / f"refactor_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        # 文件重命名映射
        self.file_rename_map = {}
        
        # 硬编码路径模式
        self.hardcoded_patterns = [
            get_project_path()\']*',
            r'get_project_path()"\']*',
            r'r["\']C:\\\\Users\\\\[^"\']*["\']',
            r'r["\'][^"\']*TsyFSpectrumClassify[^"\']*["\']'
        ]
        
        # 需要跳过的目录
        self.skip_dirs = {'.git', '__pycache__', '.venv', 'venv', 'node_modules', 'backup_*'}
        
        # 深度学习项目命名规范
        self.naming_rules = {
            'python_files': {
                'pattern': r'^[a-z][a-z0-9_]*\.py$',
                'description': '使用小写字母和下划线，如: model_trainer.py'
            },
            'class_names': {
                'pattern': r'^[A-Z][a-zA-Z0-9]*$',
                'description': '使用驼峰命名，如: ModelTrainer'
            },
            'function_names': {
                'pattern': r'^[a-z][a-z0-9_]*$',
                'description': '使用小写字母和下划线，如: train_model'
            },
            'constants': {
                'pattern': r'^[A-Z][A-Z0-9_]*$',
                'description': '使用大写字母和下划线，如: MAX_EPOCHS'
            }
        }
    
    def log(self, message: str):
        """记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry + '\n')
    
    def analyze_naming_issues(self) -> Dict[str, List[str]]:
        """分析文件命名问题"""
        self.log("🔍 分析文件命名问题...")
        
        issues = {
            'irregular_python_files': [],
            'irregular_directories': [],
            'typos': [],
            'case_issues': []
        }
        
        # 检查Python文件命名
        for py_file in self.project_root.rglob("*.py"):
            if any(skip in str(py_file) for skip in self.skip_dirs):
                continue
                
            filename = py_file.name
            
            # 检查不规范的命名
            if not re.match(self.naming_rules['python_files']['pattern'], filename):
                issues['irregular_python_files'].append(str(py_file.relative_to(self.project_root)))
        
        # 检查目录命名
        for directory in self.project_root.rglob("*"):
            if not directory.is_dir():
                continue
            if any(skip in str(directory) for skip in self.skip_dirs):
                continue
                
            dir_name = directory.name
            
            # 检查拼写错误
            if 'classfication' in dir_name:
                issues['typos'].append(str(directory.relative_to(self.project_root)))
            
            # 检查大小写问题
            if re.search(r'[A-Z]', dir_name) and dir_name not in ['README.md', 'LICENSE']:
                issues['case_issues'].append(str(directory.relative_to(self.project_root)))
        
        # 输出分析结果
        self.log("\n📊 命名问题分析结果:")
        for category, files in issues.items():
            if files:
                self.log(f"  {category}: {len(files)} 个问题")
                for file in files[:5]:  # 只显示前5个
                    self.log(f"    - {file}")
                if len(files) > 5:
                    self.log(f"    ... 还有 {len(files) - 5} 个")
        
        return issues
    
    def generate_rename_plan(self, issues: Dict[str, List[str]]) -> Dict[str, str]:
        """生成重命名计划"""
        self.log("\n📋 生成重命名计划...")
        
        rename_plan = {}
        
        # 处理不规范的Python文件
        for file_path in issues['irregular_python_files']:
            old_path = self.project_root / file_path
            filename = old_path.name
            
            # 规范化文件名
            new_filename = self._normalize_filename(filename)
            if new_filename != filename:
                new_path = old_path.parent / new_filename
                rename_plan[str(old_path)] = str(new_path)
        
        # 处理拼写错误的目录
        for dir_path in issues['typos']:
            old_path = self.project_root / dir_path
            if 'classfication' in old_path.name:
                new_name = old_path.name.replace('classfication', 'classification')
                new_path = old_path.parent / new_name
                rename_plan[str(old_path)] = str(new_path)
        
        self.log(f"  生成了 {len(rename_plan)} 个重命名操作")
        return rename_plan
    
    def _normalize_filename(self, filename: str) -> str:
        """规范化文件名"""
        name, ext = os.path.splitext(filename)
        
        # 处理常见的不规范命名
        name = name.replace('-', '_')  # 连字符改为下划线
        name = re.sub(r'(\d+)D_', r'\1d_', name)  # 2D_CNN -> 2d_cnn
        name = re.sub(r'([A-Z]+)', lambda m: m.group(1).lower(), name)  # 大写改小写
        name = re.sub(r'_+', '_', name)  # 多个下划线合并
        name = name.strip('_')  # 去除首尾下划线
        
        # 特殊处理
        special_cases = {
            'read)mat': 'read_mat',  # 修复括号错误
            '2d_cnn1': 'cnn_2d_v1',
            '1d_cnn': 'cnn_1d',
            'lstm1': 'lstm_v1',
            'rf1': 'random_forest_v1',
            'knn1': 'knn_v1',
            'transformer1': 'transformer_v1'
        }
        
        if name in special_cases:
            name = special_cases[name]
        
        return name + ext
    
    def scan_hardcoded_paths(self) -> Dict[str, List[Tuple[str, int, str]]]:
        """扫描硬编码路径"""
        self.log("\n🔍 扫描硬编码路径...")
        
        hardcoded_files = {}
        
        for file_path in self.project_root.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix not in ['.py', '.json', '.md', '.txt']:
                continue
            if any(skip in str(file_path) for skip in self.skip_dirs):
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                matches = []
                for line_num, line in enumerate(content.split('\n'), 1):
                    for pattern in self.hardcoded_patterns:
                        for match in re.finditer(pattern, line):
                            matches.append((str(file_path.relative_to(self.project_root)), 
                                          line_num, match.group()))
                
                if matches:
                    hardcoded_files[str(file_path.relative_to(self.project_root))] = matches
                    
            except Exception as e:
                self.log(f"  ⚠️ 无法读取文件 {file_path}: {e}")
        
        self.log(f"  发现 {len(hardcoded_files)} 个文件包含硬编码路径")
        return hardcoded_files
    
    def create_path_config_system(self):
        """创建路径配置系统"""
        self.log("\n🛠️ 创建路径配置系统...")
        
        # 创建路径配置文件
        path_config = {
            "project_root": ".",
            "data": {
                "raw": "data/raw",
                "processed": "data/processed", 
                "target": "data/target"
            },
            "models": {
                "classification": "models/classification",
                "regression": "models/regression"
            },
            "results": {
                "classification": "results/classification",
                "regression": "results/regression"
            },
            "configs": {
                "main": "configs/main",
                "classification": "configs/classification",
                "regression": "configs/regression",
                "preprocessing": "configs/preprocessing"
            },
            "logs": "logs",
            "notebooks": "notebooks"
        }
        
        config_path = self.project_root / "configs" / "paths.json"
        config_path.parent.mkdir(exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(path_config, f, indent=2, ensure_ascii=False)
        
        # 创建路径管理工具
        path_manager_code = '''"""
路径管理工具
提供统一的路径管理接口，避免硬编码路径
"""

import json
import os
from pathlib import Path
from typing import Dict, Any


class PathManager:
    """路径管理器"""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            # 自动查找配置文件
            current_dir = Path(__file__).parent
            while current_dir != current_dir.parent:
                config_file = current_dir / "configs" / "paths.json"
                if config_file.exists():
                    config_path = str(config_file)
                    break
                current_dir = current_dir.parent
            
            if config_path is None:
                raise FileNotFoundError("未找到路径配置文件 paths.json")
        
        self.config_path = Path(config_path)
        self.project_root = self.config_path.parent.parent
        self._load_config()
    
    def _load_config(self):
        """加载配置"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
    
    def get_path(self, *keys) -> Path:
        """获取路径"""
        current = self.config
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                raise KeyError(f"路径配置中未找到: {'.'.join(keys)}")
        
        if isinstance(current, str):
            path = self.project_root / current
            path.mkdir(parents=True, exist_ok=True)
            return path
        else:
            raise ValueError(f"路径配置值必须是字符串: {'.'.join(keys)}")
    
    def get_data_path(self, data_type: str = "raw") -> Path:
        """获取数据路径"""
        return self.get_path("data", data_type)
    
    def get_model_path(self, model_type: str) -> Path:
        """获取模型路径"""
        return self.get_path("models", model_type)
    
    def get_result_path(self, result_type: str) -> Path:
        """获取结果路径"""
        return self.get_path("results", result_type)
    
    def get_config_path(self, config_type: str) -> Path:
        """获取配置路径"""
        return self.get_path("configs", config_type)
    
    def get_log_path(self) -> Path:
        """获取日志路径"""
        return self.get_path("logs")


# 全局路径管理器实例
try:
    path_manager = PathManager()
except FileNotFoundError:
    path_manager = None
    print("警告: 未找到路径配置文件，请确保 configs/paths.json 存在")


def get_project_path(*keys) -> Path:
    """便捷函数：获取项目路径"""
    if path_manager is None:
        raise RuntimeError("路径管理器未初始化")
    return path_manager.get_path(*keys)


def get_data_path(data_type: str = "raw") -> Path:
    """便捷函数：获取数据路径"""
    if path_manager is None:
        raise RuntimeError("路径管理器未初始化")
    return path_manager.get_data_path(data_type)
'''
        
        path_manager_file = self.project_root / "src" / "utils" / "path_manager.py"
        with open(path_manager_file, 'w', encoding='utf-8') as f:
            f.write(path_manager_code)
        
        self.log(f"  ✅ 创建路径配置文件: {config_path}")
        self.log(f"  ✅ 创建路径管理器: {path_manager_file}")
    
    def fix_hardcoded_paths(self, hardcoded_files: Dict[str, List[Tuple[str, int, str]]]):
        """修复硬编码路径"""
        self.log("\n🔧 修复硬编码路径...")
        
        fixed_count = 0
        
        for file_path, matches in hardcoded_files.items():
            full_path = self.project_root / file_path
            
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                # 替换硬编码路径
                for pattern in self.hardcoded_patterns:
                    content = re.sub(pattern, self._generate_relative_path, content)
                
                # 如果内容有变化，写回文件
                if content != original_content:
                    with open(full_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    fixed_count += 1
                    self.log(f"  ✅ 修复文件: {file_path}")
                    
            except Exception as e:
                self.log(f"  ❌ 修复文件失败 {file_path}: {e}")
        
        self.log(f"  修复了 {fixed_count} 个文件")
    
    def _generate_relative_path(self, match) -> str:
        """生成相对路径替换"""
        path_str = match.group()
        
        # 简单的路径替换策略
        if 'dataset' in path_str:
            return 'get_data_path("raw")'
        elif 'config' in path_str:
            return 'get_project_path("configs")'
        elif 'result' in path_str:
            return 'get_project_path("results")'
        else:
            return 'get_project_path()'
    
    def preview_changes(self, rename_plan: Dict[str, str], hardcoded_files: Dict):
        """预览所有更改"""
        self.log("\n👀 预览所有更改:")
        self.log("="*60)
        
        self.log(f"\n📁 文件重命名 ({len(rename_plan)} 个):")
        for old_path, new_path in list(rename_plan.items())[:10]:
            self.log(f"  {Path(old_path).name} → {Path(new_path).name}")
        if len(rename_plan) > 10:
            self.log(f"  ... 还有 {len(rename_plan) - 10} 个重命名操作")
        
        self.log(f"\n🔧 硬编码路径修复 ({len(hardcoded_files)} 个文件):")
        for file_path in list(hardcoded_files.keys())[:10]:
            self.log(f"  {file_path}")
        if len(hardcoded_files) > 10:
            self.log(f"  ... 还有 {len(hardcoded_files) - 10} 个文件")
        
        self.log("="*60)
    
    def execute_refactoring(self, rename_plan: Dict[str, str], hardcoded_files: Dict):
        """执行重构"""
        self.log("\n🚀 开始执行重构...")
        
        # 1. 执行文件重命名
        self.log("\n📁 执行文件重命名...")
        rename_success = 0
        for old_path, new_path in rename_plan.items():
            try:
                old_p = Path(old_path)
                new_p = Path(new_path)
                
                if old_p.exists():
                    new_p.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(old_p), str(new_p))
                    rename_success += 1
                    self.log(f"  ✅ {old_p.name} → {new_p.name}")
                    
            except Exception as e:
                self.log(f"  ❌ 重命名失败 {old_path}: {e}")
        
        self.log(f"  重命名成功: {rename_success}/{len(rename_plan)}")
        
        # 2. 修复硬编码路径
        self.fix_hardcoded_paths(hardcoded_files)
        
        # 3. 创建路径配置系统
        self.create_path_config_system()
    
    def run_comprehensive_refactor(self):
        """运行全面重构"""
        self.log("🎯 开始全面项目重构")
        self.log("="*60)
        
        try:
            # 1. 分析命名问题
            issues = self.analyze_naming_issues()
            
            # 2. 生成重命名计划
            rename_plan = self.generate_rename_plan(issues)
            
            # 3. 扫描硬编码路径
            hardcoded_files = self.scan_hardcoded_paths()
            
            # 4. 预览更改
            self.preview_changes(rename_plan, hardcoded_files)
            
            # 5. 用户确认
            print("\n" + "="*60)
            print("⚠️  即将执行以下操作:")
            print(f"   - 重命名 {len(rename_plan)} 个文件/目录")
            print(f"   - 修复 {len(hardcoded_files)} 个文件中的硬编码路径")
            print(f"   - 创建统一的路径管理系统")
            print("="*60)
            
            confirm = input("是否继续执行重构? (y/N): ").strip().lower()
            if confirm in ['y', 'yes']:
                # 6. 执行重构
                self.execute_refactoring(rename_plan, hardcoded_files)
                
                self.log("\n🎉 重构完成!")
                self.log(f"📄 详细日志: {self.log_file}")
            else:
                self.log("❌ 用户取消了重构操作")
                
        except Exception as e:
            self.log(f"❌ 重构过程中出现错误: {e}")
            raise


if __name__ == "__main__":
    refactor = ComprehensiveRefactor()
    refactor.run_comprehensive_refactor()