#!/usr/bin/env python3
"""
导入路径更新脚本
用于重构后更新所有Python文件的import语句
"""
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

class ImportUpdater:
    def __init__(self, project_root=None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.import_mapping = self.get_import_mapping()
        self.updated_files = []
        
    def get_import_mapping(self) -> Dict[str, str]:
        """定义导入路径映射"""
        return {
            # Utils模块映射
            'from src.utils.data_io.mat_reader import': 'from src.utils.data_io.mat_reader import',
            'from src.utils.data_io.matrix_reader import': 'from src.utils.data_io.matrix_reader import',
            'from src.utils.data_io.npz_reader import': 'from src.utils.data_io.npz_reader import',
            'from src.utils.data_io.data_loader import': 'from src.utils.data_io.data_loader import',
            'from src.utils.data_io.data_extractor import': 'from src.utils.data_io.data_extractor import',
            'from src.utils.data_io.mat_tool import': 'from src.utils.data_io.mat_tool import',
            'from src.utils.data_io.json_generator import': 'from src.utils.data_io.json_generator import',
            
            'from src.utils.visualization.spectrum_plotter import': 'from src.utils.visualization.spectrum_plotter import',
            'from src.utils.visualization.contour_plotter import': 'from src.utils.visualization.contour_plotter import',
            'from src.utils.visualization.radar_plotter import': 'from src.utils.visualization.radar_plotter import',
            'from src.utils.visualization.result_plotter import': 'from src.utils.visualization.result_plotter import',
            
            'from src.utils.metrics.similarity_calculator import': 'from src.utils.metrics.similarity_calculator import',
            'from src.utils.metrics.pearson_calculator import': 'from src.utils.metrics.pearson_calculator import',
            'from src.utils.metrics.cosine_similarity import': 'from src.utils.metrics.cosine_similarity import',
            
            'from src.utils.file_operations.batch_resizer import': 'from src.utils.file_operations.batch_resizer import',
            'from src.utils.file_operations.resizer import': 'from src.utils.file_operations.resizer import',
            'from src.utils.file_operations.json_merger import': 'from src.utils.file_operations.json_merger import',
            'from src.utils.file_operations.txt_to_xlsx import': 'from src.utils.file_operations.txt_to_xlsx import',
            
            # 分类模块映射
            'from src.classification.utils.image_dataset import': 'from src.classification.utils.image_dataset import',
            'from src.classification.utils.plot_utils import': 'from src.classification.utils.plot_utils import',
            'from src.classification.utils.matrix_reader import': 'from src.classification.utils.matrix_reader import',
            
            # 回归模块映射
            'from src.regression.training.custom_dataset import': 'from src.regression.training.custom_dataset import',
            'from src.regression.models.': 'from src.regression.models.',
            
            # 预处理模块映射
            'from src.preprocessing.': 'from src.preprocessing.',
            
            # 增强模块映射
            'from src.augmentation.': 'from src.augmentation.',
            
            # import语句映射
            'import src.utils.': 'import src.utils.',
            'import src.classification.': 'import src.classification.',
            'import src.regression.': 'import src.regression.',
            'import src.preprocessing.': 'import src.preprocessing.',
            'import src.augmentation.': 'import src.augmentation.',
        }
    
    def find_python_files(self) -> List[Path]:
        """查找所有Python文件"""
        python_files = []
        for root, dirs, files in os.walk(self.project_root):
            # 跳过某些目录
            dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', '.venv', 'venv']]
            
            for file in files:
                if file.endswith('.py'):
                    python_files.append(Path(root) / file)
        
        return python_files
    
    def update_file_imports(self, file_path: Path) -> bool:
        """更新单个文件的导入语句"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            updated = False
            
            # 应用导入映射
            for old_import, new_import in self.import_mapping.items():
                if old_import in content:
                    content = content.replace(old_import, new_import)
                    updated = True
            
            # 如果有更新，写回文件
            if updated:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.updated_files.append(str(file_path))
                print(f"✅ 更新: {file_path}")
                return True
            
            return False
            
        except Exception as e:
            print(f"❌ 更新失败 {file_path}: {str(e)}")
            return False
    
    def update_all_imports(self):
        """更新所有文件的导入语句"""
        print("开始更新导入路径...")
        
        python_files = self.find_python_files()
        print(f"找到 {len(python_files)} 个Python文件")
        
        updated_count = 0
        for file_path in python_files:
            if self.update_file_imports(file_path):
                updated_count += 1
        
        print(f"\n更新完成: {updated_count}/{len(python_files)} 个文件被更新")
        
        if self.updated_files:
            print("\n更新的文件列表:")
            for file_path in self.updated_files:
                print(f"  - {file_path}")
    
    def generate_import_report(self):
        """生成导入更新报告"""
        report_file = self.project_root / "import_update_report.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("导入路径更新报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"更新时间: {os.popen('date').read().strip()}\n")
            f.write(f"更新文件数量: {len(self.updated_files)}\n\n")
            
            f.write("更新的文件列表:\n")
            for file_path in self.updated_files:
                f.write(f"  - {file_path}\n")
            
            f.write("\n应用的映射规则:\n")
            for old, new in self.import_mapping.items():
                f.write(f"  {old} -> {new}\n")
        
        print(f"📝 报告已生成: {report_file}")

def main():
    """主函数"""
    print("导入路径更新工具")
    print("=" * 30)
    
    updater = ImportUpdater()
    
    try:
        updater.update_all_imports()
        updater.generate_import_report()
        print("\n✅ 导入路径更新完成!")
        
    except Exception as e:
        print(f"\n❌ 更新过程中发生异常: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())