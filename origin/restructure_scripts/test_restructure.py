#!/usr/bin/env python3
"""
重构后项目测试脚本
验证项目结构和导入是否正常工作
"""
import os
import sys
import importlib
from pathlib import Path
from typing import List, Dict, Tuple

class ProjectTester:
    def __init__(self, project_root=None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.test_results = {
            'structure': [],
            'imports': [],
            'files': []
        }
        
    def test_directory_structure(self) -> bool:
        """测试目录结构"""
        print("🔍 测试目录结构...")
        
        required_dirs = [
            'src',
            'src/utils',
            'src/utils/data_io',
            'src/utils/visualization',
            'src/utils/metrics',
            'src/utils/file_operations',
            'src/classification',
            'src/classification/models',
            'src/classification/models/demo',
            'src/classification/utils',
            'src/regression',
            'src/regression/models',
            'src/regression/training',
            'src/regression/utils',
            'src/augmentation',
            'src/preprocessing',
            'src/ui',
            'notebooks',
            'tests',
            'configs',
            'scripts',
            'data'
        ]
        
        missing_dirs = []
        existing_dirs = []
        
        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            if full_path.exists() and full_path.is_dir():
                existing_dirs.append(dir_path)
                self.test_results['structure'].append(('PASS', dir_path, '目录存在'))
            else:
                missing_dirs.append(dir_path)
                self.test_results['structure'].append(('FAIL', dir_path, '目录不存在'))
        
        print(f"  ✅ 存在的目录: {len(existing_dirs)}")
        print(f"  ❌ 缺失的目录: {len(missing_dirs)}")
        
        if missing_dirs:
            print("  缺失的目录:")
            for dir_path in missing_dirs:
                print(f"    - {dir_path}")
        
        return len(missing_dirs) == 0
    
    def test_key_files(self) -> bool:
        """测试关键文件是否存在"""
        print("\n🔍 测试关键文件...")
        
        key_files = [
            'src/utils/data_io/mat_reader.py',
            'src/utils/visualization/spectrum_plotter.py',
            'src/classification/models/cnn_2d_v1.py',
            'src/regression/models/unet.py',
            'src/preprocessing/data_augmenter.py',
            'requirements.txt',
            'README.md',
            '.gitignore'
        ]
        
        missing_files = []
        existing_files = []
        
        for file_path in key_files:
            full_path = self.project_root / file_path
            if full_path.exists() and full_path.is_file():
                existing_files.append(file_path)
                self.test_results['files'].append(('PASS', file_path, '文件存在'))
            else:
                missing_files.append(file_path)
                self.test_results['files'].append(('FAIL', file_path, '文件不存在'))
        
        print(f"  ✅ 存在的文件: {len(existing_files)}")
        print(f"  ❌ 缺失的文件: {len(missing_files)}")
        
        if missing_files:
            print("  缺失的文件:")
            for file_path in missing_files:
                print(f"    - {file_path}")
        
        return len(missing_files) == 0
    
    def test_init_files(self) -> bool:
        """测试__init__.py文件"""
        print("\n🔍 测试__init__.py文件...")
        
        required_init_dirs = [
            'src',
            'src/utils',
            'src/utils/data_io',
            'src/utils/visualization',
            'src/utils/metrics',
            'src/utils/file_operations',
            'src/classification',
            'src/classification/models',
            'src/classification/utils',
            'src/regression',
            'src/regression/models',
            'src/regression/training',
            'src/augmentation',
            'src/preprocessing',
            'tests'
        ]
        
        missing_init = []
        existing_init = []
        
        for dir_path in required_init_dirs:
            init_file = self.project_root / dir_path / '__init__.py'
            if init_file.exists():
                existing_init.append(dir_path)
            else:
                missing_init.append(dir_path)
        
        print(f"  ✅ 存在__init__.py: {len(existing_init)}")
        print(f"  ❌ 缺失__init__.py: {len(missing_init)}")
        
        if missing_init:
            print("  缺失__init__.py的目录:")
            for dir_path in missing_init:
                print(f"    - {dir_path}")
        
        return len(missing_init) == 0
    
    def test_sample_imports(self) -> bool:
        """测试示例导入"""
        print("\n🔍 测试示例导入...")
        
        # 添加项目根目录到Python路径
        if str(self.project_root) not in sys.path:
            sys.path.insert(0, str(self.project_root))
        
        test_imports = [
            'src.utils.data_io',
            'src.utils.visualization',
            'src.classification.models',
            'src.regression.models',
            'src.preprocessing'
        ]
        
        successful_imports = []
        failed_imports = []
        
        for module_name in test_imports:
            try:
                importlib.import_module(module_name)
                successful_imports.append(module_name)
                self.test_results['imports'].append(('PASS', module_name, '导入成功'))
            except ImportError as e:
                failed_imports.append((module_name, str(e)))
                self.test_results['imports'].append(('FAIL', module_name, f'导入失败: {str(e)}'))
            except Exception as e:
                failed_imports.append((module_name, str(e)))
                self.test_results['imports'].append(('FAIL', module_name, f'其他错误: {str(e)}'))
        
        print(f"  ✅ 成功导入: {len(successful_imports)}")
        print(f"  ❌ 导入失败: {len(failed_imports)}")
        
        if failed_imports:
            print("  导入失败的模块:")
            for module_name, error in failed_imports:
                print(f"    - {module_name}: {error}")
        
        return len(failed_imports) == 0
    
    def check_python_syntax(self) -> bool:
        """检查Python文件语法"""
        print("\n🔍 检查Python文件语法...")
        
        python_files = []
        for root, dirs, files in os.walk(self.project_root / 'src'):
            dirs[:] = [d for d in dirs if d != '__pycache__']
            for file in files:
                if file.endswith('.py'):
                    python_files.append(Path(root) / file)
        
        syntax_errors = []
        valid_files = []
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                compile(content, str(file_path), 'exec')
                valid_files.append(file_path)
            except SyntaxError as e:
                syntax_errors.append((file_path, str(e)))
            except Exception as e:
                syntax_errors.append((file_path, f"其他错误: {str(e)}"))
        
        print(f"  ✅ 语法正确: {len(valid_files)}")
        print(f"  ❌ 语法错误: {len(syntax_errors)}")
        
        if syntax_errors:
            print("  语法错误的文件:")
            for file_path, error in syntax_errors[:5]:  # 只显示前5个
                print(f"    - {file_path}: {error}")
            if len(syntax_errors) > 5:
                print(f"    ... 还有 {len(syntax_errors) - 5} 个文件有语法错误")
        
        return len(syntax_errors) == 0
    
    def generate_test_report(self):
        """生成测试报告"""
        report_file = self.project_root / "test_report.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("项目重构测试报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"测试时间: {os.popen('date').read().strip()}\n\n")
            
            # 目录结构测试结果
            f.write("目录结构测试结果:\n")
            f.write("-" * 30 + "\n")
            for status, item, message in self.test_results['structure']:
                f.write(f"{status}: {item} - {message}\n")
            
            # 文件存在性测试结果
            f.write("\n文件存在性测试结果:\n")
            f.write("-" * 30 + "\n")
            for status, item, message in self.test_results['files']:
                f.write(f"{status}: {item} - {message}\n")
            
            # 导入测试结果
            f.write("\n导入测试结果:\n")
            f.write("-" * 30 + "\n")
            for status, item, message in self.test_results['imports']:
                f.write(f"{status}: {item} - {message}\n")
        
        print(f"\n📝 测试报告已生成: {report_file}")
    
    def run_all_tests(self) -> bool:
        """运行所有测试"""
        print("开始项目重构验证测试...")
        print("=" * 50)
        
        tests = [
            ("目录结构", self.test_directory_structure),
            ("关键文件", self.test_key_files),
            ("__init__.py文件", self.test_init_files),
            ("示例导入", self.test_sample_imports),
            ("Python语法", self.check_python_syntax)
        ]
        
        results = []
        for test_name, test_func in tests:
            try:
                result = test_func()
                results.append((test_name, result))
            except Exception as e:
                print(f"❌ {test_name}测试异常: {str(e)}")
                results.append((test_name, False))
        
        # 汇总结果
        print("\n" + "=" * 50)
        print("测试结果汇总:")
        
        passed = 0
        for test_name, result in results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {status}: {test_name}")
            if result:
                passed += 1
        
        print(f"\n总体结果: {passed}/{len(results)} 项测试通过")
        
        # 生成报告
        self.generate_test_report()
        
        return passed == len(results)

def main():
    """主函数"""
    print("项目重构验证测试工具")
    print("=" * 30)
    
    tester = ProjectTester()
    
    try:
        success = tester.run_all_tests()
        
        if success:
            print("\n🎉 所有测试通过！项目重构成功！")
            return 0
        else:
            print("\n⚠️ 部分测试失败，请检查测试报告")
            return 1
            
    except Exception as e:
        print(f"\n❌ 测试过程中发生异常: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())