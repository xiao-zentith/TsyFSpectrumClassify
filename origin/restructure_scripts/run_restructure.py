#!/usr/bin/env python3
"""
一键执行项目重构脚本
按顺序执行所有重构步骤
"""
import os
import sys
import subprocess
from pathlib import Path

def run_command(command, description):
    """运行命令并显示结果"""
    print(f"\n{'='*50}")
    print(f"🚀 {description}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run([sys.executable, command], 
                              capture_output=True, 
                              text=True, 
                              cwd=Path.cwd())
        
        if result.returncode == 0:
            print(f"✅ {description} 完成")
            if result.stdout:
                print("输出:")
                print(result.stdout)
        else:
            print(f"❌ {description} 失败")
            if result.stderr:
                print("错误:")
                print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ 执行 {description} 时发生异常: {str(e)}")
        return False
    
    return True

def main():
    """主函数"""
    print("🎯 TsyF Spectrum Classification 项目重构工具")
    print("=" * 60)
    print("这个工具将按以下顺序执行重构:")
    print("1. 完整项目重构 (备份 + 目录结构 + 文件移动)")
    print("2. 更新导入路径")
    print("3. 验证重构结果")
    print("=" * 60)
    
    # 确认执行
    response = input("\n是否开始执行完整重构流程？(y/N): ")
    if response.lower() != 'y':
        print("❌ 操作已取消")
        return 1
    
    # 检查必要文件
    required_files = [
        'complete_restructure.py',
        'update_imports.py', 
        'test_restructure.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ 缺少必要文件: {missing_files}")
        return 1
    
    # 执行重构步骤
    steps = [
        ('complete_restructure.py', '完整项目重构'),
        ('update_imports.py', '更新导入路径'),
        ('test_restructure.py', '验证重构结果')
    ]
    
    success_count = 0
    for script, description in steps:
        if run_command(script, description):
            success_count += 1
        else:
            print(f"\n❌ {description} 失败，停止执行")
            break
        
        # 询问是否继续下一步
        if success_count < len(steps):
            response = input(f"\n继续执行下一步？(Y/n): ")
            if response.lower() == 'n':
                print("⏸️ 用户选择停止")
                break
    
    # 总结结果
    print(f"\n{'='*60}")
    print("🏁 重构流程执行完成")
    print(f"✅ 成功完成: {success_count}/{len(steps)} 个步骤")
    
    if success_count == len(steps):
        print("\n🎉 恭喜！项目重构全部完成！")
        print("\n📋 生成的文件:")
        generated_files = [
            'restructure.log',
            'import_update_report.txt', 
            'test_report.txt',
            'requirements.txt',
            'README.md',
            '.gitignore'
        ]
        
        for file in generated_files:
            if Path(file).exists():
                print(f"  ✅ {file}")
            else:
                print(f"  ❌ {file} (未生成)")
        
        print("\n📖 下一步:")
        print("1. 查看生成的报告文件")
        print("2. 测试重构后的功能")
        print("3. 提交代码到版本控制")
        print("4. 通知团队成员新的项目结构")
        
        return 0
    else:
        print(f"\n⚠️ 重构未完全完成，请检查错误信息")
        print("💡 建议:")
        print("1. 查看错误输出")
        print("2. 手动执行失败的步骤")
        print("3. 检查项目状态")
        
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⏹️ 用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 发生未预期的错误: {str(e)}")
        sys.exit(1)