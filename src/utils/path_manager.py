"""
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
