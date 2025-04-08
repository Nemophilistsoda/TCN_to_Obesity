import yaml
import os
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """加载YAML格式的配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件 {config_path} 不存在")

    with open(config_path, 'r', encoding='utf-8') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"配置文件解析错误: {e}")

    # 路径兼容性处理（确保相对路径基于项目根目录）
    for path_type in ['data_paths', 'model_paths', 'result_paths']:
        if path_type in config:
            for key, value in config[path_type].items():
                if not os.path.isabs(value):
                    config[path_type][key] = os.path.join(
                        os.path.dirname(__file__), '..', value
                    )

    return config