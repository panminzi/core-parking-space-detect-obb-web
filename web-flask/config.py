# MD5: d5aea0a0783223c023213c3675e81f6f
"""
版权所有 © 2025 羊羊小栈 (GJQ)

作者：羊羊小栈
时间：2025-07-31 16:25:08

本系统为原创作品，禁止二次销售！禁止将系统对应的视频、文档进行二次发布！
违者需立即停止侵权行为，并按照【羊羊小栈系统版权声明及保护条款】中规定数额进行赔偿，并承担相应法律责任。
"""

"""
配置文件
"""
import os

# 用户数据文件
USERS_FILE = 'users.json'

# 获取项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_BASE_PATH = os.path.join(BASE_DIR, 'other', 'model_train', 'detect_obb', 'output')

# 停车位检测模型配置
MODEL_CONFIGS = {
    'ready-model': {
        'name': 'YOLO11-OBB 停车位检测模型(已经训练好的模型)',
        'model_path': os.path.join(MODEL_BASE_PATH, '已经训练好的模型和测试结果', 'train', 'weights', 'best.pt'),
        'train_results_path': os.path.join(MODEL_BASE_PATH, '已经训练好的模型和测试结果', 'train', 'results.csv'),
        'val_data_path': os.path.join(MODEL_BASE_PATH, '已经训练好的模型和测试结果', 'val'),
        'val_accuracy_path': os.path.join(MODEL_BASE_PATH, '已经训练好的模型和测试结果', 'val', '测试集精度.txt'),
    },
    'training-model': {
        'name': 'YOLO11-OBB 停车位检测模型(新训练的模型)',
        'model_path': os.path.join(MODEL_BASE_PATH, 'train', 'weights', 'best.pt'),
        'train_results_path': os.path.join(MODEL_BASE_PATH, 'train', 'results.csv'),
        'val_data_path': os.path.join(MODEL_BASE_PATH, 'val'),
        'val_accuracy_path': os.path.join(MODEL_BASE_PATH, 'val', '测试集精度.txt'),
    }
}

# 停车位检测类别中文映射
CLASS_NAME_MAPPING = {
    'occupied': '已占用车位',
    'vacant': '空闲车位'
} 

# 可用模型列表
AVAILABLE_MODELS = [
    {
        'key': key,
        'name': config['name'],
        'model_path': config['model_path'],
        'train_results_path': config.get('train_results_path'),
        'val_data_path': config.get('val_data_path'),
        'val_accuracy_path': config.get('val_accuracy_path'),
        'num_classes': len(CLASS_NAME_MAPPING),
        'supported_classes': list(CLASS_NAME_MAPPING.values())
    }
    for key, config in MODEL_CONFIGS.items()
]

# 默认模型
DEFAULT_MODEL = 'ready-model'

