# MD5: 4ee205ed1c0ce729b7180d30a09644c5
"""
版权所有 © 2025 羊羊小栈 (GJQ)

作者：羊羊小栈
时间：2025-07-31 16:25:08

本系统为原创作品，禁止二次销售！禁止将系统对应的视频、文档进行二次发布！
违者需立即停止侵权行为，并按照【羊羊小栈系统版权声明及保护条款】中规定数额进行赔偿，并承担相应法律责任。
"""

"""
用户服务模块
"""
import os
import json
from datetime import datetime
from config import USERS_FILE


def load_users():
    """加载用户数据"""
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_users(users):
    """保存用户数据"""
    with open(USERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(users, f, ensure_ascii=False, indent=2)


def login_user(username, password):
    """用户登录"""
    if not username or not password:
        return {'code': 400, 'message': '用户名和密码不能为空'}
    
    users = load_users()
    if username not in users:
        return {'code': 400, 'message': '用户名或密码错误'}
    
    user = users[username]
    if user['password'] != password:  # 不加密，直接比较
        return {'code': 400, 'message': '用户名或密码错误'}
    
    return {
        'code': 200,
        'message': '登录成功',
        'data': {
            'user': {
                'id': username,
                'username': username,
                'realName': user.get('realName', username)
            }
        }
    }


def register_user(username, password, real_name=None):
    """用户注册"""
    if not username or not password:
        return {'code': 400, 'message': '用户名和密码不能为空'}
    
    if len(username) < 3 or len(password) < 6:
        return {'code': 400, 'message': '用户名至少3位，密码至少6位'}
    
    users = load_users()
    if username in users:
        return {'code': 400, 'message': '用户名已存在'}
    
    users[username] = {
        'password': password,  # 不加密，直接存储
        'realName': real_name or username,
        'createTime': datetime.now().isoformat()
    }
    save_users(users)
    
    return {'code': 200, 'message': '注册成功'}





def init_default_users():
    """初始化默认用户"""
    users = load_users()
    if not users:
        users['admin'] = {
            'password': '123456',  # 不加密
            'realName': '管理员',
            'createTime': datetime.now().isoformat()
        }
        users['user'] = {
            'password': '123456',  # 不加密
            'realName': '普通用户',
            'createTime': datetime.now().isoformat()
        }
        save_users(users)
        print("已创建默认用户：")
        print("用户名: admin, 密码: 123456")
        print("用户名: user, 密码: 123456") 