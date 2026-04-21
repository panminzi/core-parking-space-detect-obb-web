"""
智慧交通停车位检测系统 - Flask后端
"""
import os
import json
import uuid
import tempfile
import time
import re
from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
from service.user_service import (
    login_user, register_user, init_default_users
)
from service.detection_service import get_models, detect_objects
from service.model_data_service import get_model_data
from service.video_detection_service import start_video_processing, get_processing_status
from service.realtime_detection_service import detect_objects_realtime

app = Flask(__name__)
CORS(app)


# ==================== 页面路由 ====================

@app.route('/')
@app.route('/login.html')
def index():
    """登录页面"""
    login_html_path = os.path.join('templates', 'login.html')
    with open(login_html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    return html_content, 200, {'Content-Type': 'text/html; charset=utf-8'}


@app.route('/images/<filename>')
def serve_images(filename):
    """提供images目录下的静态文件"""
    images_path = os.path.join('templates', 'images')
    return send_from_directory(images_path, filename)


@app.route('/detect.html')
def detect_html():
    """停车位检测页面"""
    detect_html_path = os.path.join('templates', 'detect.html')
    with open(detect_html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    return html_content, 200, {'Content-Type': 'text/html; charset=utf-8'}



@app.route('/model-data.html')
def model_data_html():
    """模型数据查看页面"""
    model_data_html_path = os.path.join('templates', 'model-data.html')
    with open(model_data_html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    return html_content, 200, {'Content-Type': 'text/html; charset=utf-8'}


@app.route('/video-detect.html')
def video_detect_html():
    """视频停车位检测页面"""
    video_detect_html_path = os.path.join('templates', 'video-detect.html')
    with open(video_detect_html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    return html_content, 200, {'Content-Type': 'text/html; charset=utf-8'}


@app.route('/realtime-detect.html')
def realtime_detect_html():
    """实时摄像头检测页面"""
    realtime_detect_html_path = os.path.join('templates', 'realtime-detect.html')
    with open(realtime_detect_html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    return html_content, 200, {'Content-Type': 'text/html; charset=utf-8'}


# ==================== 用户API ====================

@app.route('/api/user/login', methods=['POST'])
def api_login():
    """用户登录API"""
    try:
        data = request.json
        username = data.get('username')
        password = data.get('password')
        
        result = login_user(username, password)
        return jsonify(result)
    except Exception as e:
        return jsonify({'code': 500, 'message': f'登录失败: {str(e)}'})


@app.route('/api/user/register', methods=['POST'])
def api_register():
    """用户注册API"""
    try:
        data = request.json
        username = data.get('username')
        password = data.get('password')
        real_name = data.get('realName')
        
        result = register_user(username, password, real_name)
        return jsonify(result)
    except Exception as e:
        return jsonify({'code': 500, 'message': f'注册失败: {str(e)}'})


# ==================== 停车位检测API ====================

@app.route('/api/models', methods=['GET'])
def api_get_models():
    """获取可用模型列表"""
    result = get_models()
    return jsonify(result)


@app.route('/api/detect', methods=['POST'])
def api_detect():
    """停车位检测接口"""
    try:
        data = request.json
        model_name = data.get('model', 'ready-model')
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'code': 400, 'message': '请提供图像数据'})
        
        result = detect_objects(model_name, image_data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'code': 500, 'message': f'停车位检测失败: {str(e)}'})


@app.route('/api/model-data', methods=['GET'])
def api_get_model_data():
    """获取模型训练和验证数据"""
    try:
        data_type = request.args.get('type')  # 'training' or 'validation'
        model_key = request.args.get('model')
        
        if not data_type or not model_key:
            return jsonify({'code': 400, 'message': '缺少必要参数'})
        
        result = get_model_data(data_type, model_key)
        return jsonify(result)
    except Exception as e:
        return jsonify({'code': 500, 'message': f'获取模型数据失败: {str(e)}'})


# ==================== 视频检测API ====================

@app.route('/api/video/process', methods=['POST'])
def api_video_process():
    """视频处理接口（流式响应）"""
    
    try:
        # 检查是否有上传的文件
        if 'video' not in request.files:
            return jsonify({'code': 400, 'message': '请上传视频文件'})
        
        video_file = request.files['video']
        model_name = request.form.get('model', 'ready-model')
        
        if video_file.filename == '':
            return jsonify({'code': 400, 'message': '请选择视频文件'})
        
        # 生成唯一的会话ID
        session_id = str(uuid.uuid4())
        
        # 保存上传的视频文件到临时目录
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, f"input_{session_id}.mp4")
        video_file.save(video_path)
        
        # 开始异步处理视频
        processor = start_video_processing(session_id, model_name, video_path)
        
        def generate_progress():
            """生成进度流"""
            yield "data: " + json.dumps({
                'type': 'start',
                'session_id': session_id,
                'message': '开始处理视频...'
            }) + "\n\n"
            
            last_progress = -1
            while True:
                status = processor.get_status()
                
                if status['error']:
                    yield "data: " + json.dumps({
                        'type': 'error',
                        'message': status['error']
                    }) + "\n\n"
                    break
                
                # 发送进度更新
                if status['progress'] != last_progress:
                    yield "data: " + json.dumps({
                        'type': 'progress',
                        'progress': status['progress'],
                        'message': status['message']
                    }) + "\n\n"
                    last_progress = status['progress']
                
                # 检查是否完成
                if not status['is_processing'] and status['result']:
                    # 生成视频访问URL
                    video_url = f"/api/video/download/{session_id}"
                    
                    result_data = status['result'].copy()
                    result_data['processed_video_url'] = video_url
                    result_data['session_id'] = session_id
                    
                    yield "data: " + json.dumps({
                        'type': 'result',
                        'data': result_data
                    }) + "\n\n"
                    break
                
                if not status['is_processing'] and not status['result']:
                    yield "data: " + json.dumps({
                        'type': 'error',
                        'message': '处理失败，未知错误'
                    }) + "\n\n"
                    break
                
                time.sleep(1)  # 每秒检查一次状态
        
        return Response(
            generate_progress(),
            mimetype='text/plain',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*'
            }
        )
        
    except Exception as e:
        return jsonify({'code': 500, 'message': f'视频处理失败: {str(e)}'})


@app.route('/api/video/download/<session_id>')
def api_video_download(session_id):
    """视频下载接口，支持范围请求以实现视频流播放"""
    try:
        status_result = get_processing_status(session_id)
        
        if status_result['code'] != 200:
            return jsonify(status_result)
        
        status = status_result['data']
        
        if not status['result']:
            return jsonify({'code': 404, 'message': '视频文件不存在'})
        
        processed_video_path = status['result']['processed_video_path']
        
        if not os.path.exists(processed_video_path):
            return jsonify({'code': 404, 'message': '处理后的视频文件不存在'})
        
        # 获取文件信息
        file_size = os.path.getsize(processed_video_path)
        file_ext = os.path.splitext(processed_video_path)[1].lower()
        
        # 确定MIME类型
        if file_ext == '.mp4':
            mimetype = 'video/mp4'
        elif file_ext == '.avi':
            mimetype = 'video/x-msvideo'
        elif file_ext == '.mov':
            mimetype = 'video/quicktime'
        else:
            mimetype = 'video/mp4'  # 默认
        
        # 处理Range请求
        range_header = request.headers.get('Range', None)
        if range_header:
            # 解析Range头
            byte_start = 0
            byte_end = file_size - 1
            
            if range_header:
                match = re.match(r'bytes=(\d+)-(\d*)', range_header)
                if match:
                    byte_start = int(match.group(1))
                    if match.group(2):
                        byte_end = int(match.group(2))
            
            # 确保范围有效
            byte_start = max(0, byte_start)
            byte_end = min(file_size - 1, byte_end)
            content_length = byte_end - byte_start + 1
            
            # 读取文件的指定范围
            with open(processed_video_path, 'rb') as f:
                f.seek(byte_start)
                data = f.read(content_length)
            
            # 创建部分内容响应
            response = Response(
                data,
                206,  # Partial Content
                {
                    'Content-Type': mimetype,
                    'Accept-Ranges': 'bytes',
                    'Content-Length': str(content_length),
                    'Content-Range': f'bytes {byte_start}-{byte_end}/{file_size}',
                    'Cache-Control': 'no-cache'
                }
            )
            return response
        
        # 没有Range请求，返回完整文件
        response = send_from_directory(
            os.path.dirname(processed_video_path),
            os.path.basename(processed_video_path),
            as_attachment=False,
            mimetype=mimetype
        )
        
        # 添加支持范围请求的头
        response.headers['Accept-Ranges'] = 'bytes'
        response.headers['Content-Type'] = mimetype
        response.headers['Cache-Control'] = 'no-cache'
        response.headers['Content-Length'] = str(file_size)
        
        return response
        
    except Exception as e:
        return jsonify({'code': 500, 'message': f'视频下载失败: {str(e)}'})


@app.route('/api/video/status/<session_id>')
def api_video_status(session_id):
    """获取视频处理状态"""
    try:
        result = get_processing_status(session_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({'code': 500, 'message': f'获取状态失败: {str(e)}'})


# ==================== 实时检测API ====================

@app.route('/api/realtime/detect', methods=['POST'])
def api_realtime_detect():
    """实时摄像头停车位检测接口"""
    try:
        data = request.json
        model_name = data.get('model', 'ready-model')
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'code': 400, 'message': '请提供图像数据'})
        
        result = detect_objects_realtime(model_name, image_data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'code': 500, 'message': f'实时检测失败: {str(e)}'})


# ==================== 应用启动 ====================

if __name__ == '__main__':
    # 初始化默认用户
    init_default_users()
    
    app.run(host='0.0.0.0', port=5011)