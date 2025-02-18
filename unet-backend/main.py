import os
import torch
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import yaml
import albumentations as albu
from albumentations.core.composition import Compose
import archs
from flask_cors import CORS
import logging
import uuid
import time

# 创建Flask应用
app = Flask(__name__)
CORS(app)

# 配置上传文件目录和输出目录
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# 配置日志记录
logging.basicConfig(level=logging.INFO)

# 加载模型
def load_model(model_name='data_NestedUNet_woDS'):
    try:
        model_path = f'models/{model_name}/model.pth'
        config_path = f'models/{model_name}/config.yml'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = archs.__dict__[config['arch']](config['num_classes'],
                                               config['input_channels'],
                                               config['deep_supervision'])
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        model.to(device)
        return model, config
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Failed to load model: {e}")

# 图像处理函数
def process_image(img_path, model, config):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Error: Could not load image from {img_path}")
    h, w, _ = img.shape
    
    # 图像预处理
    val_transform = Compose([
        albu.Resize(config['input_h'], config['input_w']),
        albu.Normalize(),
    ])
    augmented = val_transform(image=img)
    img = augmented['image']
    img = img.astype('float32') / 255  # 归一化图像
    img = img.transpose(2, 0, 1)  # 转换为C, H, W格式
    img = torch.from_numpy(img)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input = img.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input)
        if config['deep_supervision']:
            output = output[-1]
        output = torch.sigmoid(output).cpu().numpy()  # Sigmoid激活函数处理
        output = np.clip(output[0, 0], 0, 1)  # 确保输出值在 [0, 1] 范围内
        output = (cv2.resize(output, dsize=(w, h), interpolation=cv2.INTER_LINEAR) * 255).astype('uint8')  # 恢复原图尺寸
    
    return output, img, h, w

# 保存图像
def save_output_image(output, overlay, filename):
    # 去掉原始文件名的扩展名，确保只添加一次 .png
    base_name = os.path.splitext(filename)[0]  # 获取文件名（去掉扩展名）
    pred_mask_filename = secure_filename(f"pred_mask_{base_name}.png")
    overlay_filename = secure_filename(f"overlay_{base_name}.png")
    
    pred_mask_path = os.path.join(app.config['OUTPUT_FOLDER'], pred_mask_filename)
    overlay_path = os.path.join(app.config['OUTPUT_FOLDER'], overlay_filename)
    
    cv2.imwrite(pred_mask_path, output)
    cv2.imwrite(overlay_path, overlay)
    
    # 返回可访问的URL
    base_url = "http://localhost:5000"  # 手动指定基础 URL
    return (
        f"{base_url}/images/{pred_mask_filename}",
        f"{base_url}/images/{overlay_filename}"
    )

# 定期清理旧文件
def cleanup_old_files(folder, days=7):
    now = time.time()
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if os.stat(filepath).st_mtime < now - days * 86400:
            os.remove(filepath)

# 上传并处理图像的路由
@app.route('/predict', methods=['POST'])
def predict():
    logging.info("Received a new prediction request")
    if 'image' not in request.files:
        logging.error("No file part in the request")
        return jsonify({'error': 'No file part'}), 400
    file = request.files['image']
    if file.filename == '':
        logging.error("No selected file in the request")
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # 生成唯一文件名
        unique_filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(img_path)
        logging.info(f"File saved successfully: {img_path}")
        
        # 加载模型
        model, config = load_model('data_NestedUNet_woDS')
        logging.info("Model loaded successfully")
        
        # 处理图像并生成输出
        output, _, h, w = process_image(img_path, model, config)
        logging.info("Image processed successfully")
        
        # 生成叠加图像
        overlay = cv2.imread(img_path)
        mask = output > 0  # 创建布尔掩码
        overlay[mask] = [0, 255, 255]  # 使用黄色叠加显示预测区域
        
        # 保存预测掩码和叠加图像
        pred_mask_url, overlay_url = save_output_image(output, overlay, unique_filename)
        logging.info("Prediction results saved successfully")
        
        # 清理旧文件
        cleanup_old_files(app.config['UPLOAD_FOLDER'])
        cleanup_old_files(app.config['OUTPUT_FOLDER'])
        
        return jsonify({
            'pred_mask': pred_mask_url,
            'overlay': overlay_url
        })
    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return jsonify({'error': str(e)}), 500

# 提供静态文件服务
@app.route('/images/<filename>')
def download_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)