import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from torchvision import models
import ResNet

app = Flask(__name__)



# 데이터 전처리 정의 (훈련할 때 사용한 것과 동일해야 함)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# CSV 파일을 통해 라벨 맵 생성
csv_file = r'C:\Users\poiu4\Project\argumented_data.csv'
df = pd.read_csv(csv_file)
label_map = {i: label for i, label in enumerate(df['label'].astype('category').cat.categories)}

num_classes = len(label_map)
sample_point = torch.load('best_model.pth',map_location=torch.device('cpu'))


model = ResNet(num_classes=num_classes)

model.load_state_dict(sample_point)
model.eval()

# 이미지 예측 함수
def predict_image(image_path):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image_transformed = transform(image).unsqueeze(0)  # 배치 차원 추가
    
    with torch.no_grad():
        outputs = model(image_transformed)
        _, predicted = torch.max(outputs, 1)
        
    return label_map[predicted.item()]

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/api', methods=['POST'])
def api():
    # POST 요청에서 이미지 파일 받기
    if 'image' not in request.files:
        return jsonify({'error': 'No image found'}), 400
    image_file = request.files['image']

    try:
        filename = secure_filename(image_file.filename)
        file_path = os.path.join('uploads', filename)

        if not os.path.exists('uploads'):
            os.makedirs('uploads')
        image_file.save(file_path)

        label = predict_image(image_file)
        return render_template('result.html',label = label)
    except Exception as e:
        app.logger.error(f"Exception occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500
    


if __name__ == '__main__':
    app.run(host='0.0.0.0',debug= True)
   


# API로 json 파일 내보내기
#https://devrokket.tistory.com/3 (참고)
#https://tutorials.pytorch.kr/intermediate/flask_rest_api_tutorial.html



# 1.23.5