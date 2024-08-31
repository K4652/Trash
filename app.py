import cv2
import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, jsonify
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torchvision import models
from ResNet import ResNet
from werkzeug.utils import secure_filename

app = Flask(__name__)

# 데이터 전처리 정의 (훈련할 때 사용한 것과 동일해야 함)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# CSV 파일을 통해 라벨 맵 생성
csv_file = r'C:\Users\poiu4\Project\argumented_data.csv'
df = pd.read_csv(csv_file, )
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
def upload_page():
	return render_template('index1.html')

#파일 업로드 처리
@app.route('/fileUpload', methods = ['GET', 'POST'])
def upload_file():
	uploads_dir = os.path.join(os.getcwd(), 'static')
	if not os.path.exists(uploads_dir):
		os.makedirs(uploads_dir)
	model = torch.load("best_model.pth")


	if request.method == 'POST':
		f = request.files['image']
		filename = secure_filename(f.filename)
		file_path = os.path.join('uploads', filename)
		if not os.path.exists('uploads'):
			os.makedirs('uploads')
		f.save(file_path)
		img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
		img = cv2.resize(img,(224, 224))
		label = predict_image(f)
		data = (np.asarray(img) / 255.0).reshape(1, 784)  # Reshape for model input (flattened)

		pred = model.predict(f)
		print(pred[0])
		pred = np.argmax(pred, axis=1)
	return render_template('view.html', out=str(pred[0]), im=f.filename)

if __name__ == '__main__':
    # 서버 실행
	app.run(host='0.0.0.0',debug= True)