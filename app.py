from flask import Flask, render_template, request, url_for
import os
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as transforms
from ResNet import ResNet
from werkzeug.utils import secure_filename
import ssl

# SSL 검증 비활성화 (주의해서 사용해야 함)
ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__)

# 데이터 전처리 정의 (훈련할 때 사용한 것과 동일해야 함)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
df2 = pd.read_csv('Trash.csv')
df2.to_csv('Trash.csv', index=None)
# CSV 파일을 통해 라벨 맵 생성
csv_file = r'C:\Users\poiu4\Project\argumented_data.csv'   # 해당 위치로 변경하세요!!!!
df = pd.read_csv(csv_file)
label_map = {i: label for i, label in enumerate(df['label'].astype('category').cat.categories)}

num_classes = len(label_map)
sample_point = torch.load('best_model.pth', map_location=torch.device('cpu'), weights_only=True)

# 모델 초기화 및 가중치 로드
model = ResNet(num_classes=num_classes)
model.load_state_dict(sample_point)
model.eval()

# 이미지 예측 함수
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image_transformed = transform(image).unsqueeze(0)  # 배치 차원 추가
    
    with torch.no_grad():
        outputs = model(image_transformed)
        _, predicted = torch.max(outputs, 1)
        
    return label_map[predicted.item()]

@app.route('/')
def upload_page():
    return render_template('index1.html')

@app.route('/fileUpload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files.get('file')  # 'file' 필드에서 파일 가져오기
        if f and f.filename:  # 파일이 제대로 업로드 되었는지 확인
            filename = secure_filename(f.filename)
            file_path = os.path.join('static/uploads', filename)
            if not os.path.exists('static/uploads'):
                os.makedirs('static/uploads')
            f.save(file_path)
            
            label = predict_image(file_path)
            df2[df2['품명'] == label].to_csv('ttrash.csv', index= None)
            data = pd.read_csv('ttrash.csv')
            return render_template('view.html', tables=[data.to_html()], titles=[''], label=label, image_filename=filename)
        else:
            # 파일이 없는 경우에 대한 응답
            return "No file uploaded", 400  # 400 Bad Request 응답
    
    # GET 요청에 대한 처리 (예: 업로드 페이지 반환)
    return render_template('index1.html')

if __name__ == '__main__':
    # 서버 실행
    app.run(host='0.0.0.0', port=8088, debug=True)
