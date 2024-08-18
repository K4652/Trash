
import cv2
import numpy as np
from flask import Flask, render_template, request
from PIL import Image
from werkzeug.utils import secure_filename
path = 'c:/data/temp/'
# 입력 이미지
app = Flask(__name__)
imgSize = (300, 300)
@app.route('/')
def upload_page():
	return render_template('index1.html')

#파일 업로드 처리
@app.route('/fileUpload', methods = ['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		f = request.files['file']
		#저장할 경로 + 파일명
		file_path = './' + secure_filename(f.filename)
		f.save(file_path)
		img = Image.open(file_path)
        
        # 입력 이미지
	return '완료'


if __name__ == '__main__':
    # 서버 실행
    app.run(debug = True)


