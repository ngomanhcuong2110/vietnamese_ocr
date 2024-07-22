from paddleocr import PaddleOCR
import cv2
import argparse
from PIL import Image

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
start = time.time()
# config = Cfg.load_config_from_file('config.yml') # sử dụng config của các bạn được export lúc train nếu đã thay đổi tham số
config = Cfg.load_config_from_name('vgg_transformer')
config['device'] = 'cpu'

detector = Predictor(config)
# Also switch the language by modifying the lang parameter
ocr = PaddleOCR(lang="en",use_gpu=False) # The model file will be downloaded automatically when executed for the first time
img_path ='1.jpg'

i=cv2.imread(img_path)
import numpy as np


result = ocr.ocr(i)
# Recognition and detection can be performed separately through parameter control
# result = ocr.ocr(img_path, det=False)  Only perform recognition
# result = ocr.ocr(img_path, rec=False)  Only perform detection
# Print detection frame and recognition result
for line in result:
    print(line)

# Visualization
mat = cv2.imread(img_path)

boxes = [line[0] for line in result[0]]
scores = [line[1][1] for line in result[0]]

for box in boxes:
    print("=======",box)
    top_left     = (int(box[0][0])-5, int(box[0][1])-5)
    bottom_right = (int(box[2][0])+15, int(box[2][1])+15)

    im=mat[int(box[0][1])-5:int(box[2][1])+15,int(box[0][0])-5:int(box[2][0])+15]


    im = Image.fromarray(im)
    s = detector.predict(im, return_prob=True)
    #cv2.rectangle(mat, top_left, bottom_right, (0, 255, 0), 2)

    print(s)

# img = './a.JPG'
# img = Image.open(img)

# dự đoán
print(detector.predict(im, return_prob=True))
print(time.time()-start)
cv2.imwrite("res.jpg",mat)