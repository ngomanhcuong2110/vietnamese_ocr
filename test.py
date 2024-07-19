from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import base64
from paddleocr import PaddleOCR
import cv2
import argparse
from PIL import Image
import numpy as np
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import time
start=time.time()
ocr = PaddleOCR(lang="en") 
# config = Cfg.load_config_from_file('config.yml') # sử dụng config của các bạn được export lúc train nếu đã thay đổi tham số  
config = Cfg.load_config_from_name('vgg_transformer') 
config['device'] = 'cuda:0' # The model file will be downloaded automatically when executed for the first time
img_path ='001.jpg'

# Also switch the language by modifying the lang parameter
ocr = PaddleOCR(lang="en") # The model file will be downloaded automatically when executed for the first time


# Read the image from the BytesIO object
image = Image.open(img_path)

image=np.array(image)

result = ocr.ocr(image)


# Visualization
mat = image
size=image.shape
boxes = [line[0] for line in result[0]]
scores = [line[1][1] for line in result[0]]


string=[]
detector = Predictor(config)

for box in boxes:
    index=boxes.index(box)  
    print("Box number",index)  
    top_left     = (int(box[0][0])-5, int(box[0][1])-5)
    bottom_right = (int(box[2][0])+15, int(box[2][1])+15)

    im=mat[int(box[0][1])-5:int(box[2][1])+15,int(box[0][0])-5:int(box[2][0])+15]
    string=[]

    im = Image.fromarray(im)
    s = detector.predict(im, return_prob=True)
    #cv2.rectangle(mat, top_left, bottom_right, (0, 255, 0), 2)

    string.append({"text":s[0],"prob:":s[1]*scores[index],"top_left":{"x":top_left[0],"y":top_left[1]},"width":int(box[2][0])+20-int(box[0][0]),"height":int(box[2][1])-int(box[0][1])+20})
    print(s[0])
print(time.time()-start)