from mmocr.apis import MMOCRInferencer
infer = MMOCRInferencer(det='dbnetpp')
result = infer('1.jpg', return_vis=False)
Korr= result['predictions'][0]['det_polygons']
from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import time
import matplotlib.pyplot as plt
start=time.time()
# config = Cfg.load_config_from_file('config.yml') # sử dụng config của các bạn được export lúc train nếu đã thay đổi tham số  
config = Cfg.load_config_from_name('vgg_transformer') 
detector = Predictor(config)
import cv2
im=cv2.imread('001.jpg')
for point in Korr:
    min_y=int(min(point[1:-1:2]))
    min_x=int(min(point[0:-1:2]))
    max_y=int(max(point[1:-1:2]))
    max_x=int(max(point[0:-1:2]))
    print(min_y,max_y, min_x,max_x)
    mat=im[min_y:max_y,min_x:max_x]
    matt = Image.fromarray(mat)
    s = detector.predict(matt, return_prob=False)
    print(Korr.index(point),s)
    cv2.imwrite("Test_"+str(Korr.index(point))+".jpg",mat)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

