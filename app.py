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

config = Cfg.load_config_from_name('vgg_transformer') 
config['device'] = 'cuda:0'

app = FastAPI()

@app.post("/recognize")
async def recognize_image(image: UploadFile = File(...)):
    """
    API endpoint to upload an image, convert to base64, and perform character recognition
    using Vietocr (fallback to Tesseract if Vietocr is unavailable).
    """

    try:
        # Read image content
        content = await image.read()

        # Convert image content to PIL Image object
        img = Image.open(io.BytesIO(content))

        #Convert image to base64 string (optional, comment out if not needed)
        with io.BytesIO() as buffer:
            img.save(buffer, format="JPEG")
            base64_encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        # Attempt character recognition using Vietocr (preferred)
    
        
        detector = Predictor(config)
        # Also switch the language by modifying the lang parameter
        ocr = PaddleOCR(lang="en") # The model file will be downloaded automatically when executed for the first time
        from io import BytesIO

        # Create a BytesIO object to store the decoded bytes
        image_data = BytesIO(base64.b64decode(base64_encoded_image))

        # Read the image from the BytesIO object
        image = Image.open(image_data)

        image=np.array(image)
        
        result = ocr.ocr(image)
       
       
        # Visualization
        mat = image
        size=image.shape
        boxes = [line[0] for line in result[0]]
        scores = [line[1][1] for line in result[0]]

        
        string=[]
        for box in boxes:
            index=boxes.index(box)    
            top_left     = (int(box[0][0])-5, int(box[0][1])-5)
            bottom_right = (int(box[2][0])+15, int(box[2][1])+15)
        
            im=mat[int(box[0][1])-5:int(box[2][1])+15,int(box[0][0])-5:int(box[2][0])+15]
            
            im = Image.fromarray(im)
            s = detector.predict(im, return_prob=True)
            #cv2.rectangle(mat, top_left, bottom_right, (0, 255, 0), 2)

            string.append({"text":s[0],"prob:":s[1]*scores[index],"top_left":{"x":top_left[0],"y":top_left[1]},"width":int(box[2][0])+20-int(box[0][0]),"height":int(box[2][1])-int(box[0][1])+20})
        return {"width":size[0],"height":size[1],"res": string}
        return boxes

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="localhost", port=8000)
