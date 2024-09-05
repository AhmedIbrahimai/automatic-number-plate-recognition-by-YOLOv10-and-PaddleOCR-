import cv2 
import pandas as pd
import os
from paddleocr import PaddleOCR
from ultralytics import YOLO
import numpy as np
import cvzone 
os.chdir(r"C:/Users/RTC\Downloads/Youtube_content/Car_Plater_Paddel_OCR/rpi-ocr-number-plate-read")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

model = YOLO("best.pt")

ocr = PaddleOCR()

with open("coco.txt" , "r") as my_file:
    data = my_file.read()
    class_list = data.split("\n")

cap = cv2.VideoCapture("nr.mp4")

area = [(124,339) , (127,451) , (485,440) , (460,328)]

def perform_ocr(image_array):
    if image_array is None:
        raise ValueError(" Image is Noe")
    
    results = ocr.ocr(image_array , rec = True)
    
    detected_text = []
    if results[0] is not None:
        #print(results)
        for result in results[0]:
            text = result[1][0]
            detected_text.append(text)

        return '' .join(detected_text)    

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame , (1020,500))
    results = model.predict(frame)
    a = results[0].boxes.data
    a = a.cpu()
    px = pd.DataFrame(a).astype(float)

    list = []

    for indes , row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])

        label = int(row[5])
        c = class_list[label]

        cx = int(x1 + x2) //2
        cy = int(y1 + y2) //2


        result = cv2.pointPolygonTest(np.array(area , np.int32) , ((cx,cy)) , False)
        if result >= 0:
            cv2.rectangle(frame , (x1,y1), (x2,y2) , (255,0,0) , 2)
            crop = frame[y1:y2 , x1:x2]
            crop = cv2.resize(crop , (110,30))
            text = perform_ocr(crop)

            cvzone.putTextRect(frame , f'{text}' , (cx-50 , cy-30) , 1 ,2)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.release()
cv2.destroyAllWindows()


