import cv2
from ultralytics import YOLO
import pyttsx3
import pytesseract

model = YOLO("C:/Users/user/Desktop/Data Science/DL/DL Projects/SuperMarket Store Shelf Detector/runs/detect/train/weights/best.pt")
pytesseract.pytesseract.tesseract_cmd=r'c:\Program Files\Tesseract-OCR\tesseract.exe'
video=cv2.VideoCapture('shelf.mp4')
txt_sp=pyttsx3.init()
while True:
    suc,img=video.read()
    img1=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    results=model.predict(source=img,show=True)
    for result in results:
        for box in result.boxes:
            conf = box.conf[0].item()  # Convert tensor to a Python float
            cls = int(box.cls[0])      # Convert tensor to an integer
            if conf>0.4 and cls==0:
                text=pytesseract.image_to_string(img1)
                if text.strip():  
                    notification=f'Found empty space at {text}'
                    print(notification)
                    txt_sp.say(notification)
                    txt_sp.runAndWait()
    if cv2.waitKey(1) & 0XFF==ord('q'):
            break
video.release()
cv2.destroyAllWindows()

# txt_sp=pyttsx3.init()
# img=cv2.imread('shelf1.png')
# img1=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# results=model.predict(source=img,show=True)
# text=pytesseract.image_to_string(img1)
# print(text)
# for result in results:
#     print(result.boxes)
#     for box in result.boxes:
#         conf = box.conf[0].item()  # Convert tensor to a Python float
#         cls = int(box.cls[0])      # Convert tensor to an integer
#         if conf>0.4 and cls==0:
#             if text.strip():  
#                 notification=f'Found empty space at {text}'
#                 print(notification)
#                 txt_sp.say(notification)
#                 txt_sp.runAndWait()