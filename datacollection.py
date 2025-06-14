import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import time
import math

cap = cv2.VideoCapture(0)  # 0 because using device camera, if you are using web camera put 1 instead of 0
detector = HandDetector(maxHands=1) # max no. of hands detect at a time
offset = 20
imgSize = 300
counter = 0

folder = "C:/Sign_Language_Detection/Data/Yes"

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']
        # print(f"Bounding Box: x={x}, y={y}, w={w}, h={h}")


        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255

        imgCrop = img[y-offset: y + h + offset, x-offset: x + w + offset]
        imgCropShape = imgCrop.shape

        aspectratio = h/w

        if aspectratio > 1:
            k = imgSize / h
            wCal = math.ceil(k*w)  # wCal - weight calculated
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal)/2)
            imgWhite[:, wGap : wCal + wGap] = imgResize
        
        else:
            k = imgSize / w
            hCal = math.ceil(k*h)  # wCal - weight calculated
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal)/2)
            imgWhite[hGap : hCal + hGap, :] = imgResize

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)
    
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('s'):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
