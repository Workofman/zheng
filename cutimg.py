import cv2
import numpy as np
import os

index = 0

for path in os.listdir('C:/Users/Administrator/Desktop/ec/'):
    img = cv2.imread('C:/Users/Administrator/Desktop/ec/'+path,0)
    blurred = cv2.GaussianBlur(img,(9,9),0)
    gradX = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=1, dy=0)
    gradY = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=0, dy=1)

    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    blurred = cv2.GaussianBlur(gradient,(9,9),0)
    (_, thresh) = cv2.threshold(blurred, 20, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closed = cv2.erode(closed, None, iterations=1)
    closed = cv2.dilate(closed, None, iterations=1)
    (_,cnt,_) = cv2.findContours(closed.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    box = []

    for i in range(0,len(cnt)):
        c = sorted(cnt, key=cv2.contourArea, reverse=True)[i]

        # compute the rotated bounding box of the largest contour
        rect = cv2.minAreaRect(c)
        box_ = np.int0(cv2.boxPoints(rect))
        box.append(box_)
    draw_img = cv2.drawContours(img.copy(), box, -1, (0, 0, 255), 3)

    for j in range(0,len(box)):
        Xs = [i[0] for i in box[j]]
        Ys = [i[1] for i in box[j]]
        x1 = min(Xs)
        x2 = max(Xs)
        y1 = min(Ys)
        y2 = max(Ys)
        hight = y2 - y1
        width = x2 - x1
        crop_img= img[y1:y1+hight, x1:x1+width]
        cv2.imwrite('C:/Users/Administrator/Desktop/queue/%d.bmp'%index, crop_img)
        index = index +1


