import cv2
import os

index = 0
for path in os.listdir('queue/'):
    img = cv2.imread('queue/'+path,0)
    left = 128-img.shape[1]
    right = 128-img.shape[1]
    top = 128-img.shape[0]
    down = 128-img.shape[0]
    (_, thresh) = cv2.threshold(img,0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    thresh = 255 - thresh
    thresh = cv2.copyMakeBorder(thresh,top,down,right,left,cv2.BORDER_CONSTANT,value=(0))
    thresh = thresh[int(thresh.shape[1]/2)-20:int(thresh.shape[1]/2)+20,int(thresh.shape[0]/2)-20:int(thresh.shape[0]/2)+20]
    print(thresh.shape)
    cv2.imwrite('cells/2/%d.bmp'%index,thresh)
    index += 1