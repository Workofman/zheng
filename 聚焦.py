# -*- coding: GBK -*-
import cv2

from matplotlib import pyplot as plt
import numpy as np
import time
import tkinter
from tkinter import filedialog
import os


start = time.clock()
list = []
index = []
window = tkinter.Tk()



def get_path():
    window.withdraw()
    path = filedialog.askdirectory()
    return path
def load(path):
    for imgfile in os.listdir(path):
        image = cv2.imdecode(np.fromfile(path+imgfile,dtype=np.uint8),0)
        # image = cv2.GaussianBlur(image,(9,9),0)
        imgVar1 = cv2.Laplacian(image, cv2.CV_64F).var()


        if imgVar1>=300:
            blurred = cv2.GaussianBlur(image, (9, 9), 0)#高斯去噪
            gradx = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=1, dy=0)
            grady = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=0, dy=1)

            gradient = cv2.subtract(gradx, grady)
            gradient = cv2.convertScaleAbs(gradient)#边缘检测
            blurred = cv2.GaussianBlur(gradient, (9, 9), 0)
            (_, thresh) = cv2.threshold(blurred, 20, 255, cv2.THRESH_BINARY)#二值化
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            closed = cv2.erode(closed, None, iterations=1)
            closed = cv2.dilate(closed, None, iterations=1)#膨胀
            (_, cnt, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            imgvar = imgVar1 - (len(cnt)*2.5)
            list.append(imgvar)
            index.append(imgfile)

    x = np.linspace(0, len(list), len(list))
    index_ = list.index(max(list))
    # image = cv2.imread(index[index_])
    plt.figure(figsize=(8, 4))
    plt.plot(x, list, label="img", color="red", linewidth=2)
    end = time.clock()
    print(end - start)
    print(index[index_])
    plt.show()


if __name__ == '__main__':
    path = get_path()+'/'
    load(path)
