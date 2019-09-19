import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
import cv2
import os


# for path in os.listdir('mnist_digits_images/0/1.bmp'):
#     print(path)
# index = 38
# for path in os.listdir('cells/2/'):
#     img  = cv2.imread('cells/2/'+path)
#     (h,w) = img.shape[:2]
#     center = (h/2,w/2)
#     M = cv2.getRotationMatrix2D(center, 90, 1)  # 旋转缩放矩阵：(旋转中心，旋转角度，缩放因子)
#     rotated = cv2.warpAffine(img, M, (h,w))
#     cv2.imwrite('cells/2/%d.bmp'%index,rotated)
#     index += 1
#     if index == 100:
#         break
# print(img)
# print(img.shape)
# print(a.shape)
# print(a.shape[0])

def forword(X,w,b):
    Z1 = w*X + b
    A1 = 1/(1+np.exp(-Z1))
    return A1,Z1
def backforword(A1,Z1,X,Y):
    dZ1 = A1 - Y
    dw = dZ1*X
    db = dZ1
    return dZ1,dw,db