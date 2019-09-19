import tensorflow as tf
import os
import cv2
import warnings
import time
import numpy as np
from sklearn.utils import shuffle


start = time.clock()
n_classes = 3
batch_size = 30
kernel_h=kernel_w = 5
depth_in = 3
depth_out1 = 32
depth_out2 = 64
img_size = 40

#占位符
x = tf.placeholder(tf.float32,[None,40,40,3])
y = tf.placeholder(tf.float32,[None,n_classes])
keep_prob = tf.placeholder(tf.float32)

fla = int((img_size*img_size/16)*depth_out2)

#权重
weights = {
    'conv1_w':tf.Variable(tf.random.normal([kernel_h,kernel_w,depth_in,depth_out1])),
    'conv2_w':tf.Variable(tf.random.normal([kernel_h,kernel_w,depth_out1,depth_out2])),
    'fc1_w':tf.Variable(tf.random.normal([fla,1024])),
    'fc2_w':tf.Variable(tf.random.normal([1024,512])),
    'out':tf.Variable(tf.random.normal([512,n_classes]))
}
#偏置
bias = {
    'conv1_b':tf.Variable(tf.random.normal([depth_out1])),
    'conv2_b':tf.Variable(tf.random.normal([depth_out2])),
    'fc1_b':tf.Variable(tf.random.normal([1024])),
    'fc2_b':tf.Variable(tf.random.normal([512])),
    'out':tf.Variable(tf.random.normal([n_classes]))
}
def save_model(sess, step):
    MODEL_SAVE_PATH = "cells_model/"
    MODEL_NAME = "model.ckpt"
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=step)
#定义卷积层
def conv2d(x,W,b,stride=1):
    x = tf.nn.conv2d(x,W,strides=[1,stride,stride,1],padding='SAME')
    x = tf.nn.bias_add(x,b)
    return tf.nn.relu(x)

#定义Pooling层
def maxpool2d(x,stride=2):
    return tf.nn.max_pool2d(x,ksize=[1,stride,stride,1],strides=[1,stride,stride,1],padding='SAME')


#定义卷积网络

def conv_net(x,weights,biases):

    #conv1
    conv1 = conv2d(x,weights['conv1_w'],biases['conv1_b'])
    conv1 = maxpool2d(conv1)

    #conv2
    conv2 = conv2d(conv1,weights['conv2_w'],biases['conv2_b'])
    conv2 = maxpool2d(conv2)


    #fc1
    flatten = tf.reshape(conv2,[-1,fla])
    fc1 = tf.add(tf.matmul(flatten,weights['fc1_w']),biases['fc1_b'])
    fc1 = tf.nn.relu(fc1)

    #fc2
    fc2 = tf.add(tf.matmul(fc1,weights['fc2_w']),biases['fc2_b'])
    fc2 = tf.nn.relu(fc2)

    # fc2 = tf.nn.dropout(fc2,dropout)

    prediction = tf.add(tf.matmul(fc2,weights['out']),biases['out'])
    return prediction


#优化器


# cross_entropy = tf.nn.softmax(prediction)
# cross_entropy = tf.reduce_mean(cross_entropy)
# optimizer = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
#评估模型
# correct_pred = tf.argmax(prediction,1)
# accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))


with tf.Session() as sess:
   warnings.filterwarnings("ignore")
   # sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))
   # saver = tf.train.import_meta_graph('model/model.ckpt-99.meta')
   saver = tf.train.Saver()
   saver.restore(sess, "cells_model/model.ckpt-19")
   rc = 0
   wc = 0
   for img in os.listdir('cells_test/'):
       # image = np.array(image)
       # image.reshape(image,[28,28,3])
       image1 = cv2.imread('cells_test/' + img)
       image = tf.cast(image1, tf.float32)
       image = tf.reshape(image, [-1, 40,40, 3])
       # feed_dic = {x: train_x, y: train_y}
       # y_ = sess.run(train_y)
       prediction = conv_net(image, weights, bias)
       # sess.run(optimizer, feed_dict={x: train_x})
       correct_pred = tf.argmax(prediction, 1)
       label = sess.run(correct_pred)
       # loss = cross_entropy[0][label]
       if label[0]  == 0:
           rc += 1
           c = int(img.split('.')[0])
           cv2.imwrite('rb_pre/%d.bmp'%c,image1)
       else:
           wc += 1
           c = int(img.split('.')[0])
           cv2.imwrite('wb_pre/%d.bmp'%c,image1)
   # print('image:',img,'label:', label[0])
   print('rc:',rc,'wc:',wc)
   end = time.clock()
   print('time:',end-start)



    # test_x = test_x
    # test_y = test_y
    # test_y = sess.run(test_y)
    # test_feed = {x: test_x, y: test_y, keep_prob: 0.8}
    # y1 = sess.run(prediction, feed_dict=test_feed)
    # test_classes = np.argmax(y1, 1)
    # print('Testing Accuracy:', sess.run(accuracy, feed_dict=test_feed))


