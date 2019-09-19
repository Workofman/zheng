import tensorflow as tf
import os
import cv2
import numpy as np
from sklearn.utils import shuffle


def load_images(dir_path):
    images = []
    labels = []
    lab = os.listdir(dir_path)
    for path in lab:
        for file in os.listdir(dir_path+path+'/'):
            images.append(cv2.imread(dir_path+path+'/'+file))
            labels.append(path)
    return images,labels

def prepare_data(images,labels,n_classes):

    train_X = np.array(images)
    train_Y = np.array(labels)

    index = np.arange(0,train_Y.shape[0])
    index = shuffle(index)
    train_X = train_X[index]
    train_Y = train_Y[index]
    train_Y = tf.one_hot(train_Y,n_classes)
    return train_X,train_Y


img_data,label_data= load_images('cells/')
train_x ,train_y = prepare_data(img_data,label_data,3)
# img_test_data,label_test_data = load_images('test/')
# test_x,test_y = prepare_data(img_test_data,label_test_data,1)


n_classes = 3
batch_size = 30
kernel_h=kernel_w = 5
depth_in = 3
depth_out1 = 32
depth_out2 = 64
img_size = train_x.shape[1]
n_sample = train_x.shape[0]

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
def get_small_data(inputs,batch_size):
    i=0
    while True:
        small_data=inputs[i:(batch_size+i)]
        i+=batch_size
        yield small_data

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

def conv_net(x,weights,biases,dropout):

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

    fc2 = tf.nn.dropout(fc2,dropout)

    prediction = tf.add(tf.matmul(fc2,weights['out']),biases['out'])
    return prediction

#优化器
prediction = conv_net(x,weights,bias,dropout=0.8)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y)
cross_entropy = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

#评估模型
correct_pred = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))


with tf.Session() as sess:
    tf.global_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    try:
        for i in range(20):
            if coord.should_stop():
                break
            train_x,train_y = prepare_data(img_data,label_data,3)
            train_x = get_small_data(train_x,30)
            y_ = sess.run(train_y)
            train_y = get_small_data(y_,30)
            for j in range(int(n_sample / batch_size)):
                x_ = next(train_x)
                y_ = next(train_y)
            # feed_dic = {x:train_x,y:train_y}

                sess.run(optimizer,feed_dict={x:x_,y:y_})
                if (j+1)%10 == 0:
                    loss,acc = sess.run([cross_entropy,accuracy],feed_dict={x:x_,y:y_})

                    print('Epoch:','%d'%(i+1),'cost:',loss,'train accuracy:','{:.5f}'.format(acc))
        save_model(sess,i)
    except tf.errors.OutOfRangeError:

        print('Optimization Completed')
    finally:
        coord.request_stop()
        coord.join(threads)



    # test_x = test_x
    # test_y = test_y
    # test_y = sess.run(test_y)
    # test_feed = {x: test_x, y: test_y, keep_prob: 0.8}
    # y1 = sess.run(prediction, feed_dict=test_feed)
    # test_classes = np.argmax(y1, 1)
    # print('Testing Accuracy:', sess.run(accuracy, feed_dict=test_feed))


