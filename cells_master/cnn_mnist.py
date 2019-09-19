import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/path/to/mnist_data',one_hot=True)

xs = tf.placeholder(tf.float32,[None,784])
ys = tf.placeholder(tf.float32,[None,10])
x_image = tf.reshape(xs,[-1,28,28,1])

def weights_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
def pooling(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#conv1
W_conv1 = weights_variable([5,5,1,32])
b_conv1 = bias_variable([32])
l1 = tf.nn.relu(conv2d(x_image,W_conv1)+ b_conv1)
#pool1
p1 = pooling(l1)
#conv2
W_conv2 = weights_variable([5,5,32,64])
b_conv2 = bias_variable([64])
l2 = tf.nn.relu(conv2d(p1,W_conv2) + b_conv2)
#pool2
p2 = pooling(l2)
#FC1
W_fc1 = weights_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])
p2_ = tf.reshape(p2,[-1,7*7*64])
fc1 = tf.nn.relu(tf.matmul(p2_,W_fc1)+b_fc1)
#FC2
W_fc2 = weights_variable([1024,10])
b_fc2 = bias_variable([10])
fc2 = tf.matmul(fc1,W_fc2) + b_fc2

loss = tf.nn.softmax_cross_entropy_with_logits(labels=ys,logits=fc2)
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)
c_p = tf.equal(tf.argmax(ys,1),tf.argmax(fc2,1))
acc = tf.reduce_mean(tf.cast(c_p,tf.float32))
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(10000):
        x_batch,y_batch = mnist.train.next_batch(128)
        sess.run(train_step,feed_dict={xs:x_batch,ys:y_batch})
        if i%50 == 0:
            print('step:',i,'acc:',sess.run(acc,feed_dict={xs:mnist.test.images,ys:mnist.test.labels}))