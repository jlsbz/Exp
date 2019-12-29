""" import your model here """
import tensorflow as tf
import numpy as np
""" your model should support the following code """


def weight_variable(shape):
    initial = tf.random_normal(shape, stddev=0.025)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.025, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1],
            padding='SAME')


save_file = './model_mnist.ckpt'
tf.reset_default_graph()
# saver = tf.train.Saver()



# input
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)

# first layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1 )
h_conv1 = tf.nn.dropout(h_conv1, keep_prob)

# Residual
W_conv1 = weight_variable([5, 5, 32, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(h_conv1, W_conv1) + b_conv1 + h_conv1 )
h_conv1 = tf.nn.dropout(h_conv1, keep_prob)

# Residual
W_conv1 = weight_variable([5, 5, 32, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(h_conv1, W_conv1) + b_conv1 + h_conv1 )
h_conv1 = tf.nn.dropout(h_conv1, keep_prob)

# pooling
h_pool1 = max_pool_2x2(h_conv1)


# second layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_conv2 = tf.nn.dropout(h_conv2, keep_prob)

# Residual
W_conv2 = weight_variable([5, 5, 64, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_conv2, W_conv2) + b_conv2 + h_conv2)
h_conv2 = tf.nn.dropout(h_conv2, keep_prob)

# Residual
W_conv2 = weight_variable([5, 5, 64, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_conv2, W_conv2) + b_conv2 + h_conv2)
h_conv2 = tf.nn.dropout(h_conv2, keep_prob)

# pooling
h_pool1 = max_pool_2x2(h_conv2)

# # third layer
# W_conv2 = weight_variable([5, 5, 64, 128])
# b_conv2 = bias_variable([128])

# h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# h_conv2 = tf.nn.dropout(h_conv2, keep_prob)

# # pooling
# h_pool1 = max_pool_2x2(h_conv2)

# densely connected layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool1, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# readout layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# loss
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

####################
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 5e-3
learning_rate = tf.compat.v1.train.exponential_decay(starter_learning_rate,
    global_step,400, 0.75, staircase=True)
# Passing global_step to minimize() will increment it at each step.
# learning_step = (
    # tf.compat.v1.train.GradientDescentOptimizer(learning_rate)
    # .minimize(...my loss..., global_step=global_step)
# )

train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy,global_step=global_step)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
dx, = tf.gradients(cross_entropy,x)


# Get the mnist dataset (use tensorflow here)
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST/", one_hot=True)
saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, save_file)
    
    # sess.run(tf.global_variables_initializer())
    # for i in range(4000):
    #     if i % 50 == 0:
    #         batch_ = mnist.test.next_batch(5000)
    #         train_accuracy, loss__ = sess.run([accuracy,cross_entropy],feed_dict = { x: batch_[0],
    #                                        y_: batch_[1], keep_prob: 1.0})
    #         print('step %d, testing accuracy %g \t %g' % (i, train_accuracy,loss__))
    #     batch = mnist.train.next_batch(500)
    #     if i % 50 == 0:
    #         # batch = mnist.train.next_batch(5000)
    #         train_accuracy, loss__ = sess.run([accuracy,cross_entropy],feed_dict = { x: batch[0],
    #                                        y_: batch[1], keep_prob: 1.0})
    #         print('step %d, trainning accuracy %g \t %g' % (i, train_accuracy,loss__))

    #     train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.75})

    batch = list(mnist.test.next_batch(100))
    
    for i in range(1000):
        delta_x = dx.eval(feed_dict = {x:batch[0], y_:batch[1],keep_prob:1.0})
        # print(delta_x.shape)
        # print(batch[0].shape)
        batch[0] = batch[0] + delta_x*1e-1
        batch[0] = np.maximum(batch[0],0)
        batch[0] = np.minimum(batch[0],1)
        ans = accuracy.eval(feed_dict={ x:batch[0]+delta_x*10,
                                        y_: batch[1], keep_prob: 1.0})
        print('test accuracy %6d %6g' % (i,ans))
        # print(type(batch[0]))
        assert ans > 0.72
    acc , result =  sess.run([accuracy, correct_prediction],feed_dict = { x:batch[0]+delta_x*10,
                                        y_: batch[1], keep_prob: 1.0})
    print(result)
    for i in range(100):
        if (result[i]==0):
            print(batch[0][i].reshape((28,28)))
            a = batch[0][i].reshape((28,28))
            
            pass
    # saver.save(sess, save_file)
