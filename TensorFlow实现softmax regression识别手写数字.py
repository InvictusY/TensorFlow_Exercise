# -*- coding: utf-8 -*-
'''
1.定义算法公式，也就是神经网络forward时的计算
2.定义loss，选定优化器，并指定优化器优化loss
3.迭代的对数据进行训练
4.在测试集或者验证集上对准确类率进行评测
'''

# 加载MNIST数据集
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# 查看MNIST数据集情况
print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)

# 载入TensorFlow模块，创建一个session，不同session之间的数据和运算都是相互独立的
import tensorflow as tf
sess = tf.InteractiveSession()
# 创建输入数据的地方，None表示不限条数的输入
x = tf.placeholder(tf.float32, [None,784])
w = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, w) + b)

# y_真实的label
y_ = tf.placeholder(tf.float32, [None,10])

# 定义loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# 定义一个优化算法
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# 全局参数初始化
init = tf.global_variables_initializer()
sess.run(init)

# 训练
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys})

# 对模型准确率进行验证
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))


