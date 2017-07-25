import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers

#to solve convergence zero problem, use ReLU function
x_data = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
y_data = np.array([[0],[1],[1],[0]], dtype=np.float32)

learning_rate = 0.01

X = tf.placeholder(tf.float32, [4,2])
Y = tf.placeholder(tf.float32, [4,1])

with tf.name_scope("layer1") as scope:
    W1 = tf.get_variable("W1", [2,2], tf.float32, layers.xavier_initializer())
    b1 = tf.Variable(tf.random_normal([2]), name="bias1")
    layer1 = tf.nn.relu(tf.matmul(X, W1) + b1)

with tf.name_scope("layer2") as scope:
    W2 = tf.get_variable("W2", [2,2], tf.float32, layers.xavier_initializer())
    b2 = tf.Variable(tf.random_normal([2]), name="bias2")
    layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)

with tf.name_scope("layer3") as scope:
    W3 = tf.get_variable("W3", [2,2], tf.float32, layers.xavier_initializer())
    b3 = tf.Variable(tf.random_normal([2]), name="bias3")
    layer3 = tf.nn.relu(tf.matmul(layer2, W3) + b3)

with tf.name_scope("layer4") as scope:
    W4 = tf.get_variable("W4", [2,2], tf.float32, layers.xavier_initializer())
    b4 = tf.Variable(tf.random_normal([2]), name="bias4")
    layer4 = tf.nn.relu(tf.matmul(layer3, W4) + b4)

with tf.name_scope("layer5") as scope:
    W5 = tf.get_variable("W5", [2,2], tf.float32, layers.xavier_initializer())
    b5 = tf.Variable(tf.random_normal([2]), name="bias5")
    layer5 = tf.nn.relu(tf.matmul(layer4, W5) + b5)

with tf.name_scope("layer6") as scope:
    W6 = tf.get_variable("W6", [2,2], tf.float32, layers.xavier_initializer())
    b6 = tf.Variable(tf.random_normal([2]), name="bias6")
    layer6 = tf.nn.relu(tf.matmul(layer5, W6) + b6)

with tf.name_scope("layer7") as scope:
    W7 = tf.get_variable("W7", [2,2], tf.float32, layers.xavier_initializer())
    b7 = tf.Variable(tf.random_normal([2]), name="bias7")
    layer7 = tf.nn.relu(tf.matmul(layer6, W7) + b7)

with tf.name_scope("layer8") as scope:
    W8 = tf.get_variable("W8", [2,2], tf.float32, layers.xavier_initializer())
    b8 = tf.Variable(tf.random_normal([2]), name="bias8")
    layer8 = tf.nn.relu(tf.matmul(layer7, W8) + b8)

with tf.name_scope("layer9") as scope:
    W9 = tf.get_variable("W9", [2,2], tf.float32, layers.xavier_initializer())
    b9 = tf.Variable(tf.random_normal([2]), name="bias9")
    layer9 = tf.nn.relu(tf.matmul(layer8, W9) + b9)

with tf.name_scope("hypothesis") as scope:
    W10 = tf.get_variable("W10", [2,1], tf.float32, layers.xavier_initializer())
    b10 = tf.Variable(tf.random_normal([1]), name="bias10")
    hypothesis = tf.nn.sigmoid(tf.matmul(layer9, W10) + b10)

cost = -tf.reduce_mean(tf.reduce_sum(Y*tf.log(hypothesis)+(1-Y)*tf.log(1-hypothesis)))
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(20000):
        cost_val, hypothesis_val, _ = sess.run([cost, hypothesis, train], feed_dict={X:x_data, Y:y_data})

        if step % 2000 == 0:
            print("[{}] cost {} hypothesis{}".format(step, cost_val, hypothesis_val))

"""
<case1>
[0] cost 3.0137195587158203 hypothesis[[ 0.34374678] [ 0.36928704] [ 0.29200172] [ 0.30603549]]
=========================================================================================================================
[18000] cost 0.000872039410751313 hypothesis[[  6.37881312e-05] [  9.99627829e-01] [  9.99627829e-01] [  6.37993217e-05]]

<case2>
[0] cost 2.805219888687134 hypothesis[[ 0.56360662] [ 0.56360662] [ 0.56360662] [ 0.56360662]]
===================================================================================================
[18000] cost 2.7725887298583984 hypothesis[[ 0.50000012] [ 0.50000012] [ 0.50000012] [ 0.50000012]]

<case1>:<case> = 1:10 depends on initialized weight
"""