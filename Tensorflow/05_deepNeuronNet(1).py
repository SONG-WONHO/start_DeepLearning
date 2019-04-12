import tensorflow as tf
import numpy as np

#XOR problem
x_data = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
y_data = np.array([[0],[1],[1],[0]], dtype=np.float32)

learning_rate = 0.01

X = tf.placeholder(tf.float32, [4,2])
Y = tf.placeholder(tf.float32, [4,1])

W = tf.Variable(tf.random_normal([2,1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

logits = tf.matmul(X, W) + b
hypothesis = tf.nn.sigmoid(logits)

cost = -tf.reduce_mean(tf.reduce_sum(Y*tf.log(hypothesis)+(1-Y)*tf.log(1-hypothesis)))
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(20000):
        cost_val, hypothesis_val, _ = sess.run([cost, hypothesis, train], feed_dict={X:x_data, Y:y_data})

        if step % 2000 == 0:
            print("[{}] cost {} hypothesis{}".format(step, cost_val, hypothesis_val))

"""
[0] cost 3.102274179458618 hypothesis[[ 0.30139181] [ 0.3955116 ] [ 0.24100791] [ 0.32504442]]
==============================================================================================
[18000] cost 2.7725887298583984 hypothesis[[ 0.5] [ 0.5] [ 0.5] [ 0.5]]
"""

#to solve XOR problem
x_data = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
y_data = np.array([[0],[1],[1],[0]], dtype=np.float32)

learning_rate = 0.01

X = tf.placeholder(tf.float32, [4,2])
Y = tf.placeholder(tf.float32, [4,1])

with tf.name_scope("layer1") as scope:
    W1 = tf.Variable(tf.random_normal([2,2]), name="weight1")
    b1 = tf.Variable(tf.random_normal([2]), name="bias1")
    layer1 = tf.nn.sigmoid(tf.matmul(X, W1) + b1)

with tf.name_scope("hypothesis") as scope:
    W2 = tf.Variable(tf.random_normal([2,1]), name="weight2")
    b2 = tf.Variable(tf.random_normal([1]), name="bias2")
    hypothesis = tf.nn.sigmoid(tf.matmul(layer1, W2) + b2)

cost = -tf.reduce_mean(tf.reduce_sum(Y*tf.log(hypothesis)+(1-Y)*tf.log(1-hypothesis)))
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(20000):
        cost_val, hypothesis_val, _ = sess.run([cost, hypothesis, train], feed_dict={X:x_data, Y:y_data})

        if step % 2000 == 0:
            print("[{}] cost {} hypothesis{}".format(step, cost_val, hypothesis_val))

"""
[0] cost 3.6086764335632324 hypothesis[[ 0.23846868] [ 0.19450448] [ 0.22089356] [ 0.17211264]]
====================================================================================================
[18000] cost 0.07535417377948761 hypothesis[[ 0.01831727] [ 0.98349243] [ 0.97555089] [ 0.01534975]]
"""

#but this methoid has problem
x_data = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
y_data = np.array([[0],[1],[1],[0]], dtype=np.float32)

learning_rate = 0.01

X = tf.placeholder(tf.float32, [4,2])
Y = tf.placeholder(tf.float32, [4,1])

with tf.name_scope("layer1") as scope:
    W1 = tf.Variable(tf.random_normal([2,2]), name="weight1")
    b1 = tf.Variable(tf.random_normal([2]), name="bias1")
    layer1 = tf.nn.sigmoid(tf.matmul(X, W1) + b1)

with tf.name_scope("layer2") as scope:
    W2 = tf.Variable(tf.random_normal([2,2]), name="weight2")
    b2 = tf.Variable(tf.random_normal([2]), name="bias2")
    layer2 = tf.nn.sigmoid(tf.matmul(layer1, W2) + b2)

with tf.name_scope("layer3") as scope:
    W3 = tf.Variable(tf.random_normal([2,2]), name="weight3")
    b3 = tf.Variable(tf.random_normal([2]), name="bias3")
    layer3 = tf.nn.sigmoid(tf.matmul(layer2, W3) + b3)

with tf.name_scope("layer4") as scope:
    W4 = tf.Variable(tf.random_normal([2,2]), name="weight4")
    b4 = tf.Variable(tf.random_normal([2]), name="bias4")
    layer4 = tf.nn.sigmoid(tf.matmul(layer3, W4) + b4)

with tf.name_scope("layer5") as scope:
    W5 = tf.Variable(tf.random_normal([2,2]), name="weight5")
    b5 = tf.Variable(tf.random_normal([2]), name="bias5")
    layer5 = tf.nn.sigmoid(tf.matmul(layer4, W5) + b5)

with tf.name_scope("layer6") as scope:
    W6 = tf.Variable(tf.random_normal([2,2]), name="weight6")
    b6 = tf.Variable(tf.random_normal([2]), name="bias6")
    layer6 = tf.nn.sigmoid(tf.matmul(layer5, W6) + b6)

with tf.name_scope("layer7") as scope:
    W7 = tf.Variable(tf.random_normal([2,2]), name="weight7")
    b7 = tf.Variable(tf.random_normal([2]), name="bias7")
    layer7 = tf.nn.sigmoid(tf.matmul(layer6, W7) + b7)

with tf.name_scope("layer8") as scope:
    W8 = tf.Variable(tf.random_normal([2,2]), name="weight8")
    b8 = tf.Variable(tf.random_normal([2]), name="bias8")
    layer8 = tf.nn.sigmoid(tf.matmul(layer7, W8) + b8)

with tf.name_scope("layer9") as scope:
    W9 = tf.Variable(tf.random_normal([2,2]), name="weight9")
    b9 = tf.Variable(tf.random_normal([2]), name="bias9")
    layer9 = tf.nn.sigmoid(tf.matmul(layer8, W9) + b9)

with tf.name_scope("hypothesis") as scope:
    W10 = tf.Variable(tf.random_normal([2,1]), name="weight10")
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
[0] cost 2.915419101715088 hypothesis[[ 0.36873242] [ 0.36873242] [ 0.36873242] [ 0.36873242]]
====================================================================================================
[18000] cost 2.7725887298583984 hypothesis[[ 0.49999988] [ 0.49999988] [ 0.49999988] [ 0.49999994]]
"""