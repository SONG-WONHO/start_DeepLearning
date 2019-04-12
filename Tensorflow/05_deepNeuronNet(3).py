from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from tensorflow.contrib import layers

mnist = input_data.read_data_sets("./MNIST_data", one_hot='True')

#constants
learning_rate = 0.1
training_epoch = 30
batch_size = 100

X = tf.placeholder(tf.float32, [None, 28*28])
Y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

#layer1
with tf.name_scope("layer1") as scope:
    W1 = tf.get_variable("W1", [28*28, 256], tf.float32, layers.xavier_initializer())
    b1 = tf.Variable(tf.random_normal([256]), name="bias1")

    layer1 = tf.nn.relu(tf.matmul(X, W1) + b1)
    layer1 = tf.nn.dropout(layer1, keep_prob)

#layer2
with tf.name_scope("layer2") as scope:
    W2 = tf.get_variable("W2", [256,128], tf.float32, layers.xavier_initializer())
    b2 = tf.Variable(tf.random_normal([128]), name="bias2")

    layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)
    layer2 = tf.nn.dropout(layer2, keep_prob)

#layer3
with tf.name_scope("layer3") as scope:
    W3 = tf.get_variable("W3", [128,64], tf.float32, layers.xavier_initializer())
    b3 = tf.Variable(tf.random_normal([64]), name="bias3")

    layer3 = tf.nn.relu(tf.matmul(layer2, W3) + b3)
    layer3 = tf.nn.dropout(layer3, keep_prob)

#layer4
with tf.name_scope("layer4") as scope:
    W4 = tf.get_variable("W4", [64,64], tf.float32, layers.xavier_initializer())
    b4 = tf.Variable(tf.random_normal([64]), name="bias4")

    layer4 = tf.nn.relu(tf.matmul(layer3, W4) + b4)
    layer4 = tf.nn.dropout(layer4, keep_prob)

#layer5
with tf.name_scope("layer5") as scope:
    W5 = tf.get_variable("W5", [64,64], tf.float32, layers.xavier_initializer())
    b5 = tf.Variable(tf.random_normal([64]), name="bias5")

    layer5 = tf.nn.relu(tf.matmul(layer4, W5) + b5)
    layer5 = tf.nn.dropout(layer5, keep_prob)

#layer6
with tf.name_scope("layer6") as scope:
    W6 = tf.get_variable("W6", [64,64], tf.float32, layers.xavier_initializer())
    b6 = tf.Variable(tf.random_normal([64]), name="bias6")

    layer6 = tf.nn.relu(tf.matmul(layer5, W6) + b6)
    layer6 = tf.nn.dropout(layer6, keep_prob)

#layer7
with tf.name_scope("layer7") as scope:
    W7 = tf.get_variable("W7", [64,64], tf.float32, layers.xavier_initializer())
    b7 = tf.Variable(tf.random_normal([64]), name="bias7")

    layer7 = tf.nn.relu(tf.matmul(layer6, W7) + b7)
    layer7 = tf.nn.dropout(layer7, keep_prob)

#layer8
with tf.name_scope("layer8") as scope:
    W8 = tf.get_variable("W8", [64,64], tf.float32, layers.xavier_initializer())
    b8 = tf.Variable(tf.random_normal([64]), name="bias8")

    layer8 = tf.nn.relu(tf.matmul(layer7, W8) + b8)
    layer8 = tf.nn.dropout(layer8, keep_prob)

#hypothesis
with tf.name_scope("hypothesis") as scope:
    W9 = tf.get_variable("W9", [64,10], tf.float32, layers.xavier_initializer())
    b9 = tf.Variable(tf.random_normal([10]), name="bias9")

    logits = tf.matmul(layer8, W9) + b9

#cost function
cost = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
cost = tf.reduce_mean(cost)

train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

#verification
prediction = tf.argmax(logits, -1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(Y, -1)), tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epoch):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            cost_val, _ = sess.run([cost, train], feed_dict={X:batch_x,Y:batch_y, keep_prob:0.7})
            avg_cost += cost_val/total_batch

        print("[epoch : {}]  cost {}".format(epoch, avg_cost))

    print("accuracy : {}".format(sess.run(accuracy, feed_dict={X:mnist.test.images, Y:mnist.test.labels, keep_prob:1})))


"""
<case1: three layers>
[epoch : 0]  cost 1.0906419989737604
[epoch : 1]  cost 0.44141744659705584
======================================
[epoch : 28]  cost 0.10395035373554991
[epoch : 29]  cost 0.10062386982142925
accuracy : 0.963100016117096

<case2: nine layers>
[epoch : 0]  cost 1.739546862949024
[epoch : 1]  cost 0.6752396106178111
=======================================
[epoch : 28]  cost 0.035486407370772216
[epoch : 29]  cost 0.034942295421338226
accuracy : 0.9692000150680542

<case3: dropout>
[epoch : 0]  cost 2.3446306311000464
[epoch : 1]  cost 1.8724166965484632
=======================================
[epoch : 28]  cost 0.25287483566186647
[epoch : 29]  cost 0.2510733996196226
accuracy : 0.961899995803833
"""