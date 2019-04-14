from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import layers
import tensorflow as tf

# load data
mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)

# constants
learning_rate = 0.01
conv_strides = [1, 1, 1, 1]
pool_strides = [1, 2, 2, 1]
training_epoch = 30
batch_size = 100

X = tf.placeholder(tf.float32, [None, 28 * 28])
X_image = tf.reshape(X, [-1, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 10])

# convolution layer 1
with tf.name_scope("layer1") as scope:
    W1 = tf.get_variable("W1", [3, 3, 1, 16], tf.float32, layers.xavier_initializer())
    layer1 = tf.nn.conv2d(X_image, W1, conv_strides, 'SAME')
    layer1 = tf.nn.relu(layer1)
    layer1 = tf.nn.max_pool(layer1, [1, 2, 2, 1], pool_strides, 'SAME')

# convolution layer 2
with tf.name_scope("layer2") as scope:
    W2 = tf.get_variable("W2", [3, 3, 16, 32], tf.float32, layers.xavier_initializer())
    layer2 = tf.nn.conv2d(layer1, W2, conv_strides, 'SAME')
    layer2 = tf.nn.relu(layer2)
    layer2 = tf.nn.max_pool(layer2, [1, 2, 2, 1], pool_strides, 'SAME')

# convolution layer 3
with tf.name_scope("layer3") as scope:
    W3 = tf.get_variable("W3", [3, 3, 32, 64], tf.float32, layers.xavier_initializer())
    layer3 = tf.nn.conv2d(layer2, W3, conv_strides, 'SAME')
    layer3 = tf.nn.relu(layer3)
    layer3 = tf.nn.max_pool(layer3, [1, 2, 2, 1], pool_strides, 'SAME')
    layer3 = tf.reshape(layer3, [-1, 4 * 4 * 64])

# fully connected layer 1
with tf.name_scope("layer4") as scope:
    W4 = tf.get_variable("W4", [4 * 4 * 64, 64], tf.float32, layers.xavier_initializer())
    b4 = tf.Variable(tf.random_normal([64]), name="bias1")
    layer4 = tf.nn.relu(tf.matmul(layer3, W4) + b4)

# fully connected layer 2
with tf.name_scope("hypothesis") as scope:
    W5 = tf.get_variable("W5", [64, 10], tf.float32, layers.xavier_initializer())
    b5 = tf.Variable(tf.random_normal([10]), name="bias2")
    logits = tf.matmul(layer4, W5) + b5

# cost function
cost = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
cost = tf.reduce_mean(cost)
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# verification
prediction = tf.argmax(logits, -1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(Y, -1)), tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epoch):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)

            cost_val, accuracy_val, _ = sess.run([cost, accuracy, train], feed_dict={X: batch_x, Y: batch_y})

            avg_cost += cost_val / total_batch

        print("[epoch : {}] cost {} accuracy {}".format(epoch, avg_cost, accuracy_val))

    print("accuracy : {} ".format(sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels})))

"""
<train data>
[epoch : 0] cost 1.6416320951418424 accuracy 0.8700000047683716
[epoch : 1] cost 0.33713997257026773 accuracy 0.9300000071525574
==================================================================
[epoch : 28] cost 0.03144095546184956 accuracy 1.0
[epoch : 29] cost 0.03062903371745383 accuracy 0.9900000095367432

<test data>
accuracy : 0.9872999787330627
"""
