import tensorflow as tf
import numpy as np

#data
x_data = np.array([[88,100,70,45,50],[84,88,80,45,45],[96,100,96,50,50],[100,70,90,40,45],[70,70,70,35,35]], np.float32)
y_data = [[96],[92],[100],[88],[80]]

#constants
num_feature = len(x_data[0])
learning_rate = 0.01

#data normalization
for i in range(num_feature):
    x_data[:,i] = (x_data[:,i]-x_data[:,i].mean())/x_data[:,i].std()

#multi feature
X = tf.placeholder(tf.float32, [None, num_feature])
Y = tf.placeholder(tf.float32, [None, 1])

W = tf.Variable(tf.random_normal([num_feature, 1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

hypothesis = tf.matmul(X, W)+b

cost = tf.reduce_mean(tf.square(hypothesis - Y), -1, name="cost")
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2000):
        cost_val, hypothesis_val, _ = sess.run([cost,hypothesis,train], feed_dict={X:x_data, Y:y_data})

        if step % 20 == 0:
            print("[{}] cost {} prediction {}".format(step, cost_val, hypothesis_val))

"""
[0] cost [ 10158.43652344   8425.15234375   9758.29296875   7369.46240234   5935.81787109] prediction [[-4.78906918] [ 0.21137609] [ 1.21592486] [ 2.15442729] [ 2.9557395 ]]
[20] cost [ 154.68495178  112.96716309  104.17273712  126.10868835  115.0089035 ] prediction [[ 83.5627594 ] [ 81.37139893] [ 89.79349518] [ 76.77018738] [ 69.27577972]]
[40] cost [ 3.70892739  0.98799706  0.88595843  2.82840276  1.37214684] prediction [[ 94.07414246] [ 91.00601959] [ 99.05874634] [ 86.31821442] [ 78.82861328]]
============================================================================================================================================================================================
[1940] cost [  7.46012665e-06   1.37329102e-04   2.97572115e-05   4.30503860e-07   1.89116690e-05] prediction [[ 95.99726868] [ 92.01171875] [ 99.99454498] [ 88.00065613] [ 79.99565125]]
[1960] cost [  7.04918057e-06   1.29921595e-04   2.81967223e-05   4.10713255e-07   1.79294148e-05] prediction [[ 95.99734497] [ 92.01139832] [ 99.99468994] [ 88.00064087] [ 79.99576569]]
[1980] cost [  6.68928260e-06   1.23227073e-04   2.67571304e-05   3.91388312e-07   1.70362764e-05] prediction [[ 95.99741364] [ 92.01110077] [ 99.99482727] [ 88.00062561] [ 79.9958725 ]]
"""