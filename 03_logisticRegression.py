import tensorflow as tf
import numpy as np

#load data
samples = np.loadtxt("./samples/diabetes.csv", dtype=np.float32, delimiter=',')

#constants
number_feature = len(samples[0]) -1
learning_rate = 0.01

#nomalize data
for i in range(number_feature):
    samples[:,i] = (samples[:,i]-samples[:,i].mean())/samples[:,i].std()

#separate train, test
train_size = int(len(samples) * 0.7)

train = samples[:train_size]
x_train = train[:,:-1]
y_train = train[:,[-1]]

test = samples[train_size:]
x_test = test[:,:-1]
y_test = test[:,[-1]]

X = tf.placeholder(tf.float32, [None, number_feature])
Y = tf.placeholder(tf.float32, [None, 1])

W = tf.Variable(tf.random_normal([number_feature, 1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

logits = tf.matmul(X, W) + b
hypothesis = tf.nn.sigmoid(logits)

#logistic cost function
cost = -tf.reduce_mean(Y*tf.log(hypothesis)+(1-Y)*tf.log(1-hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

#verifiction
prediction = tf.cast(hypothesis > 0.5, tf.float32, "prediction")
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, Y), tf.float32, "accuracy"))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    #train
    for step in range(20000):
        cost_val, accuracy_val, _ = sess.run([cost, accuracy, train], feed_dict={X:x_train, Y:y_train})

        if step % 2000 == 0:
            print("[{}] cost {} accuracy {}".format(step,cost_val, accuracy_val))

    #test
    print("***Test data***")
    for p,y in zip(sess.run(prediction,feed_dict={X:x_test, Y:y_test}), y_test):
        print("Prediction {} True_Y {}".format(p, y))

    print("accuracy {}".format(sess.run(accuracy,feed_dict={X:x_test,Y:y_test})))

"""
***Train***
[0] cost 0.9495219588279724 accuracy 0.5512104034423828
[2000] cost 0.48801669478416443 accuracy 0.769087553024292
[4000] cost 0.48530086874961853 accuracy 0.7783985137939453
[6000] cost 0.48516276478767395 accuracy 0.7765362858772278
[8000] cost 0.48515310883522034 accuracy 0.7765362858772278
[10000] cost 0.4851524829864502 accuracy 0.7765362858772278
[12000] cost 0.48515236377716064 accuracy 0.7765362858772278
[14000] cost 0.48515236377716064 accuracy 0.7765362858772278
[16000] cost 0.4851524233818054 accuracy 0.7765362858772278
[18000] cost 0.48515230417251587 accuracy 0.7765362858772278

***Test data***
Prediction [ 0.] True_Y [ 0.]
Prediction [ 0.] True_Y [ 0.]
Prediction [ 0.] True_Y [ 1.]
==============================
Prediction [ 0.] True_Y [ 0.]
Prediction [ 0.] True_Y [ 1.]
Prediction [ 0.] True_Y [ 0.]
accuracy 0.7922077775001526
"""