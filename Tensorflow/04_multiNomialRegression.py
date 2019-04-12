import tensorflow as tf
import numpy as np

# load data
sample1 = np.genfromtxt("./samples/Iris.csv", dtype=np.float32, delimiter=',')[:, 1:-1]
sample2 = np.genfromtxt("./samples/Iris.csv", dtype=np.str, delimiter=',')[:, [-1]]
sample2 = sample2.flatten()

# index to character
idx2char = []

for i in range(len(sample2)):
    idx2char.append(sample2[i])

idx2char = list(set(idx2char))

# character to index by dictionary
char2idx = {c: i for i, c in enumerate(idx2char)}

# split in to odd and even
index = np.arange(len(sample2))

# train data
x_train = sample1[index % 2 == 0]
y_train = sample2[index % 2 == 0]
y_train = [char2idx[c] for c in y_train]

# test data
x_test = sample1[index % 2 == 1]
y_test = sample2[index % 2 == 1]
y_test = [char2idx[c] for c in y_test]

# constants
num_feature = len(sample1[0])
num_classes = 3
learning_rate = 0.01

X = tf.placeholder(tf.float32, [None, num_feature])
Y = tf.placeholder(tf.int32, [None])

# Y_one_hot
Y_one_hot = tf.one_hot(Y, num_classes)

W = tf.Variable(tf.random_normal([num_feature, num_classes]), name="weight")
b = tf.Variable(tf.random_normal([num_classes]), name="bias")

logits = tf.matmul(X, W) + b

# multiNomialRegression cost function
cost = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot)
cost = tf.reduce_mean(cost)
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

prediction = tf.argmax(logits, -1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(Y_one_hot, -1)), tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(20000):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_train, Y: y_train})

        if step % 2000 == 0:
            print("[{}] cost {}0".format(step, cost_val))

    for p, y in zip(sess.run(prediction, feed_dict={X: x_test, Y: y_test}), y_test):
        print("Prediction {} True_Y {} Predicted_Iris {}".format(p, y, idx2char[p]))

    print(sess.run(accuracy, feed_dict={X: x_test, Y: y_test}))

"""
[0] cost 3.18933439254760740
[2000] cost 0.248512089252471920
[4000] cost 0.173864498734474180
[6000] cost 0.140997543931007390
[8000] cost 0.122119165956974030
[10000] cost 0.109669417142868040
[12000] cost 0.100737161934375760
[14000] cost 0.093954540789127350
[16000] cost 0.08859033882617950
[18000] cost 0.08421567082405090

Prediction 1 True_Y 1 Predicted_Iris Iris-setosa
Prediction 1 True_Y 1 Predicted_Iris Iris-setosa
Prediction 1 True_Y 1 Predicted_Iris Iris-setosa
Prediction 1 True_Y 1 Predicted_Iris Iris-setosa
Prediction 1 True_Y 1 Predicted_Iris Iris-setosa
Prediction 1 True_Y 1 Predicted_Iris Iris-setosa
Prediction 1 True_Y 1 Predicted_Iris Iris-setosa
Prediction 1 True_Y 1 Predicted_Iris Iris-setosa
Prediction 1 True_Y 1 Predicted_Iris Iris-setosa
Prediction 1 True_Y 1 Predicted_Iris Iris-setosa
Prediction 1 True_Y 1 Predicted_Iris Iris-setosa
Prediction 1 True_Y 1 Predicted_Iris Iris-setosa
Prediction 1 True_Y 1 Predicted_Iris Iris-setosa
Prediction 1 True_Y 1 Predicted_Iris Iris-setosa
Prediction 1 True_Y 1 Predicted_Iris Iris-setosa
Prediction 1 True_Y 1 Predicted_Iris Iris-setosa
Prediction 1 True_Y 1 Predicted_Iris Iris-setosa
Prediction 1 True_Y 1 Predicted_Iris Iris-setosa
Prediction 1 True_Y 1 Predicted_Iris Iris-setosa
Prediction 1 True_Y 1 Predicted_Iris Iris-setosa
Prediction 1 True_Y 1 Predicted_Iris Iris-setosa
Prediction 1 True_Y 1 Predicted_Iris Iris-setosa
Prediction 1 True_Y 1 Predicted_Iris Iris-setosa
Prediction 1 True_Y 1 Predicted_Iris Iris-setosa
Prediction 1 True_Y 1 Predicted_Iris Iris-setosa
Prediction 2 True_Y 2 Predicted_Iris Iris-versicolor
Prediction 2 True_Y 2 Predicted_Iris Iris-versicolor
Prediction 2 True_Y 2 Predicted_Iris Iris-versicolor
Prediction 2 True_Y 2 Predicted_Iris Iris-versicolor
Prediction 2 True_Y 2 Predicted_Iris Iris-versicolor
Prediction 2 True_Y 2 Predicted_Iris Iris-versicolor
Prediction 2 True_Y 2 Predicted_Iris Iris-versicolor
Prediction 2 True_Y 2 Predicted_Iris Iris-versicolor
Prediction 2 True_Y 2 Predicted_Iris Iris-versicolor
Prediction 2 True_Y 2 Predicted_Iris Iris-versicolor
Prediction 2 True_Y 2 Predicted_Iris Iris-versicolor
Prediction 2 True_Y 2 Predicted_Iris Iris-versicolor
Prediction 2 True_Y 2 Predicted_Iris Iris-versicolor
Prediction 2 True_Y 2 Predicted_Iris Iris-versicolor
Prediction 2 True_Y 2 Predicted_Iris Iris-versicolor
Prediction 2 True_Y 2 Predicted_Iris Iris-versicolor
Prediction 0 True_Y 2 Predicted_Iris Iris-virginica
Prediction 2 True_Y 2 Predicted_Iris Iris-versicolor
Prediction 2 True_Y 2 Predicted_Iris Iris-versicolor
Prediction 2 True_Y 2 Predicted_Iris Iris-versicolor
Prediction 2 True_Y 2 Predicted_Iris Iris-versicolor
Prediction 2 True_Y 2 Predicted_Iris Iris-versicolor
Prediction 2 True_Y 2 Predicted_Iris Iris-versicolor
Prediction 2 True_Y 2 Predicted_Iris Iris-versicolor
Prediction 2 True_Y 2 Predicted_Iris Iris-versicolor
Prediction 0 True_Y 0 Predicted_Iris Iris-virginica
Prediction 0 True_Y 0 Predicted_Iris Iris-virginica
Prediction 0 True_Y 0 Predicted_Iris Iris-virginica
Prediction 0 True_Y 0 Predicted_Iris Iris-virginica
Prediction 0 True_Y 0 Predicted_Iris Iris-virginica
Prediction 0 True_Y 0 Predicted_Iris Iris-virginica
Prediction 0 True_Y 0 Predicted_Iris Iris-virginica
Prediction 0 True_Y 0 Predicted_Iris Iris-virginica
Prediction 0 True_Y 0 Predicted_Iris Iris-virginica
Prediction 0 True_Y 0 Predicted_Iris Iris-virginica
Prediction 0 True_Y 0 Predicted_Iris Iris-virginica
Prediction 0 True_Y 0 Predicted_Iris Iris-virginica
Prediction 0 True_Y 0 Predicted_Iris Iris-virginica
Prediction 0 True_Y 0 Predicted_Iris Iris-virginica
Prediction 2 True_Y 0 Predicted_Iris Iris-versicolor
Prediction 2 True_Y 0 Predicted_Iris Iris-versicolor
Prediction 2 True_Y 0 Predicted_Iris Iris-versicolor
Prediction 0 True_Y 0 Predicted_Iris Iris-virginica
Prediction 0 True_Y 0 Predicted_Iris Iris-virginica
Prediction 0 True_Y 0 Predicted_Iris Iris-virginica
Prediction 0 True_Y 0 Predicted_Iris Iris-virginica
Prediction 0 True_Y 0 Predicted_Iris Iris-virginica
Prediction 0 True_Y 0 Predicted_Iris Iris-virginica
Prediction 0 True_Y 0 Predicted_Iris Iris-virginica
Prediction 0 True_Y 0 Predicted_Iris Iris-virginica
0.946667
"""
