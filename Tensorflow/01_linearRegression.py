import tensorflow as tf

#data
x_data = [1,2,3,4,5,6,7]
y_data = [4,5,6,7,8,9,10]

#constants
learning_rate = 0.01

X = tf.placeholder(tf.float32, [None])
Y = tf.placeholder(tf.float32, [None])

W = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

#linear hypothesis
hypothesis = X*W + b

#cost function, x_value: weight, y_value: cost
cost = tf.reduce_mean(tf.square(hypothesis - Y), -1, name="cost")
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2000):
        cost_val, hypothesis_val, _ = sess.run([cost, hypothesis, train], feed_dict={X:x_data, Y:y_data})

        if step % 20 == 0:
            print("[{}] cost {} hypothesis {}".format(step, cost_val, hypothesis_val))

"""
[0] cost 169.62374877929688 hypothesis [-2.31489372 -3.33670878 -4.35852385 -5.38033915 -6.40215445 -7.42396927 -8.44578457]
[20] cost 2.3894200325012207 hypothesis [  1.2419914    2.93980169   4.63761234   6.33542252   8.03323269   9.73104286  11.42885303]
[40] cost 2.0482091903686523 hypothesis [  1.44656205   3.09268451   4.73880672   6.38492918   8.03105164   9.67717457  11.3232975 ]
=============================================================================================================================
[1940] cost 9.002223464449344e-07 hypothesis [  3.99830723   4.99873543   5.9991641    6.9995923    8.00002098   9.00044918  10.00087738]
[1960] cost 7.718243182353035e-07 hypothesis [  3.99843264   4.99882936   5.99922562   6.99962234   8.00001907   9.0004158   10.00081253]
[1980] cost 6.614169478780241e-07 hypothesis [  3.99854898   4.99891615   5.99928379   6.99965096   8.00001812   9.00038528  10.00075245]
"""