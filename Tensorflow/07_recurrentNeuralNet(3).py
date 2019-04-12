import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.contrib import rnn
from tensorflow.contrib import layers

#min-max scaling function
def minMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator + 1e-7)

#load data
samples = np.loadtxt("./samples/stocks.csv", delimiter=',', dtype=np.float32)

#convert to time-period
samples = samples[::-1]

#constants
learning_rate = 0.001
input_dim = 5
output_dim = 1
hidden_dim = 10
sequence_length = 7
batch_size = len(samples) - sequence_length
training_size = int(batch_size*0.7)

#normalization scaling
#for i in range(input_dim):
#  samples[:,i] = (samples[:,i]-samples[:,i].mean())/samples[:,i].std()

# min-max scaling
samples = minMaxScaler(samples)

x = samples
y = samples[:,[-1]]

x_data = []
y_data = []

#data slicing
for i in range(batch_size):
    x_num = x[i:i+sequence_length]
    y_num = y[i+sequence_length]

    x_data.append(x_num)
    y_data.append(y_num)

#train data
x_train = x_data[:training_size]
y_train = y_data[:training_size]

#test data
x_test = x_data[training_size:]
y_test = y_data[training_size:]

X = tf.placeholder(tf.float32, [None, sequence_length, input_dim])
Y = tf.placeholder(tf.float32, [None, 1])

#for stacked LSTM
def lstm_cell():
    cell = rnn.BasicLSTMCell(hidden_dim, state_is_tuple=True, activation=tf.tanh)
    return cell

#stacked LSTM
cell = rnn.MultiRNNCell([lstm_cell() for _ in range(2)], state_is_tuple=True)

outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

#We need last item
logits = layers.fully_connected(outputs[:,-1], output_dim, None)

cost = tf.reduce_mean(tf.square(logits-Y))
train = tf.train.AdamOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2000):
        cost_val, _  = sess.run([cost, train], feed_dict={X:x_train, Y:y_train})

        if step % 200 == 0:
            print("[step {}] cost {}".format(step, cost_val))

    test_result = sess.run(logits, feed_dict={X:x_test})

    #graph
    plt.plot(y_test)
    plt.plot(test_result)
    plt.xlabel("Time period")
    plt.ylabel("Stock price")
    plt.show()

"""
<case: Normalizaiton>
[step 0] cost 0.6887598037719727
[step 200] cost 0.03039746731519699
[step 400] cost 0.022703634575009346
======================================
[step 1400] cost 0.010026202537119389
[step 1600] cost 0.008881459012627602
[step 1800] cost 0.007830946706235409

<case: MinMax>
[step 0] cost 0.09958923608064651
[step 200] cost 0.0026880204677581787
[step 400] cost 0.002537171356379986
======================================
[step 1400] cost 0.0009560093167237937
[step 1600] cost 0.0008838321082293987
[step 1800] cost 0.0008473573252558708
"""