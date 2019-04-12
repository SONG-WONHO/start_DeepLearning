"""
RNN cell : BasicRNNcell / LSTM / GRU
RNN cell`s input : [batch_size, sequence_length, input_dimension]
RNN cell`s output : [batch_size, sequence_length, hidden_dimension]
"""
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.contrib import layers
from tensorflow.contrib import seq2seq

#sample string
samples = "hello! python, hi! tensorflow"

#index to character
idx2char = list(set(samples))

#character to index
char2idx = {c:i for i,c in enumerate(idx2char)}

#sample string to index
samples_idx = [char2idx[c] for c in samples]

#x_data : "hello! python, hi! tensorflo"
#y_data : "ello! python, hi! tensorflow"
x_data = [samples_idx[:-1]]
y_data = [samples_idx[1:]]

#constants
hidden_dim = 10
input_dim = len(idx2char)
output_dim = len(idx2char)
sequence_length = len(samples) - 1
batch_size = 1
learning_rate = 0.01

X = tf.placeholder(tf.int32, [None, sequence_length])
Y = tf.placeholder(tf.int32, [None, sequence_length])
X_one_hot = tf.one_hot(X, input_dim)
Y_one_hot = tf.one_hot(Y, output_dim)

#RNN cell
cell = rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, X_one_hot, dtype=tf.float32, initial_state=initial_state)

#rnn outputs => num classes, no activation function!
logits = layers.fully_connected(outputs, output_dim, None)

#cost function
#cost = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot)
#cost = tf.reduce_mean(cost)
#=============================================same===============================
weights = tf.ones([batch_size, sequence_length], tf.float32)
cost = seq2seq.sequence_loss(logits, Y, weights)
cost =tf.reduce_mean(cost)

train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

prediction = tf.argmax(logits, -1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(20001):
        cost_val, prediction_val, _ = sess.run([cost, prediction, train], feed_dict={X:x_data, Y:y_data})

        result_list = [idx2char[c] for c in np.squeeze(prediction_val)]

        if step % 2000 == 0:
            print("[{}] cost {} prediction {}".format(step, cost_val, result_list))

        if step == 20000:
            print("True string : {}\n".format(samples[1:])+"Prediction string : "+"".join(result_list))


"""
[0] cost 2.796222686767578 prediction ['f', ',', ',', ',', '!', ',', ',', '!', '!', 't', 't', 't', 'f', ',', ',', 'l', 'f', 'l', 'l', 't', ',', ',', 'f', 'f', 'f', 'f', ',', 'f']
[2000] cost 2.585203170776367 prediction ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o']
[4000] cost 2.2530386447906494 prediction ['l', 'l', 'l', 'o', 'o', ' ', ' ', ' ', ' ', 'h', 'h', 'h', ' ', ' ', 'h', 'h', ' ', ' ', 't', 't', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o']
=========================================================================================================================================================================================
[16000] cost 0.3224013149738312 prediction ['e', 'l', 'l', 'o', '!', ' ', 'p', 'y', 't', 'h', 'o', 'n', ',', ' ', 'h', 'i', '!', ' ', 't', 'e', 'n', 's', 'o', 'r', 'f', 'l', 'o', 'w']
[18000] cost 0.24367673695087433 prediction ['e', 'l', 'l', 'o', '!', ' ', 'p', 'y', 't', 'h', 'o', 'n', ',', ' ', 'h', 'i', '!', ' ', 't', 'e', 'n', 's', 'o', 'r', 'f', 'l', 'o', 'w']
[20000] cost 0.18942570686340332 prediction ['e', 'l', 'l', 'o', '!', ' ', 'p', 'y', 't', 'h', 'o', 'n', ',', ' ', 'h', 'i', '!', ' ', 't', 'e', 'n', 's', 'o', 'r', 'f', 'l', 'o', 'w']

True string : ello! python, hi! tensorflow
Prediction string : ello! python, hi! tensorflow
"""