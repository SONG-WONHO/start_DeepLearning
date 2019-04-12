import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import rnn

#samples, from WIKIPEDIA
samples  = "TensorFlow is an open-source software library " \
           "for machine learning across a range of tasks, " \
           "and developed by Google to meet their needs " \
           "for systems capable of building and training neural networks " \
           "to detect and decipher patterns and correlations, " \
           "analogous to the learning and reasoning which humans use."

#index to character
idx2char = list(set(samples))

#character to index
char2idx = {c:i for i,c in enumerate(idx2char)}

#constants
sequence_length = 10
hidden_dim = 10
input_dim = len(idx2char)
output_dim = len(idx2char)
batch_size = len(samples) - sequence_length
learning_rate = 0.01

x_data = []
y_data = []

#data slicing
for i in range(0, batch_size):

    #0~5/1~6/2~7/3~8 ...
    x_str = samples[i:i+sequence_length]

    #1~6/2~7/3~8/4~9 ...
    y_str = samples[i+1:i+sequence_length+1]

    #character to index
    x_str = [char2idx[c] for c in x_str]
    y_str = [char2idx[c] for c in y_str]

    #x_data
    x_data.append(x_str)

    #y_data
    y_data.append(y_str)

X = tf.placeholder(tf.int32, [None, sequence_length])
Y = tf.placeholder(tf.int32, [None, sequence_length])

X_one_hot = tf.one_hot(X, input_dim, axis= -1)
Y_one_hot = tf.one_hot(Y, output_dim, axis= -1)

#for stacked LSTM
def lstm_cell():
    cell = rnn.BasicLSTMCell(hidden_dim, state_is_tuple= True)
    return cell

#stacked LSTM
cell = rnn.MultiRNNCell([lstm_cell() for _ in range(2)], state_is_tuple= True)

outputs, _states = tf.nn.dynamic_rnn(cell, X_one_hot, dtype=tf.float32)
#outputs = tf.reshape(outputs, [-1,hidden_dim])

logits = layers.fully_connected(outputs, output_dim, None)
#logits = tf.reshape(logits, [batch_size, sequence_length, output_dim])

cost = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot)
cost = tf.reduce_mean(cost)

#We have to use AdamOptimizer
train = tf.train.AdamOptimizer(learning_rate).minimize(cost)

prediction = tf.argmax(logits, -1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(3000):
        cost_val, prediction_val, _ = sess.run([cost, prediction, train], feed_dict={X:x_data,Y:y_data})

        for j, c in enumerate(prediction_val):
            print(i, j, "".join([idx2char[t] for t in c]), cost_val)

    #print result string!
    for i,c in enumerate(prediction_val):

        if i == 0:  # print all for the first result to make a sentence
            print(''.join([idx2char[t] for t in c]), end="")
        else:
            print(idx2char[c[-1]], end="")

"""
0 0 Gaaaaaaaaa 3.36707
0 1 rrrraaaTaT 3.36707
0 2 yTTTTTTTTT 3.36707
0 3 TTTTTTTTTT 3.36707

0 289 nnnnnnnnnn 3.36707
0 290 nnnusnnnTT 3.36707
0 291 sssssrrTTT 3.36707
0 292 uusnnTTTTT 3.36707
===========================
2999 0 ensorFlow  0.287551
2999 1  sorFlow i 0.287551
2999 2 g,rFlow is 0.287551
2999 3  rFlow is  0.287551

2999 290 h humans u 0.287551
2999 291 ehumans us 0.287551
2999 292 aumans use 0.287551
2999 293 emans use. 0.287551

ensorFlow is an open-source software library for machine learning anross a range of tasks, 
and developed by Google to meet their needs for systems capable of building and training neural networks to detect 
and decipher patterns and correlations, analogous to the learning and reasoning which humans use. 
"""