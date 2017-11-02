import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np


from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn

mnist = input_data.read_data_sets(
    'MNIST_data', one_hot=True)  # call mnist function

learningRate = 1e-3
trainingIters = 120000
batchSize = 128
displayStep = 10

nInput = 28  # we want the input to take the 28 pixels
nSteps = 28  # every 28
nHidden = 256  # number of neurons for the RNN
nClasses = 10  # this is MNIST so you know

x = tf.placeholder('float', [None, nSteps, nInput])
y = tf.placeholder('float', [None, nClasses])

weights = {
    'out': tf.Variable(tf.random_normal([nHidden, nClasses]))
}

biases = {
    'out': tf.Variable(tf.random_normal([nClasses]))
}


def RNN(x, weights, biases, use_gru=False):
    # configuring so you can get it as needed for the 28 pixels
    x = tf.unstack(x, nSteps, 1)

    # find which lstm to use in the documentation
    lstmCell = rnn.BasicLSTMCell(
        nHidden) if not use_gru else rnn.GRUCell(nHidden)

    # for the rnn where to get the output and hidden state
    outputs, states = tf.nn.static_rnn(lstmCell, x, dtype=tf.float32)

    return tf.matmul(outputs[-1], weights['out']) + biases['out']


pred = RNN(x, weights, biases, use_gru=True)

# optimization
# create the cost, optimization, evaluation, and accuracy
# for the cost softmax_cross_entropy_with_logits seems really good
cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
optimizer = tf.train.AdamOptimizer(
    learning_rate=learningRate).minimize(cost)

correctPred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 1

    while step * batchSize < trainingIters:
        # mnist has a way to get the next batch
        batchX, batchY = mnist.train.next_batch(batchSize)
        batchX = batchX.reshape((batchSize, nSteps, nInput))

        sess.run(optimizer, feed_dict={
            x: batchX, y: batchY})

        if step % displayStep == 0:
            acc = accuracy.eval(feed_dict={
                x: batchX, y: batchY})
            loss = cost.eval(feed_dict={
                x: batchX, y: batchY})
            print("Iter " + str(step * batchSize) + ", Minibatch Loss= " +
                  "{:.6f}".format(loss) + ", Training Accuracy= " +
                  "{:.5f}".format(acc))
        step += 1
    print('Optimization finished')

    testData = mnist.test.images.reshape((-1, nSteps, nInput))
    testLabel = mnist.test.labels
    print("Testing Accuracy:",
          sess.run(accuracy, feed_dict={x: testData, y: testLabel}))
