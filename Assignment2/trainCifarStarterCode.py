import os
import math
from scipy import misc
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import matplotlib as mp

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x): return x
    print("Install tqdm for cool progress bars")

cwd = os.getcwd()

# --------------------------------------------------
# setup


def weight_variable(shape):
    '''
    Initialize weights
    :param shape: shape of weights, e.g. [w, h ,Cin, Cout] where
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters
    Cout: the number of filters
    :return: a tensor variable for weights with initial values
    '''

    # IMPLEMENT YOUR WEIGHT_VARIABLE HERE
    W = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    return W


def bias_variable(shape):
    '''
    Initialize biases
    :param shape: shape of biases, e.g. [Cout] where
    Cout: the number of filters
    :return: a tensor variable for biases with initial values
    '''

    # IMPLEMENT YOUR BIAS_VARIABLE HERE
    b = tf.Variable(tf.constant(0.0, shape=shape))
    return b


def conv2d(x, W):
    '''
    Perform 2-D convolution
    :param x: input tensor of size [N, W, H, Cin] where
    N: the number of images
    W: width of images
    H: height of images
    Cin: the number of channels of images
    :param W: weight tensor [w, h, Cin, Cout]
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters = the number of channels of images
    Cout: the number of filters
    :return: a tensor of features extracted by the filters, a.k.a. the results after convolution
    '''

    # IMPLEMENT YOUR CONV2D HERE
    h_conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    return h_conv


def max_pool_2x2(x):
    '''
    Perform non-overlapping 2-D maxpooling on 2x2 regions in the input data
    :param x: input data
    :return: the results of maxpooling (max-marginalized + downsampling)
    '''

    # IMPLEMENT YOUR MAX_POOL_2X2 HERE
    h_max = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME')
    return h_max


def variable_summary(name, tensor):
    with tf.name_scope(name + "-summary"):
        # Summarize the basic scalar stats
        mean, variance = tf.nn.moments(
            tensor, axes=list(range(tf.rank(tensor).eval())))
        tf.summary.scalar('mean', mean)
        tf.summary.scalar('std', tf.sqrt(variance))
        tf.summary.scalar('max', tf.reduce_max(tensor))
        tf.summary.scalar('min', tf.reduce_min(tensor))
        # Create a histogram of the tensor
        tf.summary.histogram('histogram', tensor)


def plot_filter(filters, name):
    n_filters = filters.shape[3]
    plt.figure(1, figsize=(20, 20))
    n_columns = 6
    n_rows = math.ceil(n_filters / n_columns) + 1
    plt.title(name)
    for i in range(n_filters):
        plt.subplot(n_rows, n_columns, i + 1)
        plt.title('Filter ' + str(i))
        plt.imshow(filters[0, :, :, i], interpolation="nearest", cmap="gray")


def get_filters(layer, inp_image, name):
    filters = sess.run(layer, feed_dict={
        x: inp_image[np.newaxis, ...], keep_prob: 1.0, conv_keep_prob: 1.0})
    plot_filter(filters, name)


class MaxCheckpointer(object):

    def __init__(self, save_to, sess):
        self.cur_max = -float('inf')
        # Create a saver for writing training checkpoints.
        self.saver = tf.train.Saver()
        self.save_to = save_to

    def __call__(self, new_val):
        if new_val >= self.cur_max:
            print("Got new best:", new_val, "- saving model.")
            self.saver.save(sess, os.path.join(cwd, self.save_to))
            self.cur_max = new_val


ntrain = 1000  # per class
ntest = 100  # per class
nclass = 10  # number of classes
imsize = 28
nchannels = 1
batchsize = 128
nepochs = 5

Train = np.zeros((ntrain * nclass, imsize, imsize, nchannels))
Test = np.zeros((ntest * nclass, imsize, imsize, nchannels))
LTrain = np.zeros((ntrain * nclass, nclass))
LTest = np.zeros((ntest * nclass, nclass))

itrain = -1
itest = -1
for iclass in range(0, nclass):
    for isample in range(0, ntrain):
        path = os.path.join(
            cwd, 'CIFAR10/Train/%d/Image%05d.png' % (iclass, isample))
        im = misc.imread(path)  # 28 by 28
        im = im.astype(float) / 255
        itrain += 1
        Train[itrain, :, :, 0] = im
        LTrain[itrain, iclass] = 1  # 1-hot lable
    for isample in range(0, ntest):
        path = os.path.join(
            cwd, 'CIFAR10/Test/%d/Image%05d.png' % (iclass, isample))
        im = misc.imread(path)  # 28 by 28
        im = im.astype(float) / 255
        itest += 1
        Test[itest, :, :, 0] = im
        LTest[itest, iclass] = 1  # 1-hot lable

sess = tf.InteractiveSession()

# tf variable for the data, remember shape is [None, width, height, numberOfChannels]
tf_data = tf.placeholder(tf.float32, shape=[None, imsize, imsize, nchannels])
# tf variable for labels
tf_labels = tf.placeholder(tf.float32, shape=[None, nclass])

# --------------------------------------------------
# model
# create your model
# First convolutional layer
conv_keep_prob = tf.placeholder(tf.float32)

x = tf_data
# x = tf.nn.dropout(x, conv_keep_prob)

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Variable summaries
variable_summary("W_conv1", W_conv1)
variable_summary("b_conv1", b_conv1)
variable_summary("h_conv1", h_conv1)
variable_summary("h_pool1", h_pool1)

h_pool1_drop = tf.nn.dropout(h_pool1, conv_keep_prob)

# Second convolutional layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1_drop, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Variable summaries
variable_summary("W_conv2", W_conv2)
variable_summary("b_conv2", b_conv2)
variable_summary("h_conv2", h_conv2)
variable_summary("h_pool2", h_pool2)

h_pool2_drop = tf.nn.dropout(h_pool2, conv_keep_prob)

# W_conv3 = weight_variable([5, 5, 64, 128])
# b_conv3 = bias_variable([128])
# h_conv3 = tf.nn.relu(conv2d(h_pool2_drop, W_conv3) + b_conv3)
# h_pool3 = max_pool_2x2(h_conv3)

conv_out = h_pool2_drop

# Densely connected layer
conv_out_shape = conv_out.get_shape().as_list()
flat_shape = conv_out_shape[1] * conv_out_shape[2] * conv_out_shape[3]
flat_conv = tf.reshape(
    conv_out, [-1, flat_shape])

# dropout
keep_prob = tf.placeholder(tf.float32)
flat_conv_drop = tf.nn.dropout(flat_conv, keep_prob)

# softmax
W_fc2 = weight_variable([flat_shape, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(flat_conv_drop, W_fc2) + b_fc2

# Variable summaries
variable_summary("W_fc2", W_fc2)
variable_summary("b_fc2", b_fc2)
variable_summary("y_conv", y_conv)

# --------------------------------------------------
# loss
# set up the loss, optimization, evaluation, and accuracy
# setup training
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=tf_labels, logits=y_conv))
opt = tf.train.AdamOptimizer(1e-3)
optimizer = opt.minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(tf_labels, 1))
tr_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
te_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.scalar("Loss", cross_entropy)
tf.summary.scalar("Train Accuracy", tr_accuracy)
tf.summary.scalar("Test Accuracy", te_accuracy)


# --------------------------------------------------
# optimization
sess.run(tf.global_variables_initializer())
# Set up the summary writer
result_dir = os.path.join(cwd, "results")
summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(result_dir, sess.graph)
# Create model checkpointer
checkpointer = MaxCheckpointer("cifar10-best", sess)

# setup as [batchsize, width, height, numberOfChannels] and use np.zeros()
batch_xs = np.empty((batchsize, imsize, imsize, nchannels))
# setup as [batchsize, the how many classes]
batch_ys = np.zeros((batchsize, nclass))
nsamples = ntrain * nclass
# batch indicies
perm = np.arange(nsamples)
# try a small iteration size once it works then continue
for i in range(nepochs):
    print("Epoch {}/{}".format(i + 1, nepochs))
    train_ep_acc_sum = 0
    # Shuffle the indicies
    np.random.shuffle(perm)
    # Initialize a batch generator
    batchgen = ((Train[perm[j:j + batchsize]], LTrain[perm[j:j + batchsize]])
                for j in tqdm(range(0, nsamples, batchsize)))
    # Create an augmenter for the images
    # imagegen = create_augmenter(batchgen)
    for batch_xs, batch_ys in batchgen:
        # output the training accuracy every 100 iterations
        train_accuracy = tr_accuracy.eval(feed_dict={
            tf_data: batch_xs, tf_labels: batch_ys, keep_prob: 1.0, conv_keep_prob: 1.0})
        train_ep_acc_sum += (train_accuracy * len(batch_xs))
        # Collect the summary statistics
        summary_str = sess.run(summary_op, feed_dict={
                               tf_data: batch_xs, tf_labels: batch_ys, keep_prob: 1.0, conv_keep_prob: 1.0})
        summary_writer.add_summary(summary_str, i)
        summary_writer.flush()
        # dropout only during training
        optimizer.run(feed_dict={
            tf_data: batch_xs, tf_labels: batch_ys, keep_prob: 0.5, conv_keep_prob: 0.75})
    print("Train Accuracy: {}".format(train_ep_acc_sum / nsamples))
    test_acc = te_accuracy.eval(
        feed_dict={tf_data: Test, tf_labels: LTest, keep_prob: 1.0, conv_keep_prob: 1.0})
    print("Test accuracy %g" % test_acc)
    checkpointer(test_acc)

# Select 32 images to visualize through the convnet
for i in np.random.permutation(ntest)[:32]:
    get_filters(h_pool1, Test[i], "Conv1")
    get_filters(h_pool2, Test[i], "Conv2")


sess.close()
