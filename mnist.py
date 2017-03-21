import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./data/")

X_train = mnist.train.images
y_train = mnist.train.labels.astype("int")


X_test  = mnist.test.images
y_test = mnist.test.labels.astype("int")


# CONSTRUCTION PHASE

tf.reset_default_graph()

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10
learning_rate = 0.01

# Placeholder operations hold information about the type and shape of the
# tensor they represent. We can use placeholders to represent the training
# data and target labels.
#
# We know X will be a 2-d tensor, or matrix, with training examples
# along the first dimension and features along the second dimension.
# We also know that the number of features is going to be: 28 * 28 = 784 and
# will be represented as 32-bit floating point values.
#
# We do not need to specify the size of the first dimension in the tensor because
# that depends on the batch size we use. If we hard code that value, we'd have to
# change the shape of our placeholder every time. Fortunately, TF has all the
# information it needs. We can feed the placeholder any number of training
# examples so long as they conform to the standards we've specified. That is,
# it has to be a 32-bit floating point value and it must have 784 input values.
#
# The same logic applies to the y placeholder, although the implementation is
# different. We want y to be a one-dimensional tensor, or vector. Therefore,
# we need to specify its shape as None.
#
# What then happens during the execution phase is that these placeholders are
# replaced by one training batch at a time.
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")

def neuron_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name):
        # Indexing into position 1 will return the number of features
        # whereas position 0 would tell us the number of instances.
        n_inputs = int(X.get_shape()[1])

        # Construct a matrix which contains all the connection
        # weights between each input feature and each activation neuron.
        # It's shape will be (n_inputs, n_neurons). The weights will be
        # initialized randomly using a normal Gaussian distribution with
        # a standard deviation of 2 / sqrt(n_inputs). This standard deviation
        # is a techniality that helps neural networks converge much faster.
        stddev = 2 / np.sqrt(n_inputs)
        W_init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W      = tf.Variable(W_init, name="weights")

        # Create a vector of bias units for each neuron.
        b = tf.Variable(tf.zeros([n_neurons]), name="biases")

        # A vectorized implementation of computing a weighted sum of the
        # inputs, weights, and the bias term.
        z = tf.matmul(X, W) + b

        # If we want to use relu activation function, then use tensorflow's
        # implementation. Otherwise, return the weighted sum.
        if activation == "relu":
            return tf.nn.relu(z) # max(0, z)
        else:
            return z

with tf.name_scope("dnn"):
    hidden1 = neuron_layer(X, n_hidden1, "hidden1", activation="relu")
    hidden2 = neuron_layer(hidden1, n_hidden2, "hidden2", activation="relu")
    hidden3 = neuron_layer(hidden2, 75, "hidden3", activation="relu")

    logits  = neuron_layer(hidden3, n_outputs, "output")

with tf.name_scope("loss"):
    # Take the logits, feed them into the softmax function, then compute
    # the cross entropy.
    #
    # tf.nn.sparse_softmax_cross_entropy_with_logits() expects labels in the form of
    # integers ranging from 0 to the number of classes minus 1. This will
    # return a 1D tensor containing the cross entropy for each instance.
    entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)

    # We can then sum all the entropy values and divide to find the mean.
    loss = tf.reduce_mean(entropy, name="loss")

with tf.name_scope("train"):
    optimizer   = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    # For each instance determine if the highest logit corresponds to the
    # target class. Returns a 1D tensor of boolean values.
    correct = tf.nn.in_top_k(logits, y, 1)

    # Cast the boolean values to integers, find the sum, then divide to
    # compute the mean value and get our accuracy.
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Execution Phase

n_epochs   = 20
batch_size = 50
iterations = mnist.train.num_examples // batch_size

with tf.Session() as sess:
    print("Running TensorFlow session...")
    print("# Training examples:", mnist.train.num_examples)
    print("# Test examples:", mnist.test.num_examples)

    init.run()

    for epoch in range(n_epochs):
        for iteration in range(iterations):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test  = accuracy.eval(feed_dict={X: X_test, y: y_test})
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)

    save_path = saver.save(sess, "model.ckpt")
