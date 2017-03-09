import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./data/")

X_train = mnist.train.images
y_train = mnist.train.labels.astype("int")


X_test  = mnist.test.images
y_test = mnist.test.labels.astype("int")


# Construction Phase

tf.reset_default_graph()

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10
learning_rate = 0.01

# We can use placeholders to represent the training data and targets.
# In the case of X, we know it will be a matrix (i.e. a 2-d tensor) with
# instances along the first dimension and features along the second dimension.
# We know that the number of features is going to be: 28 * 28 = 784
# Since we don't know the number of instances, the shape of X is (None, 784).
#
# Similarly, we know that y will be a 1-d tensor, or vector, with one entry
# per instance. Just like in the case of X, we do not know how many instances
# we will have. Therefore, the shape of y is (None).
#
# The placeholder X will act as the input. During the execution
# phase it will hold one training batch at a time.
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
        init   = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W      = tf.Variable(init, name="weights")

        # Create a vector of bias units for each neuron.
        b = tf.Variable(tf.zeros([n_neurons]), name="biases")

        # A vectorized implementation of computing a weighted sum of the
        # inputs plus the bias term.
        z = tf.matmul(X, W) + b

        # If we want to use relu activation function, then use tensorflow's
        # implementation. Otherwise, return z.
        if activation == "relu":
            return tf.nn.relu(z)
        else:
            return z

with tf.name_scope("dnn"):
    hidden1 = neuron_layer(X, n_hidden1, "hidden1", activation="relu")
    hidden2 = neuron_layer(hidden1, n_hidden2, "hidden1", activation="relu")
    logits  = neuron_layer(hidden2, n_outputs, "output")

with tf.name_scope("loss"):
    entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(entropy, name="loss")

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
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
