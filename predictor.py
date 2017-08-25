import pandas as p
import numpy as np
import tensorflow as tf


def error_rate(p, t):
    return np.mean(p != t)


# import the data
df = p.read_csv("data.csv")

training = df[:500]
test = df[500:]

# target
ytrain = [[1, 0] if d == 'M' else [0, 1] for d in training["diagnosis"]]
ytest = [1 if d == 'M' else 0 for d in test["diagnosis"]]

# input features
xtrain = training[training.columns[12:22]]
xtest = test[test.columns[12:22]]

N, D = xtrain.shape

# add hidden layer
M = 30
K = 2

# initialise weights
W1_init = np.random.randn(D, M)
b1_init = np.zeros(M)
W2_init = np.random.randn(M, K) / np.sqrt(M)
b2_init = np.zeros(K)

# set-up TensorFlow parameters
X = tf.placeholder(tf.float32, shape=(None, D), name='X')
T = tf.placeholder(tf.float32, shape=(None, K), name='T')
W1 = tf.Variable(W1_init, dtype=tf.float32)
b1 = tf.Variable(b1_init, dtype=tf.float32)
W2 = tf.Variable(W2_init, dtype=tf.float32)
b2 = tf.Variable(b2_init, dtype=tf.float32)

# define the model (graph)
Z = tf.nn.relu(tf.matmul(X, W1) + b1)
Yish = tf.nn.relu(tf.matmul(Z, W2) + b2)

# cost
cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=Yish, labels=T))

# cost optimiser
train_op = tf.train.RMSPropOptimizer(learning_rate=0.0000004, decay=0.99, momentum=0.9).minimize(cost)

# prediction
predict_op = tf.argmax(Yish, axis=1)

costs = []
init = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(init)
    session.run(train_op, feed_dict={X: xtrain, T: ytrain})
    prediction = session.run(predict_op, feed_dict={X: xtest})
    error = error_rate(prediction, ytest)

print(error)
