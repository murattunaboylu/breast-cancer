import pandas as p
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def error_rate(p, t):
    return np.mean(p != t)


def y2indicator(y):
    n = len(y)
    y = y.astype(np.int32)
    ind = np.zeros((n, 2))
    for i in range(n):
        ind[i, y[i]] = 1
    return ind


def get_normalized_data():
    print("Reading in and transforming data...")

    df = p.read_csv("data.csv")

    # binarise diagnosis
    df.loc[df['diagnosis'] == 'M', 'diagnosis'] = 1
    df.loc[df['diagnosis'] == 'B', 'diagnosis'] = 0

    data = df.as_matrix().astype(np.float32)
    np.random.shuffle(data)
    X = data[:, 12:22]
    # mu = X.mean(axis=0)
    # std = X.std(axis=0)
    # np.place(std, std == 0, 1)
    # X = (X - mu) / std # normalize the data
    Y = data[:, 1]
    return X, Y

# import the data
X, Y = get_normalized_data()

# input features
xtrain = X[:500, ]
xtest = X[500:, ]

# target
ytrain = Y[:500]
ytest = Y[500:]

ytrain_ind = y2indicator(ytrain)
ytest_ind = y2indicator(ytest)

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
train_op = tf.train.RMSPropOptimizer(learning_rate=0.0004, decay=0.99, momentum=0.9).minimize(cost)

# prediction
predict_op = tf.argmax(Yish, axis=1)

costs = []
max_iter = 1500
init = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(init)

    for i in range(max_iter):
        session.run(train_op, feed_dict={X: xtrain, T: ytrain_ind})

        test_cost = session.run(cost, feed_dict={X: xtest, T: ytest_ind})
        prediction = session.run(predict_op, feed_dict={X: xtest})
        err = error_rate(prediction, ytest)
        print("Cost / err at iteration i=%d, %.3f / %.3f" % (i, test_cost, err))
        costs.append(test_cost)


plt.plot(prediction, marker='o', linestyle='None')
plt.plot(ytest, marker='x', linestyle='None', markerfacecolor='#333399')
plt.show()

plt.plot(costs)
plt.show()

