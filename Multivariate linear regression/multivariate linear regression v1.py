
import tensorflow as tf

# You will NOT need to do this if you're using tensorflow v1
tf.compat.v1.disable_eager_execution()

# Training set
train_x1 = [400, 430, 405, 180, 220, 405, 500, 1500, 1500, 1000]
train_x2 = [180, 140, 255, 180, 100, 255, 350, 900, 900, 1000]
train_x3 = [200, 305, 300, 180, 160, 300, 440, 200, 900, 1000]

train_y = [4.20, 4.85, 6, 3.50, 2.70, 6.50, 11, 20.50, 39.80, 35.30]

# Place holders
x1 = tf.compat.v1.placeholder(tf.float32)
x2 = tf.compat.v1.placeholder(tf.float32)
x3 = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

# Set values for our thetas
t1 = tf.Variable(tf.compat.v1.random_normal([1]))
t2 = tf.Variable(tf.compat.v1.random_normal([1]))
t3 = tf.Variable(tf.compat.v1.random_normal([1]))
t0 = tf.Variable(tf.compat.v1.random_normal([1]))

# Create our hypothesis
h = train_x1 * t1 + train_x2 * t2 + train_x3 * t3 + t0

# Create loss/cost function
cost = tf.reduce_mean(tf.square(h - train_y))

# Create our optimizer
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.0000001).minimize(cost)

# We need to now run our computational graph
# by creating a session.
with tf.compat.v1.Session() as sess:
    # Initializes global variables in the graph.
    sess.run(tf.compat.v1.global_variables_initializer())

    # Fit a linear line
    for step in range(2001):
        cost_value, hypothesis_value, _ = sess.run([cost, h, train],
            feed_dict={x1: train_x1, x2: train_x2, x3: train_x3, y: train_y})
        # Print the cost function, theta1, theta0 values for each step
        print(f"Step: {step}", f"Cost: {cost_value}")
