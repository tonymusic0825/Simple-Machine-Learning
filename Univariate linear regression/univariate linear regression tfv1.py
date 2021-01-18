
import tensorflow as tf

# Training set
popularity = [3, 8.9, 12, 17, 20, 25, 31.6, 51.2]
price = [500, 900, 1050, 1300, 1250, 1400, 1500, 1550]

# You will NOT need to do this if you're using tensorflow v1
tf.compat.v1.disable_eager_execution()

# Remember that we are trying to compute the values of theta 1 and theta0,
# so let's set them with normal distribution of values
theta1 = tf.Variable(tf.compat.v1.random_normal([1]), name="weight")
theta0 = tf.Variable(tf.compat.v1.random_normal([1]), name="theta0")

# Create our hypothesis
h = popularity * theta1 + theta0

# Create loss/cost function
cost = tf.reduce_mean(tf.square(h - price))

# Create our optimizer
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.0015).minimize(cost)

# We need to now run our computational graph
# by creating a session.
with tf.compat.v1.Session() as sess:
    # Initializes global variables in the graph.
    sess.run(tf.compat.v1.global_variables_initializer())

    # Fit a linear line
    for step in range(2001):
        _, cost_value, theta1_value, theta0_value = sess.run([train, cost, theta1, theta0])
        # Print the cost function, theta1, theta0 values for each step
        if step % 20 == 0:
            print(step, cost_value, theta1_value, theta0_value)
