
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import matplotlib.pyplot as plt

# Training Set
popularity = [3, 8.9, 12, 17, 20, 25, 31.6, 51.2]
price = [500, 900, 1050, 1300, 1250, 1400, 1500, 1550]

# If you'd like to graph the training set, run after uncommenting the following
# plt.scatter(popularity, price, label="Training Set")
# plt.xlabel("Populartiy of ML101")
# plt.ylabel("Priced of ML101")
# plt.show()

# Create a basic Keras model
model = Sequential()
model.add(Dense(1, input_dim=1))

# Gradient descent alogrithm 
sgd = SGD(0.001)

# Cost/Loss Function
model.compile(loss='mse', optimizer=sgd)

# Make the keras model learn/train
history = model.fit(popularity, price, epochs=2000)

# Plot the loss/cost vs interation
plt.plot(history.history['loss'])
plt.show()

# Predict when popularity is 1000 people
print(model.predict([10]))
