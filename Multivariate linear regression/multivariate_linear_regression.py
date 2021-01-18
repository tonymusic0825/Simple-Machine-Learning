import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np


# Training set
train_x = np.array([[400, 180, 200], [430, 140, 305], [405, 255, 300],
                    [180, 180, 180], [220, 100, 160], [405, 255, 300],
                    [500, 350, 440], [1500, 900, 200], [1500, 900, 900],
                    [1000, 1000, 1000]], dtype=float)
train_y = np.array([4.20, 4.85, 6, 3.50, 2.70, 6.50, 11, 20.5, 39.8, 35.3], dtype=float)

# train_x = np.array([[73, 80, 75], [93, 88, 93], [89, 91, 90],
#                     [96, 98, 100], [73, 66, 70]], dtype=float)
# train_y = np.array([152, 185, 180, 196, 142], dtype=float)

# Create Keras model
model = Sequential()
model.add(Dense(1, input_dim=3))

# Gradient descent algorithm
sgd = SGD(0.0000001)

model.compile(loss='mse', optimizer=sgd)
history = model.fit(train_x, train_y, epochs=2000)

plt.plot(history.history['loss'])
plt.xlabel("No. of Iterations")
plt.ylabel("J(Theta1 Theta0)/Cost")
plt.show()

predict = np.array([[73., 80., 75.]])
print(model.predict(predict))







