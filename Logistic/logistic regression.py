

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Training set
train_x = np.array([[1, 4], [2, 3], [5, 3], [10, 2], [16, 3], 
                    [17, 4], [22, 5], [25, 5]])

train_x = np.array([1, 2, 5, 10, 16, 17, 22, 25])
train_y = np.array([0, 0, 0, 0, 1, 0, 1, 1])

# Create Keras model
model = Sequential()
model.add(Dense(1, input_dim=1, activation='sigmoid'))

# Gradient descent algorithm
sgd = SGD(0.05)

model.compile(loss='binary_crossentropy', optimizer=sgd
              , metrics=['binary_accuracy'])
history = model.fit(train_x, train_y, epochs=1000)

plt.plot(history.history['loss'])
plt.xlabel("No. of Iterations")
plt.ylabel("J(Theta1 Theta0)/Cost")
plt.show()

predict = np.array([[23], [300], [10], [1]])
print(model.predict(predict))
