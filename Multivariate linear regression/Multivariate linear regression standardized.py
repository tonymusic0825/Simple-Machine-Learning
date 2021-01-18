
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

# Training set
train_x = np.array([[400, 180, 200], [430, 140, 305], [405, 255, 300],
                    [180, 180, 180], [220, 100, 160], [405, 255, 300],
                    [500, 350, 440], [1500, 900, 200], [1500, 900, 900],
                    [1000, 1000, 1000]], dtype=float)

scaler = StandardScaler()
scaler.fit(train_x)
standardized_train_x = scaler.transform(train_x)

train_y = np.array([4.20, 4.85, 6, 3.50, 2.70, 6.50, 11, 15, 20.9, 17.8], dtype=float)

# Create Keras model
model = Sequential()
model.add(Dense(1, input_dim=3))

# Gradient descent algorithm
sgd = SGD(0.01)

model.compile(loss='mse', optimizer=sgd)
history = model.fit(standardized_train_x, train_y, epochs=2000)

plt.plot(history.history['loss'])
plt.xlabel("No. of Iterations")
plt.ylabel("J(Theta1 Theta0)/Cost")
plt.show()

predict = np.array([[73., 80., 75.]])
predict_standardized = scaler.transform(predict)
print(model.predict(predict_standardized))
