import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer as timer

model = keras.models.load_model('model.h5')
model.summary()
exit()
# basic mnist setup from tf.org tutorial https://www.tensorflow.org/overview/
mnist = tf.keras.datasets.mnist

# test / train split
(x_train, y_train),(x_test, y_test) = mnist.load_data()

# normalise data
x_train = x_train / 255.0
x_test = x_test / 255.0


# mould data
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)


times = []
for i in range(100, len(x_train), 100):
    print(i)
    start = timer()
    model.predict(x_train, steps=i)
    end = timer()
    times.append(end-start)

plt.title("Running times over input size")
plt.plot(list(range(100, len(x_train), 100)), times)
plt.xlabel("Input Size")
plt.ylabel("Running time")
plt.show()
