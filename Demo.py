import matplotlib.pyplot as plt
import tensorflow as tf
import keras
# import timeit.default_timer as timer

# Because the code dosent work i though i would demonstraght the classifier part of the network

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


model = keras.models.Sequential([
    # Conv block 1
    keras.layers.Conv2D(128, (3, 3), activation='relu', shape=(28, 28, 1)), # retrofitted for demo (inputsize)
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.AvgPool2D(2, 2),

    # Conv block 2
    keras.layers.Conv2D(256, (3, 3), activation='relu'),
    keras.layers.Conv2D(256, (3, 3), activation='sigmoid'),
    keras.layers.AvgPool2D(2, 2),

    # Conv block 3
    keras.layers.Conv2D(512, (3, 3), activation='relu'),
    keras.layers.Conv2D(512, (3, 3), activation='sigmoid'),
    keras.layers.AvgPool2D(2, 2),

    # Dense layer
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='sigmoid'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='sigmoid'),
    keras.layers.Dense(10, activation='softmax') # retrofitted for demo (classifications)
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

his_train = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
model.save('model.h5')
# print("Training complete testing:")
# loss, accuracy = model.evaluate(x_test, y_test)
# times = []
# for i in range(len(x_train), 100):
#     start = timer()
#     model.predict(x_train[:i])
#     end = timer()
#     times.append(end-start)
# # plt.title("Running time over Input size")
# # plt.ylabel("Running time")
# # plt.xlabel("Input size")
# # plt.plot(times, )
# # plt.show()
# print("Accuracy: {}%".format(accuracy))

