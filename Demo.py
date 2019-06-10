import tensorflow as tf
import keras

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
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)), # retrofitted for demo (inputsize)
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.AvgPool2D(2, 2),

    # Conv block 1
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.Conv2D(64, (3, 3), activation='sigmoid'),
    keras.layers.AvgPool2D(2, 2),

    # Dense layer
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='sigmoid'),
    keras.layers.Dense(10, activation='softmax') # retrofitted for demo (classifications)
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=10)
print("Training complete testing:")
model.evaluate(x_test, y_test)
