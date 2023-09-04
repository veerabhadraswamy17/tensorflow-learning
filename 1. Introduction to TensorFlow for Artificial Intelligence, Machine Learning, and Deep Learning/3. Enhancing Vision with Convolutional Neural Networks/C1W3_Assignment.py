import numpy as np
import tensorflow as tf
from tensorflow import keras

fmnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (testing_images, testing_labels) = fmnist.load_data()


def reshape_and_normalize(training_images, testing_images):
    testing_images = np.expand_dims(testing_images, axis=-1)
    training_images = np.expand_dims(training_images, axis=-1)

    testing_images = testing_images / 255.0
    training_images = training_images / 255.0

    return training_images, testing_images


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.95:
            print('\n Training accuracy is more than 99%, hence stopping the training')
            self.model.stop_training = True

training_images,testing_images=reshape_and_normalize(training_images,testing_images)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),activation=tf.nn.relu,input_shape=(28,28,1)),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128,activation=tf.nn.relu),
    tf.keras.layers.Dense(units=10,activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
callbacks = myCallback()

model.fit(training_images,training_labels,epochs=10,callbacks=[callbacks])

model.evaluate(testing_images,testing_labels)