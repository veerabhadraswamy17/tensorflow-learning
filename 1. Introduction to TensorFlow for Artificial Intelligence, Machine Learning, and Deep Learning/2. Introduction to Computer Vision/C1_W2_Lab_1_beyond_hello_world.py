# import libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

fmnist = tf.keras.datasets.fashion_mnist

(training_images,training_labels),(testing_images,testing_labels) = fmnist.load_data()

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') is not None and logs.get('accuracy') > 0.85:
            print(f"\nReached accuracy of 85%, stopping the training")
            self.model.stop_training = True

# index = 1
# np.set_printoptions(linewidth=200)
# print(f"Label: {training_labels[index]}")
# print(f"\nImage Pixel Array:\n {training_images[index]}")
# plt.imshow(training_images[index])

training_images = training_images/255.0
testing_images = testing_images/255.0

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(units=128,activation=tf.nn.relu),
                                    tf.keras.layers.Dense(units=10,activation=tf.nn.softmax)
])

model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
callbacks = myCallback()

model.fit(training_images,training_labels,epochs=5,callbacks=[callbacks])
model.evaluate(testing_images,testing_labels)