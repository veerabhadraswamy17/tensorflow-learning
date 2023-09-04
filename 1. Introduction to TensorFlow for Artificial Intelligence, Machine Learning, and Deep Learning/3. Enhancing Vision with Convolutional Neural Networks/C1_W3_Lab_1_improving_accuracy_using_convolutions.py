import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import models

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
            print(f"\nAccuracy has reached more than 95%, hence stopping the training")
            self.model.stop_training = True


training_images,testing_images = reshape_and_normalize(training_images,testing_images)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),activation=tf.nn.relu,input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32,(3,3),activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128,activation=tf.nn.relu),
    tf.keras.layers.Dense(units=10,activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics='accuracy')

callbacks = myCallback()

model.fit(training_images,training_labels,epochs=10,callbacks=[callbacks])

test_loss = model.evaluate(testing_images,testing_labels)

f, axarr = plt.subplots(3,4)
First_Image = 0
Second_Image = 23
Third_Image = 28
ConvolutionNumber=1

layer_outputs = [layer.output for layer in model.layers]
activation_model = tf.keras.models.Model(inputs = model.input, outputs= layer_outputs)

for x in range(0,4):
    f1 = activation_model.predict(testing_images[First_Image].reshape(1,28,28,1))[x]
    axarr[0,x].imshow(f1[0, : , :, ConvolutionNumber], cmap='inferno')
    axarr[0, x].grid(False)
    f2 = activation_model.predict(testing_images[Second_Image].reshape(1,28,28,1))[x]
    axarr[0,x].imshow(f2[0, : , :, ConvolutionNumber], cmap='inferno')
    axarr[0, x].grid(False)
    f3 = activation_model.predict(testing_images[Third_Image].reshape(1,28,28,1))[x]
    axarr[0,x].imshow(f3[0, : , :, ConvolutionNumber], cmap='inferno')
    axarr[0, x].grid(False)



