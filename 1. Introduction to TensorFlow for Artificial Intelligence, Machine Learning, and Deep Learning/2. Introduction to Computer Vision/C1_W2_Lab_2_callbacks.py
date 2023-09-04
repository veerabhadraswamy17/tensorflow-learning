import tensorflow as tf
fmnist = tf.keras.datasets.fashion_mnist
(training_images,training_labels),(testing_images,testing_labels) = fmnist.load_data()

training_images = training_images/255.0
testing_images = testing_images/255.0

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('loss') < 0.25:
            print('\n Loss is less than 25%, hence stopping the training')
            self.model.stop_training = True

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(units=512,activation=tf.nn.relu),
                                    tf.keras.layers.Dense(units=256,activation=tf.nn.relu),
                                    tf.keras.layers.Dense(units=10,activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
callbacks = myCallback()
model.fit(training_images,training_labels,epochs=10,callbacks=[callbacks])
model.evaluate(testing_images,testing_labels)