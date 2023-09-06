import tensorflow as tf
import numpy as np
import os

import tensorflow.python.keras.callbacks
#from tensorflow.keras.preprocessing.image import image_to_array
from keras.utils import img_to_array
from keras.utils import load_img
from keras.preprocessing.image import ImageDataGenerator


happy_dir = "./data/Happy/"
sad_dir = "./data/Sad/"
angry_dir = "./data/Angry/"

# sample_image  = load_img(f"{os.path.join(happy_dir, os.listdir(happy_dir)[0])}")
# sample_array = img_to_array(sample_image)
# print(f"Each image has shape: {sample_array.shape}")
# print(f"The maximum pixel value used is: {np.max(sample_array)}")

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.99:
            print("\nStopping the training, as accuracy reached  90%")
            self.model.stop_training = True


def image_generator():
    train_datagen = ImageDataGenerator(rescale=1/255)
    train_generator = train_datagen.flow_from_directory(directory='./data/',
                                                        target_size=(150,150),
                                                        batch_size=5,
                                                        class_mode='sparse')
    return train_generator


gen = image_generator()


def train_sad_angry_happy_model(train_generator):

    callbacks = myCallback()

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=512,activation='relu'),
        #tf.keras.layers.Dense(units=128,activation='relu'),
        tf.keras.layers.Dense(units=3,activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(train_generator,
                        epochs=20,
                        callbacks=[callbacks])

    return history, model


hist, trained_model = train_sad_angry_happy_model(gen)
print(f"model reached desired accuracy after {len(hist.epoch)} epochs")

#from keras.preprocessing import image

images = os.listdir("./tmp/images/")

print(images)

for i in images:
    print('predicting images')
    path = './tmp/images/' + i
    img = load_img(path,target_size=(150,150))
    x = img_to_array(img)
    x = np.expand_dims(x,axis=0)

    images = np.vstack([x])
    classes = trained_model.predict(images,batch_size=5)
    print(path)
    if classes[0][1] == 1.0:
        print('person is happy')
        print(classes)
    elif classes[0][0] == 1.0:
        print('person is Angry')
        print(classes)
    else:
        print('person is sad')
        print(classes)
    # print(classes[0][0])



