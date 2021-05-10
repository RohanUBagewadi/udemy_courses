import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
from keras_preprocessing.image import ImageDataGenerator
from keras_preprocessing import image

train_datagen = ImageDataGenerator(rescale=1 / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
train_generator = train_datagen.flow_from_directory('dataset/training_set',
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='binary')

test_datagen = ImageDataGenerator(rescale=1 / 255)
test_generator = train_datagen.flow_from_directory('dataset/test_set',
                                                   target_size=(64, 64),
                                                   batch_size=32,
                                                   class_mode='binary')

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=(64, 64, 3)))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

model.add(tf.keras.layers.Conv2D(32, (2, 2), activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

model.fit_generator(train_generator, validation_data=test_generator, epochs=80)

test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64, 64))
test_image = np.expand_dims(image.img_to_array(test_image), axis=0)

indices = train_generator.class_indices

result_image = model.predict(test_image)

if result_image > .5:
    print('dog')
else:
    print('cat')
