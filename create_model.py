from textblob import Word
import cv2
import numpy as np
import tensorflow as tf
from tf_keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tf_keras.applications.mobilenet_v2 import decode_predictions
import argostranslate.translate
from PIL import ImageFont, ImageDraw, Image
from tf_keras.preprocessing import image_dataset_from_directory
import tf_keras
import tensorflow as tf
import os

PATH = 'dataset2/'
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'test')
#LABELS = ['fresh apple', 'fresh banana', 'fresh orange', 
#          'rotten apple', 'rotten banana', 'rotten orange']
LABELS = os.listdir('dataset2/train')
BATCH_SIZE = 32
EPOCHS = 20
IMG_SIZE = (100, 100)
NUM_CLASSES = len(LABELS)

train_dataset = image_dataset_from_directory(train_dir,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE,
                                             label_mode='categorical')

validation_dataset = image_dataset_from_directory(validation_dir,
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE,
                                                  image_size=IMG_SIZE,
                                                  label_mode='categorical')

val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 5)
validation_dataset = validation_dataset.skip(val_batches // 5)

val_batches_num = tf.data.experimental.cardinality(validation_dataset)
test_batches_num = tf.data.experimental.cardinality(test_dataset)
print('Number of validation batches: %d' % val_batches_num)
print('Number of test batches: %d' % test_batches_num)

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

preprocess_input = tf_keras.applications.mobilenet_v2.preprocess_input

IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf_keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

base_model.trainable = False

image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)
print(feature_batch.shape)
base_model.summary()
global_average_layer = tf_keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)
prediction_layer = tf_keras.layers.Dense(NUM_CLASSES, activation='softmax')
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)
inputs = tf_keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = preprocess_input(inputs)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf_keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf_keras.Model(inputs, outputs)
base_learning_rate = 0.0001
model.compile(optimizer=tf_keras.optimizers.Adam(lr=base_learning_rate),
              loss=tf_keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()
initial_epochs = 10

loss0, accuracy0 = model.evaluate(validation_dataset)

print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

history = model.fit(train_dataset,
                    epochs=EPOCHS,
                    validation_data=validation_dataset)
model.save('fruit_classifier_fruit360.h5')
loss_final, accuracy_final = model.evaluate(test_dataset)
print("Final loss: {:.2f}".format(loss_final))
print("Final accuracy: {:.2f}".format(accuracy_final))