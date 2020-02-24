import json
from json import dumps
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D, BatchNormalization
from tensorflow.python.keras.applications import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array


import os
os.listdir("/content/drive/My Drive/Google_Colaboratory/Intel_Images/dataset/seg_train")

data_train = "dataset/seg_train"
data_test = "dataset/seg_test"

data_generator = ImageDataGenerator(horizontal_flip=False,
                                   width_shift_range = 0,
                                   height_shift_range = 0,
                                   zoom_range=0,
                                   rotation_range=0,
                                   )

image_h = 150
image_w = 150
batch_size = 42
learning_rate = 0.0016
epocas = 30

#Lectura de Datos de entrenamiento
train_generator = data_generator.flow_from_directory(
        data_train,
        target_size=(image_h, image_w),
        batch_size=batch_size,
        class_mode='categorical')

#Lectura de Datos de test
test_data = data_generator.flow_from_directory(
                                              data_test,
                                              target_size=(image_h, image_w),
                                              batch_size = batch_size,
                                              class_mode= 'categorical')

num_classes = len(train_generator.class_indices)

# Guardar el diccionario de clases en un archivo json
json = json.dumps(train_generator.class_indices)
f = open("/content/drive/My Drive/Google_Colaboratory/Intel_Images/resnet_model/class_dict.json","w")
f.write(json)
f.close()


model = Sequential()

model.add(ResNet50(include_top=False, pooling='avg', weights=None)) #'imagenet' --> Pesos por defecto en resnet50 (creo)
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(1024, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(num_classes, activation='softmax'))

#model.layers[0].trainable = False

model.summary()


model.compile(optimizer=optimizers.Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])


model.fit_generator(
        train_generator,
        steps_per_epoch= 8000,
        epochs=epocas,
        validation_data=test_data,
        validation_steps=800)

#Save the model and weights
model.save("resnet_model/model_2.h5")
model.save_weights("resnet_model/weights_2.h5")

# Eval SAVED Model
#model = "/content/drive/My Drive/Google_Colaboratory/Intel_Images/model/model_1.h5"
#pesos = "/content/drive/My Drive/Google_Colaboratory/Intel_Images/model/weights_1.h5"

#model = tf.keras.models.load_model(model)
#model.load_weights(pesos)

#Eval Model
model.evaluate(test_data)