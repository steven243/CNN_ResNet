import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array

image_h = 150
image_w = 150


filename = "resnet_model/class_dict.json"

with open(filename, 'r') as f:
  class_dict = json.load(f)

modelo = "resnet_model/model_1.h5"
pesos = "resnet_model/weights_1.h5"

model = tf.keras.models.load_model(modelo)
model.load_weights(pesos)


from IPython.display import Image, display

import os, random
img_locations = []

directory = "dataset/seg_pred/"
sample = [directory + '/' + s for s in random.sample(os.listdir(directory), 15)]
img_locations += sample

#Display predicted class images
def read_and_prep_images(img_paths, img_height=image_h, img_width=image_w):
    imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]
    img_array = np.array([img_to_array(img) for img in imgs])
    return preprocess_input(img_array)

random.shuffle(img_locations)
imgs = read_and_prep_images(img_locations)
predictions = model.predict_classes(imgs)
classes = dict((v,k) for k,v in class_dict.items())

for img, prediction in zip(img_locations, predictions):
    display(Image(img))
    print(classes[prediction])