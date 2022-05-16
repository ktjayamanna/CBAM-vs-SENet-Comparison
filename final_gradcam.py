# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 15:35:23 2021
Reference: https://keras.io/examples/vision/grad_cam/
@author: kjayamanna
"""
import os
from tensorflow import keras
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from matplotlib import pyplot
import grad_cam as gc
from keras import backend as K
#%%
img_path = r"C:\Users\keven\Desktop\my\taperloc_bernadette_xray_x2.22_y12.46_z24.40_att0.07.png"
img_size = (224, 224)
path = r"C:\Users\keven\OneDrive - University of Nebraska at Omaha\Fall 2021\CSCI 8110\Final\Dataset\se_after"
my_models = os.listdir(path)
super_imposed = []
for i in my_models:
    temp_path = os.path.join(path,i)
    model = keras.models.load_model(temp_path);
    img_array = gc.preprocess_input(gc.get_img_array(img_path, size=img_size))
    preds = model.predict(img_array)
    try:
        heatmap = gc.make_gradcam_heatmap(img_array, model, 'Pooling5', pred_index=None)
    except ValueError:
        heatmap = gc.make_gradcam_heatmap(img_array, model, 'block5_pool', pred_index=None)
    super_imposed.append(gc.save_and_display_gradcam(img_path, heatmap))

#%% Before
fig = pyplot.figure(figsize=(20,15))
for i in range(1,len(super_imposed)+1):
    pyplot.subplot(1,7,i)
    pyplot.imshow(super_imposed[i-1]/255)    
pyplot.show()
pyplot.suptitle("Before CBAM which is located between pooling and dense")





