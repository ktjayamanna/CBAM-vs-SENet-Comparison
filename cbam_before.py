# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 08:18:39 2021

@author: kjayamanna
@Description: This is the code for the CBAM
@Reference: https://github.com/kobiso/CBAM-keras
"""
#%%
import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Concatenate, multiply, Reshape
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPool2D, BatchNormalization
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Concatenate, multiply, Reshape
from tensorflow.keras.layers import Add, Lambda, GlobalMaxPooling2D, Activation
from keras import backend as K
from tensorflow.keras.models import Sequential, Model, load_model
plt.close('all')
#%%
#%%
#swap is the layer index of the layer before SENet. (index starts at 0)
# swap = 5
# swap = 9
# swap = 13
# swap = 17
# swap = 0    
swap = 2   
numSELayers = 14
# rat = 1
# model = tf.keras.applications.VGG16(include_top=True, weights='imagenet')

#%%
layerNames = ['Input',
              'Conv1-1',
              'Conv1-2',
              'Pooling1',
              'Conv2-1',
              'Conv2-2',
              'Pooling2',
              'Conv3-1',
              'Conv3-2',
              'Conv3-3',
              'Pooling3',
              'Conv4-1',
              'Conv4-2',
              'Conv4-3',
              'Pooling4',
              'Conv5-1',
              'Conv5-2',
              'Conv5-3',
              'Pooling5',
              'Dense1',
              'Dense2',
              'Dense3',
              'Output']
#%%
# %%
def channel_attention(input_feature, ratio=1):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature.shape[channel_axis]
 	
    shared_layer_one = Dense(channel//ratio,
							 activation='relu',
							 kernel_initializer='he_normal',
							 use_bias=True,
							 bias_initializer='zeros')
    shared_layer_two = Dense(channel,
							 kernel_initializer='he_normal',
							 use_bias=True,
							 bias_initializer='zeros')
 	
    avg_pool = GlobalAveragePooling2D()(input_feature)    
    avg_pool = Reshape((1,1,channel))(avg_pool)
    # assert avg_pool.shape[1:] == (1,1,channel)
    avg_pool = shared_layer_one(avg_pool)
    # assert avg_pool.shape[1:] == (1,1,channel//ratio)
    avg_pool = shared_layer_two(avg_pool)
    # assert avg_pool.shape[1:] == (1,1,channel)
 	
    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1,1,channel))(max_pool)
    # assert max_pool.shape[1:] == (1,1,channel)
    max_pool = shared_layer_one(max_pool)
    # assert max_pool.shape[1:] == (1,1,channel//ratio)
    max_pool = shared_layer_two(max_pool)
    # assert max_pool.shape[1:] == (1,1,channel)
 	
    cbam_feature = Add()([avg_pool,max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)
 	
    # if K.image_data_format() == "channels_first":
    #       cbam_feature = Permute((3, 1, 2))(cbam_feature)
 	
    return multiply([input_feature, cbam_feature])

def spatial_attention(input_feature):
    kernel_size = 7
    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(input_feature)
    # assert avg_pool.shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(input_feature)
    # assert max_pool.shape[-1] == 1
    
    
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    # assert concat.shape[-1] == 2
    cbam_feature = Conv2D(filters = 1,
 					kernel_size=kernel_size,
 					strides=1,
 					padding='same',
 					activation='sigmoid',
 					kernel_initializer='he_normal',
 					use_bias=False)(concat)	
    # assert cbam_feature.shape[-1] == 1	
    return multiply([input_feature, cbam_feature])

def cbam_block(cbam_feature, ratio=1):
 	"""Contains the implementation of Convolutional Block Attention Module(CBAM) block.
 	As described in https://arxiv.org/abs/1807.06521.
 	"""
 	
 	cbam_feature = channel_attention(cbam_feature, ratio)
 	cbam_feature = spatial_attention(cbam_feature)
 	return cbam_feature

#%%
input_shape = (224,224,3)
img_input = Input(shape=input_shape, name = layerNames[0])
x = img_input
# x = cbam_block(x)
# Block ,1
x = Conv2D(64, (3, 3),
                  activation='relu',
                  padding='same',
                  name = layerNames[1])(x)



x = Conv2D(64, (3, 3),
                  activation='relu',
                  padding='same',
                  name = layerNames[2])(x)

#
x = cbam_block(x)
#

x = MaxPool2D((2, 2), strides=(2, 2), name = layerNames[3])(x)







# Block 2
x = Conv2D(128, (3, 3),
                  activation='relu',
                  padding='same',
                  name = layerNames[4])(x)
x = Conv2D(128, (3, 3),
                  activation='relu',
                  padding='same',
                  name = layerNames[5])(x)

# # #
# x = cbam_block(x)
# # #

x = MaxPool2D((2, 2), strides=(2, 2), name = layerNames[6])(x)



# Block 3
x = Conv2D(256, (3, 3),
                  activation='relu',
                  padding='same',
                  name = layerNames[7])(x)
x = Conv2D(256, (3, 3),
                  activation='relu',
                  padding='same',
                  name = layerNames[8])(x)
x = Conv2D(256, (3, 3),
                  activation='relu',
                  padding='same',
                  name = layerNames[9])(x)

# # #
# x = cbam_block(x)
# # #

x = MaxPool2D((2, 2), strides=(2, 2), name = layerNames[10])(x)



# Block 4
x = Conv2D(512, (3, 3),
                  activation='relu',
                  padding='same',
                  name = layerNames[11])(x)
x = Conv2D(512, (3, 3),
                  activation='relu',
                  padding='same',
                  name = layerNames[12])(x)
x = Conv2D(512, (3, 3),
                  activation='relu',
                  padding='same',
                  name = layerNames[13])(x)

# # #
# x = cbam_block(x)
# # #

x = MaxPool2D((2, 2), strides=(2, 2), name = layerNames[14])(x)



# Block 5
x = Conv2D(512, (3, 3),
                  activation='relu',
                  padding='same',
                  name = layerNames[15])(x)
x = Conv2D(512, (3, 3),
                  activation='relu',
                  padding='same',
                  name = layerNames[16])(x)
x = Conv2D(512, (3, 3),
                  activation='relu',
                  padding='same',
                  name = layerNames[17])(x)

# #
# x = cbam_block(x)
# #
x = MaxPool2D((2, 2), strides=(2, 2), name = layerNames[18])(x)



# Classification block
x = Flatten(name = layerNames[19])(x)
x = Dense(4096, activation='relu', name = layerNames[20])(x)
x = Dense(4096, activation='relu', name = layerNames[21])(x)
classes = 3
# classes = 10
# x = layers.Dense(classes, activation='softmax', name='Output')(x)
x = Dense(classes, activation='softmax', name = layerNames[22])(x)
model = Model(img_input, x, name='vgg16')
#%%
trained_model = tf.keras.applications.VGG16(include_top=True, weights='imagenet')
#%
trainedWeights = []
for i in trained_model.layers:
  trainedWeights.append(i.get_weights())
#%% Take care of predecessors of SeNet
## Uncomment this for loop and comment below too for loops if you need the base model.
# for k in range(len(trainedWeights)-1):
#     model.layers[k].set_weights(trainedWeights[k])
#     model.layers[k].trainable = False

#Uncomment these two and comment the above for loop if you don't need the base model.
for k in range(0, swap + 1):
    model.layers[k].set_weights(trainedWeights[k])
    # model.layers[k]._name = layerNames[k]
    model.layers[k].trainable = False
#%  -1 in the loop to exclude the last layer swap
for i in range(swap + 1 + numSELayers,len(trainedWeights) + numSELayers - 1):
  model.layers[i].set_weights(trainedWeights[i-numSELayers])
  # model.layers[i]._name = layerNames[i-numSELayers]
  model.layers[i].trainable = False
#%% Make the last layer Trainable just in case if it is not already.
model.layers[-1].trainable = True
#%%
#Num of images the model run through before updating the model weights.
batch_size = 8
# This is the augmentation configuration we will use for training.
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255)


test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255)

#%%Define the image generators.
train_generator = train_datagen.flow_from_directory(
        r'C:\Users\keven\OneDrive - University of Nebraska at Omaha\Dr.Khazanchi\Week 1\Medical Imaging\hw 4\Reduced\training',  
        target_size=(224, 224), 
        batch_size=batch_size,
        class_mode='sparse'
        )  
#%
validation_generator = test_datagen.flow_from_directory(
        r'C:\Users\keven\OneDrive - University of Nebraska at Omaha\Dr.Khazanchi\Week 1\Medical Imaging\hw 4\Reduced\validation',
        target_size=(224, 224),
        batch_size=batch_size,
        shuffle = False,
        class_mode='sparse'
        )
#%%Define the optimizer.lr=3e-4
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
# optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
#%%  Compile the model.
model.compile(loss= keras.losses.SparseCategoricalCrossentropy(),
              optimizer=optimizer,
              metrics=['accuracy'])
#%%
monitor = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-3, 
        patience=5, verbose=1, mode='auto',
        restore_best_weights=True)
#%%
hist = model.fit(
        train_generator,
        validation_data=validation_generator,callbacks=[monitor], verbose=1,epochs=100)
#%% Get the Training metrics
acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']
epochs=range(len(acc)) # Get number of epochs
#%%
# =============================================================================
# Plot training and validation accuracy per epoch
# =============================================================================
plt.figure()
plt.plot(epochs, acc, label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training & Validation accuracy')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.figure()
plt.plot(epochs, loss, label='Training Loss')
plt.title('Training & Validation loss Using Keras Early Stopping')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
#%% Define test image generator.
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        r'C:\Users\keven\OneDrive - University of Nebraska at Omaha\Dr.Khazanchi\Week 1\Medical Imaging\hw 4\Reduced\testing',
        target_size=(224, 224),
        batch_size=1,
        class_mode='sparse',
        shuffle = False,
        )
#%% Make predictions with test generator.
preds = model.evaluate(test_generator)
#%%
model.save('position2b.h5')
