# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 09:31:18 2021

@author: kjayamanna
Ref: https://github.com/kobiso/CBAM-keras/blob/master/models/attention_module.py
https://androidkt.com/keras-confusion-matrix-in-tensorboard/
Description: This experiment uses the OOT data to train the model. And the
original set is used 
"""
#%%
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import Model
import numpy as np
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
plt.close('all')
#%%
'''
Swap Table
swap = 0 for position1
swap = 3 for position2
swap = 6 for position3
swap = 10 for position4
swap = 14 for position5
swap = 19 for position6
'''
#swap is the layer index of the layer before SENet. (index starts at 0)
swap = 19
numSELayers = 5
rat = 1
#%%
def se_block(input_feature, ratio=rat):
    """Contains the implementation of Squeeze-and-Excitation(SE) block.
    As described in https://arxiv.org/abs/1709.01507.
    """

    # channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel_axis = -1
    channel = input_feature.shape[channel_axis]
    

    se_feature = layers.GlobalAveragePooling2D()(input_feature)
    se_feature = layers.Reshape((1, 1, channel))(se_feature)
    assert se_feature.shape[1:] == (1,1,channel)
    
    # denseShape = math.ceil(channel / ratio)
    denseShape = channel // ratio

    se_feature = layers.Dense(denseShape,
                       # activation=None,
                       activation='relu',
                       kernel_initializer='he_normal',
                       use_bias=True,
                       bias_initializer='zeros')(se_feature)

    
    assert se_feature.shape[1:] == (1,1,denseShape)
    se_feature = layers.Dense(channel,
                       # activation= None,
                       activation='sigmoid',
                       kernel_initializer='he_normal',
                       use_bias=True,
                       bias_initializer='zeros')(se_feature)


    assert se_feature.shape[1:] == (1,1,channel)

    se_feature = layers.Multiply()([input_feature, se_feature])
    return se_feature

#%%
input_shape = (224,224,3)
img_input = layers.Input(shape=input_shape)
trained_model = tf.keras.applications.VGG16(include_top=True, weights='imagenet')
#%%
trainedWeights = []
for i in trained_model.layers:
  trainedWeights.append(i.get_weights())

#%%

# x = se_block(img_input)
x = img_input
# Block ,1
x = layers.Conv2D(64, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block1_conv1')(x)



x = layers.Conv2D(64, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block1_conv2')(x)

x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

#
# x = se_block(x)
#



# Block 2
x = layers.Conv2D(128, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block2_conv1')(x)
x = layers.Conv2D(128, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block2_conv2')(x)
x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

# # #
# x = se_block(x)
# # #

# Block 3
x = layers.Conv2D(256, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block3_conv1')(x)
x = layers.Conv2D(256, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block3_conv2')(x)
x = layers.Conv2D(256, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block3_conv3')(x)
x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

# # #
# x = se_block(x)
# # #

# Block 4
x = layers.Conv2D(512, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block4_conv1')(x)
x = layers.Conv2D(512, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block4_conv2')(x)
x = layers.Conv2D(512, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block4_conv3')(x)
x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

# # #
# x = se_block(x)
# # #

# Block 5
x = layers.Conv2D(512, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block5_conv1')(x)
x = layers.Conv2D(512, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block5_conv2')(x)
x = layers.Conv2D(512, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block5_conv3')(x)
x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

# #
x = se_block(x)
# #

# Classification block
x = layers.Flatten(name='flatten1')(x)
x = layers.Dense(4096, activation='relu', name='fc1')(x)
x = layers.Dense(4096, activation='relu', name='fc2')(x)
classes = 3
# classes = 10
# x = layers.Dense(classes, activation='softmax', name='Output')(x)
x = layers.Dense(classes, activation='softmax', name='Output')(x)
model = Model(img_input, x, name='vgg16')
#%% Take care of predecessors of SeNet
## Uncomment this for loop and comment below two for loops if you need the base model.
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
batch_size = 8
# this is the augmentation configuration we will use for training
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        )


dev_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        )

#%%
train_generator = train_datagen.flow_from_directory(
        r'C:\Users\keven\OneDrive - University of Nebraska at Omaha\Dr.Khazanchi\Week 1\Medical Imaging\hw 4\Reduced\training',  # this is the target directory
        target_size=(224, 224), 
        batch_size=batch_size,
        class_mode='sparse'
        )  
#%%
validation_generator = dev_datagen.flow_from_directory(
        r'C:\Users\keven\OneDrive - University of Nebraska at Omaha\Dr.Khazanchi\Week 1\Medical Imaging\hw 4\Reduced\validation',
        target_size=(224, 224),
        batch_size= batch_size,
        shuffle = False,
        class_mode='sparse'
        )
#%%3e-4
optimizer = keras.optimizers.Adam(lr=3e-4)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy',
                       ])


#%%
monitor = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-3, 
        patience=5, verbose=1, mode='auto',
        restore_best_weights=True)
#%%
hist = model.fit(
        train_generator,
        validation_data=validation_generator,callbacks=[monitor], verbose=1,epochs=100)

#%%
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
plt.title('Training & Validation loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
# %%
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        )
test_generator = test_datagen.flow_from_directory(
        r'C:\Users\keven\OneDrive - University of Nebraska at Omaha\Dr.Khazanchi\Week 1\Medical Imaging\hw 4\Reduced\testing',
        target_size=(224, 224),
        batch_size= batch_size,
        shuffle = False,
        class_mode='sparse'
        )
#%%
preds = model.evaluate(test_generator)
print('Test Accracy is '+ str(preds[1]*100)+ '%')
#%%
model.save('position6.h5');
model.summary()



    
