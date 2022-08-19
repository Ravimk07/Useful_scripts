'''
# Created on 12-Nov-2019
# @author: Ravi
'''  
import os
import tensorflow as tf
import glob
from keras.optimizers import adam
from keras.backend.tensorflow_backend import set_session
import numpy as np
import pandas as pd
import cv2
import random
import gc
from keras.callbacks import TensorBoard
import keras
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import sgd
from keras.layers import (Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D,
                          BatchNormalization, Input, Conv2D, GlobalAveragePooling2D,concatenate,Concatenate,multiply, LocallyConnected2D, Lambda)
from sklearn.model_selection import KFold
# import sparsenet
import numpy as np
# import sklearn.metrics as metrics
# from losses import binary_focal_loss 
import keras.backend as K
K.set_image_data_format('channels_last')

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
np.random.seed(1337)  # for reproducibility

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.95
set_session(tf.Session(config=config))
 
# Training parameters 
epochs = 250
data_augmentation = True 
num_classes = 28
#  

nrows = 380
ncolumns = 380 
channels = 3

train_dir = '/mnt/X/Ravi K/RIDD/New_data/b4/'
# # # del train_AMD
# # # del train_non_AMD
gc.collect()


print('check_3')

img_rows, img_cols = 380, 380
# img_rows, img_cols = 460, 460
img_channels = 3 

#img_dim = (img_channels, img_rows, img_cols) if K.image_dim_ordering() == "th" else (img_rows, img_cols, img_channels)
img_dim = (img_rows, img_cols, img_channels)
# modelPath = '/mnt/X/Ravi K/DR_Analysis/data/HVDEH/model/effic_B4_noisy_HVDEH_attention.036-0.295312-0.921986.h5'
def swish_activation(x): 
    return x * K.sigmoid(x)

 
import efficientnet.keras as efn 
def create_model():
    in_lay = Input(shape=(380,380,3))
    base_model = efn.EfficientNetB4(input_shape=img_dim, weights='noisy-student', include_top=False)  # or weights='noisy-student'
    # base_model = efn.EfficientNetB4(input_shape=img_dim, weights=weights, include_top=False)
    
    pt_depth = base_model.get_output_shape_at(0)[-1]
    pt_features = base_model(in_lay)
    bn_features = BatchNormalization()(pt_features) 
    
    # here we do an attention mechanism to turn pixels in the GAP on an off
    attn_layer = Conv2D(64, kernel_size = (1,1), padding = 'same', activation = 'relu')(Dropout(0.5)(bn_features))
    attn_layer = Conv2D(16, kernel_size = (1,1), padding = 'same', activation = 'relu')(attn_layer)
    attn_layer = Conv2D(8, kernel_size = (1,1), padding = 'same', activation = 'relu')(attn_layer)
    attn_layer = Conv2D(1, 
                        kernel_size = (1,1), 
                        padding = 'valid', 
                        activation = 'sigmoid')(attn_layer)
    # fan it out to all of the channels
    up_c2_w = np.ones((1, 1, 1, pt_depth))
    up_c2 = Conv2D(pt_depth, kernel_size = (1,1), padding = 'same', 
                   activation = 'linear', use_bias = False, weights = [up_c2_w])
    up_c2.trainable = False
    attn_layer = up_c2(attn_layer)
    
    mask_features = multiply([attn_layer, bn_features])
    gap_features = GlobalAveragePooling2D()(mask_features)
    gap_mask = GlobalAveragePooling2D()(attn_layer)
    # to account for missing values from the attention model
    gap = Lambda(lambda x: x[0]/x[1], name = 'RescaleGAP')([gap_features, gap_mask])
    gap_dr = Dropout(0.25)(gap)
    dr_steps = Dropout(0.25)(Dense(128, activation = 'relu')(gap_dr))
    out_layer = Dense(28, activation = 'sigmoid')(dr_steps)
    model = Model(inputs = [in_lay], outputs = [out_layer])
    opt = adam(lr=1e-4)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])
    return model

img_dim = (380, 380, 3)
def create_model_effb4():
    base_model = efn.EfficientNetB4(input_shape=img_dim, weights='noisy-student', include_top=False)  # or weights='noisy-student'
 
# from keras_efficientnets import EfficientNetB7 
# base_model = EfficientNetB7(input_shape=img_dim, weights='imagenet', include_top=False) 

    x = base_model.output  
    
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation=swish_activation, name='fc1')(x)
    x = Dropout(0.6)(x)
    x = Dense(256, activation=swish_activation, name='fc2')(x)
    x = Dropout(0.6)(x)
    x = Dense(128, activation=swish_activation, name='fc3')(x)
    x = Dropout(0.6)(x)
    
    predictions = Dense(28, activation='sigmoid', name='predictions')(x)
    # add the top layer block to your base model
    model = Model(base_model.input, predictions)
    opt = adam(lr=1e-4) 
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])
    return model

# model.load_weights(modelPath)
# x = base_model.output   
# 
# x = GlobalAveragePooling2D()(x)
# x = Dense(512, activation=swish_activation, name='fc1')(x)
# x = Dropout(0.4)(x)
# x = Dense(256, activation=swish_activation, name='fc2')(x)
# x = Dropout(0.3)(x)
# x = Dense(128, activation=swish_activation, name='fc3')(x)
# x = Dropout(0.3)(x)
# 
# predictions = Dense(5, activation='softmax', name='predictions')(x)
# # add the top layer block to your base model
# model = Model(base_model.input, predictions) 
#print(model.summary())
print("ok")
X_train = np.load('/mnt/X/Ravi K/RIDD/New_data/b4/Train_data2.npy')
Y_train = np.load('/mnt/X/Ravi K/RIDD/New_data/b4/Train_label2.npy')
model_path = '/mnt/X/Ravi K/RIDD/New_data/b4/modelb4/'
log_path = '/mnt/X/Ravi K/RIDD/New_data/b4/log/'
# X_val = np.load('/home/aman/Desktop/aman/RIDD challenge/trial-7 fold b4 28 class/fold 4/Val_data.npy')
# y_val = np.load('/home/aman/Desktop/aman/RIDD challenge/trial-7 fold b4 28 class/fold 4/val_label.npy')

# print("Shape of the train images is:", X_train.shape)
# print("Shape of labels is:", y_train.shape)
# print(np.unique(y_val))
print(np.unique(Y_train))
# print("Shape of the Validation images is:", X_val.shape)
# print("Shape of labels is:", y_val.shape)

# ntrain =len(X_train) 
# nval =len(X_val)
batch_size = 4
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_val = keras.utils.to_categorical(y_val, num_classes)
print(Y_train.shape)
############################## K fold ######################################
fold_var = 1
skf = KFold(n_splits=5, random_state=7,shuffle=True)
for train_index,val_index in skf.split(X_train,Y_train):
    X_t,X_v = X_train[train_index],X_train[val_index]
    Y_t,Y_v = Y_train[train_index],Y_train[val_index]
    print(X_t.shape)
    print(X_v.shape)
    print(Y_t.shape)
    print(Y_v.shape)
    model = create_model_effb4()
    filepath_loss = model_path + 'low_loss_fold_'+str(fold_var) + '.h5'
    checkpoint1 = ModelCheckpoint(filepath=filepath_loss,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only= False,)
    filepath_acc = model_path + 'high_acc_fold_'+str(fold_var) + '.h5'
    checkpoint2 = ModelCheckpoint(filepath=filepath_acc,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only= False,)
    newpath = log_path + str(fold_var)+'/'    # Path for storing tensorboard file 
    tensorboard = TensorBoard(log_dir = newpath)
    callbacks = [checkpoint1,checkpoint2,tensorboard]
    model.fit(X_t, Y_t,
          batch_size=14,
          epochs=60, 
          validation_data=(X_v, Y_v), 
          class_weight={0:0.055,1:0.085,2:0.074,3:0.061,4:0.087,5:0.119,6:0.083,7:0.357,8:0.172,9:0.333,10:0.227,11:0.063,12:0.313,13:0.833,14:0.313,15:0.125,16:0.149,17:1.0,18:0.294,19:0.455,20:0.357,21:0.203,22:0.267,23:0.333,24:0.398,25:0.455,26:0.833,27:0.232},
#           class_weight={0:0.66,1:0.72,2:0.60,3:1,4:0.77}
          shuffle=True,verbose=2,
          callbacks=callbacks)
    fold_var = fold_var + 1
    keras.backend.clear_session()


############################################################################33
# newpath = '/home/aman/Desktop/aman/RIDD challenge/trial-7 fold b4 28 class/fold 4/log/'    # Path for storing tensorboard file 
# tensorboard = TensorBoard(log_dir = newpath)








