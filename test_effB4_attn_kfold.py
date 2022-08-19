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
from sklearn.model_selection import KFold, StratifiedKFold
# import sparsenet
import numpy as np
from tqdm import tqdm 
# import sklearn.metrics as metrics
# from losses import binary_focal_loss 
import keras.backend as K
K.set_image_data_format('channels_last')

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
np.random.seed(1337)  # for reproducibility

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.95
set_session(tf.Session(config=config))
 
# Training parameters 
# epochs = 250
data_augmentation = True
num_classes = 4
#  

nrows = 380
ncolumns = 380
channels = 3

# train_dir = '/home/aman/Desktop/aman/RIDD challenge/trial-6 k fold effb4 attn /'
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
def create_model_attn():
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
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])
    return model


def create_model_effb4():
    img_dim = (380, 380, 3)
    base_model = efn.EfficientNetB4(input_shape=img_dim, weights='noisy-student', include_top=False)  # or weights='noisy-student'
 
# from keras_efficientnets import EfficientNetB7 
# base_model = EfficientNetB7(input_shape=img_dim, weights='imagenet', include_top=False) 

    x = base_model.output  
    
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation=swish_activation, name='fc1')(x)
    x = Dropout(0.6)(x)
    x = Dense(64, activation=swish_activation, name='fc2')(x)
    x = Dropout(0.6)(x)
    x = Dense(32, activation=swish_activation, name='fc3')(x)
    x = Dropout(0.6)(x)
    
    predictions = Dense(4, activation='sigmoid', name='predictions')(x)
    # add the top layer block to your base model
    model = Model(base_model.input, predictions)
    opt = adam(lr=1e-4) 
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])
    return model


out_path = '/home/aman/Desktop/aman/dr_analysis/trial-12 5 fold effn b4/'

path2 = sorted(glob.glob('/mnt/X/Ravi K/Fundus_database/HVDEH_grade_wise/DR_Grading/combine/test/grade1/' + '*'))
path3 = sorted(glob.glob('/mnt/X/Ravi K/Fundus_database/HVDEH_grade_wise/DR_Grading/combine/test/grade2/' + '*'))
path4 = sorted(glob.glob('/mnt/X/Ravi K/Fundus_database/HVDEH_grade_wise/DR_Grading/combine/test/grade3/' + '*'))
path5 = sorted(glob.glob('/mnt/X/Ravi K/Fundus_database/HVDEH_grade_wise/DR_Grading/combine/test/grade4/' + '*'))


print('Path2:' + str(len(path2)))
print('Path3:' + str(len(path3)))
print('Path4:' + str(len(path4)))
print('Path5:' + str(len(path5)))

#  
test_images =   path2+ path3  + path4 + path5

resized_coordinates = {'FileName': [],
                       'Prob_0':[],
                       'Prob_1':[],
                       'Prob_2':[],
                       'Prob_3':[],
#                        'Prob_4':[],
                       'pred_class': [],
                       'true_class': []}

# print(np.unique(Y_train))
model_path1 = '/home/aman/Desktop/aman/dr_analysis/trial-12 5 fold effn b4/model/low_loss_fold_1.h5'
model_path2 = '/home/aman/Desktop/aman/dr_analysis/trial-12 5 fold effn b4/model/low_loss_fold_2.h5'
model_path3 = '/home/aman/Desktop/aman/dr_analysis/trial-12 5 fold effn b4/model/low_loss_fold_3.h5'
model_path4 = '/home/aman/Desktop/aman/dr_analysis/trial-12 5 fold effn b4/model/low_loss_fold_4.h5'
model_path5 = '/home/aman/Desktop/aman/dr_analysis/trial-12 5 fold effn b4/model/low_loss_fold_5.h5'
model1 = create_model_effb4()
model2 = create_model_effb4()
model3 = create_model_effb4()
model4 = create_model_effb4()
model5 = create_model_effb4()
model1.load_weights(model_path1)
model2.load_weights(model_path2)
model3.load_weights(model_path3)
model4.load_weights(model_path4)
model5.load_weights(model_path5)


def clahe_single(ori_img,clipLimit,tileGridSize):

    # ori_img = Image.open(pth)
    # bgr = cv2.imread(pth)
    lab = cv2.cvtColor(ori_img, cv2.COLOR_RGB2LAB)
    
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit,tileGridSize)
    lab_planes[0] = clahe.apply(lab_planes[0])
    
    lab = cv2.merge(lab_planes)
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return rgb

def test_model():
    for image in tqdm(test_images):
        resized_coordinates['FileName'].append(image)
        img = cv2.imread(image)
        inp = cv2.resize(img, (380, 380),interpolation=cv2.INTER_NEAREST)
        inp = clahe_single(inp, 2.0 , (8,8))
        MK = []
        MK.append(((inp)))
        Test_image = np.array(MK)
        test_1 = model1.predict(Test_image)
        test_2 = model2.predict(Test_image)
        test_3 = model3.predict(Test_image)
        test_4 = model4.predict(Test_image)
        test_5 = model5.predict(Test_image)
        res_avg = []
        
        for i in range(4):
            prob = ( test_1[0][i]+ test_2[0][i] + test_3[0][i] + test_4[0][i] + test_5[0][i])/5.
            res_avg.append(prob_0)
        res_arr = np.array(res_avg)
        res = np.argmax(res_arr)
        resized_coordinates['Prob_0'].append(res_arr[0])
        resized_coordinates['Prob_1'].append(res_arr[1])
        resized_coordinates['Prob_2'].append(res_arr[2])
        resized_coordinates['Prob_3'].append(res_arr[3])
        opimage = image.split('/')
        imgLabel = opimage[-2]
        if  'grade0' in imgLabel:
    #         val = 0
                print('check1')
            #print('Glaucoma')
        elif 'grade1' in imgLabel:
            val = 0
        elif 'grade2' in imgLabel:
            val = 1
        elif 'grade3' in imgLabel:
            val = 2
        elif 'grade4' in imgLabel:
            val = 3
    
        #resized_coordinates['DME_Grade'].append("{0:0.4f}".format(pred[0][0]))
        resized_coordinates['pred_class'].append(res)
        resized_coordinates['true_class'].append(val)
    coords_df = pd.DataFrame(data=resized_coordinates)
    coords_df.to_csv(out_path + 'dr jay reviewd low loss 5 fold without 1.csv')

test_model()

        
