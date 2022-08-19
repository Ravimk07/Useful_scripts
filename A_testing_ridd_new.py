'''
Created on 12-Nov-2019

@author: Ravi
'''
import os
import tensorflow as tf
import glob
from keras.optimizers import adam
from keras.backend.tensorflow_backend import set_session
import numpy as np
import pandas as pd
import cv2 
# from losses import binary_focal_loss  
import random
import gc
from keras.callbacks import TensorBoard
import keras
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import sgd
from keras.applications import resnet50
from matplotlib import pyplot as plt 
from keras.layers import (Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D,
                          BatchNormalization, Input, Conv2D, GlobalAveragePooling2D,concatenate,Concatenate,multiply, LocallyConnected2D, Lambda)
import keras.backend as K
K.set_image_data_format('channels_last')

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
np.random.seed(1337)  # for reproducibility

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.95
# set_session(tf.Session(config=config))
from tqdm import tqdm

path = '/mnt/X/Ravi K/RIDD/Evaluation_Set/processed3/'

out_path = '/mnt/X/Ravi K/RIDD/New_data/model/'

# modelPath = '/mnt/x/Ravi K/Glaucoma/refuge_2/training/task1/patches/augment/clahe/train/model/effic_B7_noisy_softma_out2.033-0.270422-0.943820.h5'    # .9189
modelPath = '/mnt/X/Ravi K/RIDD/New_data/model/low_loss_fold_3.h5'    # 
# modelPath = '/mnt/X/Ravi K/RIDD/New_data/model/low_loss_fold_2.h5'    # 
# modelPath = '/mnt/X/Ravi K/RIDD/New_data/model/low_loss_fold_3.h5'    # 
# modelPath = '/mnt/X/Ravi K/RIDD/New_data/model/low_loss_fold_4.h5'    # 
# modelPath = '/mnt/X/Ravi K/RIDD/New_data/model/low_loss_fold_5.h5'    # 

imheight = 512
imwidth = 512
imdepth = 3
data_shape = imheight * imwidth
classes = 28
data_format='.png'

resized_coordinates = {'ID': [],'Disease_risk':[],'DR':[],'ARMD':[],'MH':[],'DN':[],'MYA':[],'BRVO':[],'TSLN':[],
                       'ERM':[],'LS':[],'MS':[],'CSR':[],'ODC':[],
                       'CRVO':[],'TV':[],'AH':[],'ODP':[],'ODE':[],
                       'ST':[],'AION':[],'PT':[],'RT':[],'RS':[],
                       'CRS':[],'EDN':[],'RPEC':[],'MHL':[],'RP':[],
                       'OTHER':[]}
num_classes = 28

img_dim = (imwidth,imheight,imdepth)
def swish_activation(x): 
    return x * K.sigmoid(x)

import efficientnet.keras as efn 

base_model = efn.EfficientNetB7(input_shape=img_dim, weights='noisy-student', include_top=False)  # or weights='noisy-student'
x = base_model.output  

x = GlobalAveragePooling2D()(x)
x = Dense(512, activation=swish_activation, name='fc1')(x)
x = Dropout(0.6)(x)
x = Dense(256, activation=swish_activation, name='fc2')(x)
x = Dropout(0.6)(x)
x = Dense(128, activation=swish_activation, name='fc3')(x)
x = Dropout(0.6)(x)

predictions = Dense(28, activation='sigmoid', name='predictions')(x)
model = Model(base_model.input, predictions)
opt = adam(lr=1e-4) 
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])
print(model.summary())
####################################################################
 
model.load_weights(modelPath)
preds = []
fns = []
Y_pred = []
Y_prob = []


X = [] # images
y = [] # labels
clas_prob=[]
cls_lbl = []
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
print('ready to predict')
for i in tqdm(range(1,641)):
    MK = []   
    image = os.path.join(path, str(i))
    image = image + '.png'
#     img = cv2.imread(filename)
    img = cv2.imread(image, cv2.IMREAD_COLOR)
    inp = cv2.resize(img, (512, 512),interpolation=cv2.INTER_NEAREST)
    inp = clahe_single(inp, 2.0 , (8,8))
#     opimage = image.split('/')
#     imgLabel = opimage[-1]
#     imgLabel1 = imgLabel[0:3]
#     print(imgLabel)
    
    resized_coordinates['ID'].append(str(i))
    
#     img1 = inp/255
    img1 = (inp - inp.mean(axis=(0,1))) / (inp.std(axis=(0,1)))

    MK.append(((img1)))
    Test_image = np.array(MK)
    
    pred = model.predict(Test_image) 

    clas_prob.append(pred[0][0])
#     clas_prob.append((1-pred[0][0]))
    
    resized_coordinates['Disease_risk'].append(0)
    resized_coordinates['DR'].append(1 if pred[0][0]>0.5 else 0)
    resized_coordinates['ARMD'].append(1 if pred[0][1]>0.5 else 0)
    resized_coordinates['MH'].append(1 if pred[0][2]>0.5 else 0)
    resized_coordinates['DN'].append(1 if pred[0][3]>0.5 else 0)
    resized_coordinates['MYA'].append(1 if pred[0][4]>0.5 else 0)
    resized_coordinates['BRVO'].append(1 if pred[0][5]>0.5 else 0)
    resized_coordinates['TSLN'].append(1 if pred[0][6]>0.5 else 0)
    resized_coordinates['ERM'].append(1 if pred[0][7]>0.5 else 0)
    resized_coordinates['LS'].append(1 if pred[0][8]>0.5 else 0)
    resized_coordinates['MS'].append(1 if pred[0][9]>0.5 else 0)
    resized_coordinates['CSR'].append(1 if pred[0][10]>0.5 else 0)
    resized_coordinates['ODC'].append(1 if pred[0][11]>0.5 else 0)
    resized_coordinates['CRVO'].append(1 if pred[0][12]>0.5 else 0)
    resized_coordinates['TV'].append(1 if pred[0][13]>0.5 else 0)
    resized_coordinates['AH'].append(1 if pred[0][14]>0.5 else 0)
    resized_coordinates['ODP'].append(1 if pred[0][15]>0.5 else 0)
    resized_coordinates['ODE'].append(1 if pred[0][16]>0.5 else 0)
    resized_coordinates['ST'].append(1 if pred[0][17]>0.5 else 0)
    resized_coordinates['AION'].append(1 if pred[0][18]>0.5 else 0)
    resized_coordinates['PT'].append(1 if pred[0][19]>0.5 else 0)
    resized_coordinates['RT'].append(1 if pred[0][20]>0.5 else 0)
    resized_coordinates['RS'].append(1 if pred[0][21]>0.5 else 0)
    resized_coordinates['CRS'].append(1 if pred[0][22]>0.5 else 0)
    resized_coordinates['EDN'].append(1 if pred[0][23]>0.5 else 0)
    resized_coordinates['RPEC'].append(1 if pred[0][24]>0.5 else 0)
    resized_coordinates['MHL'].append(1 if pred[0][25]>0.5 else 0)
    resized_coordinates['RP'].append(1 if pred[0][26]>0.5 else 0)
    resized_coordinates['OTHER'].append(1 if pred[0][27]>0.5 else 0)

coords_df = pd.DataFrame(data=resized_coordinates,columns=['ID','Disease_risk','DR','ARMD','MH','DN','MYA','BRVO','TSLN','ERM','LS','MS','CSR','ODC','CRVO','TV','AH','ODP','ODE','ST','AION','PT','RT','RS','CRS','EDN','RPEC','MHL','RP','OTHER'])
coords_df.to_csv(out_path + 'RIDD_lowloss_fold3_testdata.csv')
# coords_df.to_csv(out_path + 'RIDD_lowloss_fold2.csv')
# coords_df.to_csv(out_path + 'RIDD_lowloss_fold3.csv')
# coords_df.to_csv(out_path + 'RIDD_lowloss_fold4.csv')
# coords_df.to_csv(out_path + 'RIDD_lowloss_fold5.csv')
 

