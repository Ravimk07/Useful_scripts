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
from tqdm import tqdm 
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
#  
# Training parameters 
epochs = 500
data_augmentation = True
num_classes = 28
# 

nrows = 600
ncolumns = 600
channels = 3

# train_dir = '/mnt/X/Aman/Opthalmology/cropped and resized data/trial 1 - Gauss Blur/model/'
     
path1 = sorted(glob.glob('/mnt/X/Ravi K/RIDD/New_data/crop1/' + '*' ))   

print('Path:' + str(len(path1)))
file = pd.read_csv('/mnt/X/Ravi K/RIDD/New_data/RFMiD_Training_Labels_new.csv')
#  
# train_images = path1 + path3 + path4 + path5 + path6 + path7  + path9 + path10 + path11 + path12 #+ path13 + path15  + path17 + path18 + path19 + path20
train_images =   path1
print(len(train_images))
# # train_images = train_mitosis + train_nonmitosis
random.shuffle(train_images)
# # del train_AMD
# # del train_non_AMD
gc.collect()
# 
############################Clahe ###########################################################################

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


#####################################################################################################################################
def make_label(df):
    label = []
    for i in range(28):
        label.append(df.iloc[0][i])
    return label

def read_and_process_image(list_of_images): 
                    
    X = [] # images
    y_label  = [] # labels
    y_final = []
    
    for image in tqdm(list_of_images):
          
        inp = cv2.imread(image, cv2.IMREAD_COLOR)
        inp = cv2.resize(inp, (600, 600),interpolation=cv2.INTER_NEAREST)
        inp = clahe_single(inp, 2.0 , (8,8))
#         img1 = inp/255
        img1 = (inp - inp.mean(axis=(0,1))) / (inp.std(axis=(0,1)))
#         cv2.imwrite(save_path+opimage[-1], img1)
#         input = imnormalization_globallmean(input)
        X.append(((img1)))
        

        name = os.path.basename(image)
        name = name.split('.')[0]
#         print(name)
        reqd_file = file[file['ID']==int(name)]
        y_one = reqd_file['Disease_Risk'].iloc[0]
        reqd_file.drop(['ID','Disease_Risk'],axis=1,inplace = True)
        
#         temp = make_label(reqd_file)
        
#         print(temp)
#         y_label.append(temp)
        y_final.append(y_one)
    
        # get the labels
                           
    return X, y_final
         
         
# ##############
# 
X, y = read_and_process_image(train_images) 
     
#  
X_train = np.array(X)
y_train = np.array(y)
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=2)
print('check___')   
print(type(X_train))
# #  
np.save('/mnt/X/Ravi K/RIDD/New_data/Train_data2.npy', X_train)
np.save('/mnt/X/Ravi K/RIDD/New_data/Train_label2.npy', y_train)
# #     
# np.save('/mnt/X/Ravi K/RIDD/New_data/Train_data2.npy', X_train)
# np.save('/mnt/X/Ravi K/RIDD/New_data/Train_label2.npy', y_train)
# # #       
# np.save('/mnt/X/Ravi K/RIDD/New_data/Val_data2.npy', X_val)
# np.save('/mnt/X/Ravi K/RIDD/New_data/Val_label2.npy', y_val)
print('check_3')
