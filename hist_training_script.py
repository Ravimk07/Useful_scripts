
import os
import tensorflow as tf
from keras.optimizers import adam
from keras.layers import Input
from keras import callbacks 
from keras.backend.tensorflow_backend import set_session
import numpy as np
from keras.callbacks import LearningRateScheduler

from sklearn.model_selection import train_test_split

from HistNet import histNet_v1,histNet_v2

from os.path import exists#, join
# from TrainValTensorBoardH2 import TrainValTensorBoard


import keras.backend as K
K.set_image_data_format('channels_last')


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
np.random.seed(1337)  # for reproducibility



config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.95
set_session(tf.Session(config=config))

# tensorboard make dire
log_dir = '/mnt/X/Ravi K/Fundus_database/HVDEH_grade_wise/Glaucoma_Grading/segmentation_crop/od_segmentation/od_patches/augment/log/'

if not exists(log_dir):
    os.makedirs(log_dir)



# Define Directories to read numpy arrays
#===============================================================================

train_data_path = '/mnt/X/Ravi K/Fundus_database/HVDEH_grade_wise/Glaucoma_Grading/segmentation_crop/od_segmentation/od_patches/augment/'
train_label_path= train_data_path
# Define Model Path
#===============================================================================
path2write = '/mnt/X/Ravi K/Fundus_database/HVDEH_grade_wise/Glaucoma_Grading/segmentation_crop/od_segmentation/od_patches/augment/model/'
if not exists(path2write):
    os.makedirs(path2write)

preTrained ='/mnt/X/Pranab_Samanta/colonoscopy/numpyfile/hist_Modified_v2_modified_reverse_breastCancer/hist_Modified_v2-507-0.448270-0.90-0.91.h5'

# Define the Number of classes of the trainable dataset

classes=4
#2. new_classes is also applicable for the testing the pretrained model; where new_classes would be equal to the number of classes
new_classes=4
depth=3
height=512
width=512

model= histNet_v2(width, height, depth, classes, new_classes, data_format='channels_last', weightsPath=None)
print(model.summary())
#===============================================================================
#===============================================================================
# Split the training data into train and validation sets
# train_data = np.load(train_data_path + 'data.npy')
# train_label = np.load(train_label_path + 'label.npy')
# print(train_data.shape)
# print(train_label.shape)
data_shape = height * width
# # 
# train_label = np.reshape(train_label, (len(train_label), data_shape, classes))

# X_train, X_test, y_train, y_test = train_test_split(train_data, train_label, test_size=0.15, random_state=10)
  
# print(X_train.shape)
# print(X_test.shape)  
# # save the train and validation sets for further use (retrained purpose) 
# np.save(path2write+'Train_data',X_train)
# np.save(path2write+'Val_data',X_test)
# np.save(path2write+'Train_label',y_train)
# np.save(path2write+'Val_label',y_test) 
# print(X_train.shape)
# print(y_train.shape) 
#===============================================================================

train_data = np.load(train_data_path + 'Train_data.npy')
train_label = np.load(train_label_path + 'Train_label.npy')
# Reshape Label array
train_label = np.reshape(train_label, (len(train_label), data_shape, classes))

Val_data = np.load(train_data_path + 'Val_data.npy')
Val_label = np.load(train_label_path + 'Val_label.npy')
# Reshape Label array
Val_label = np.reshape(Val_label, (len(Val_label), data_shape, classes))
 
#===============================================================================

       
def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 200:
        lr *= 1e-1
    elif epoch > 700:
        lr *= 1e-2
    elif epoch > 1000:
        lr *= 1e-3
    elif epoch > 2000:
        lr *= 1e-4
    elif epoch > 3000:
        lr *= 1e-5
    print('Learning rate: ', lr)
    return lr

# Store the network weights whenever loss is minimum than previous epoch
filepath=path2write+'hist_v2_glaucma-{epoch:02d}-{val_loss:02f}-{val_dice_coef:.2f}-{val_acc:.2f}.h5'
modelCheck = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only= False, mode='auto', period=5)
opt = adam(lr=lr_schedule(0))

def dice_coef(y_true, y_pred):
    return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)

print ("Compiling Model...")

# Set the compiler parameter for the training
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=[dice_coef,"accuracy"], sample_weight_mode='auto')

lr_scheduler = LearningRateScheduler(lr_schedule)
# Train the Network

#===============================================================================
# X_train, X_test, y_train, y_test = train_test_split(train_data, train_label, test_size=0.10, random_state=10)
tensorboard = callbacks.TensorBoard(log_dir=log_dir,
                                  histogram_freq=0,
                                  batch_size=8,
                                  write_graph=True,
                                  write_grads=False,
                                  write_images=False,
                                  embeddings_freq=0,
                                  embeddings_layer_names=None,
                                  embeddings_metadata=None)
print ("Training the Model...")
# unn_2= model.fit(train_data, train_label, batch_size=8, epochs=4000, verbose=2, callbacks=[TrainValTensorBoard(write_graph=True),modelCheck,lr_scheduler],validation_data=(val_data,val_label))
unn_2= model.fit(train_data, train_label, batch_size=4, epochs=1800, verbose=2, callbacks=[modelCheck,tensorboard],validation_data=(Val_data,Val_label))

print ("Dumping Weights to file...")
path2save1 = path2write + 'glaucoma.h5'
# After training for given number of epochs save the model weights on the disk
model.save_weights(path2save1, overwrite=True)
print ("Models Saved :-)")
