"""
updated on 27th May,2021
@author: adarshK
"""

import sys
import numpy as np
import os
from torchsummary import summary
from dataloader_new import DataGenerator, ValDataGenerator
import segmentation_models_pytorch as smp
import torch
from trainer_pytorch import Trainer
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler, SequentialSampler
from visual import plot_training
from loss_function_for_segmentation import FocalTverskyLoss, DiceLoss, soft_dice_cldice

gpu_id= 0
savepath = '/home/adarshk/Desktop/Kidney/mineralization/artifacts_upd/effib3_3/'
log_dir = savepath
root_folder = '/home/adarshk/Desktop/Kidney/mineralization/artifacts_upd/data/'
classes=2 # current dataset num_classes

if not os.path.exists(savepath):
    os.makedirs(savepath)

#Define the Image Size
batchsize = 12
depth=3
height=512
width=512

preTrained = None#'/home/adarsh/Desktop/codes/Keras_training_augmentation_general/pretrained/efficientnet-b4_noisy-student_notop.h5' #'/home/adarsh/Desktop/cyst/iter3_1024/model/histnet_v1-515-0.524869-0.92-0.98.h5'
new_classes=classes

model = smp.Unet(
        encoder_name='timm-efficientnet-b3',  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7 or timm-efficientnet-b4
        encoder_weights='noisy-student',  # use `imagenet` or 'noisy-student' pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=new_classes,  # model output channels (number of classes in your dataset)
    )

print(f'\n{model}\n')
filepath=savepath+'bestmodel.h5'

params = {'dim': (height, width),
          'batch_size': batchsize,
          'n_classes': classes,
          'n_channels': depth,
          'shuffle': True}

# Generators
training_generator = DataGenerator(root_folder, 'train',  **params)
validation_generator = ValDataGenerator(root_folder, 'val',  **params)

training_generator = torch.utils.data.DataLoader(training_generator, batch_size=batchsize,  num_workers=6)
validation_generator = torch.utils.data.DataLoader(validation_generator, batch_size=batchsize,  num_workers=6)


print ("Compiling Model...")


# Set the compiler parameter for the training
# loss_fn= torch.nn.CrossEntropyLoss()
# loss_fn=  DiceLoss() #soft_dice_cldice()
loss_fn= FocalTverskyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,80,150], gamma=0.1)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 300], gamma=0.1)

trainer = Trainer(model=model,
                  device=gpu_id,
                  criterion=loss_fn,
                  optimizer=optimizer,
                  training_DataLoader=training_generator,
                  validation_DataLoader=validation_generator,
                  lr_scheduler=scheduler,
                  epochs=1000,
                  epoch=0,
                  notebook=True,
                  path2write= savepath)


print ("Training the Model...")
training_losses, validation_losses, lr_rates = trainer.run_trainer()
'''fig= plot_training(training_losses, validation_losses, lr_rates, gaussian= True, sigma=1, figsize= (10,4) )
fig_name= 'graph.png'
torch.save(fig, os.path.join(savepath,fig_name))'''
model_name =  'carvana_model.pth'
torch.save(model.state_dict(), os.path.join(savepath,model_name))
