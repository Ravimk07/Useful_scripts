import glob
import numpy as np
import cv2
import os
from cProfile import label

#Define Size of images
imwidth = 512
imheight = 512
imdepth = 3
  
#Number Of classes in labels
classes = 2
data_shape = imwidth*imheight

#Path to read Tiles for Label and Data
data_path = '/mnt/x/Ravi K/Glaucoma/aman/lense_degene/data/augment/val/images/'
label_path ='/mnt/x/Ravi K/Glaucoma/aman/lense_degene/data/augment/val/Label/'

#Function to create label array for binary classification
def binarylab(labels):
    
    #Define an Empty Array 
    x = np.zeros([imheight,imwidth,classes],dtype="uint8")
    
    #Read Each pixel label and put it into corresponding label plane
    for i in range(imheight):
        for j in range(imwidth):
            x[i,j,labels[i][j]]=1
    
    return x

def prepareDataSet():
        
    labelpaths = sorted(glob.glob(label_path+"/*_modlabel.png"))
    
    #Create Empty Lists to store Image and Label Data
    data = []
    label = []
     
  
        
    for i in range(len(labelpaths)):
        #for i in range(50):    
        tlp = labelpaths[i]
#         tilepath = tlp.split("/")
        tilepath = os.path.basename(tlp)
        tileno = tilepath.split("_modlabel.png")
        tdp = data_path+tileno[0]+'.png'
        
        #Read Images
        
        im = cv2.imread(tdp)
        if im is not None:
            im_  = im.astype("float32")
            
            #Compute the mean for data normalization
            b_ch=np.mean(im[:,:,0])
            g_ch=np.mean(im[:,:,1])
            r_ch=np.mean(im[:,:,2])  
            # Mean substraction     
            im_ = np.array(im, dtype=np.float32)                             
            im_ -= np.array((b_ch,g_ch,r_ch))
            
            #compute the standard deviation
            b_ch=np.std(im[:,:,0])
            g_ch=np.std(im[:,:,1])
            r_ch=np.std(im[:,:,2])
            im_ /= np.array((b_ch,g_ch,r_ch))
                     
            lab = cv2.imread(tlp)[:,:,0]
            
            #Append Images into corresponding List
            data.append(np.rollaxis((im_),2))
                        
            #Convert label into binary form
            lab = binarylab(lab)
            
            #Append Images into corresponding List
            label.append(((lab)))
            
            print('\n'+tdp)
        else:
            print("error: "+tdp)
     
#     print(mean_b,mean_g,mean_r)  
    return data,label   
 
        
        
data,label = prepareDataSet()

#Store Data to the directory
np.save(data_path+'Val_data.npy',np.array(data))
np.save(label_path+'Val_label.npy',np.array(label))

print ("Done")
