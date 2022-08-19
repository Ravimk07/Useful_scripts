import cv2 
import numpy as np 
import glob 
import os 
from tqdm import tqdm




def make_patch(img,size=256, name=None, resize=True, resize_size=1024,save_loc = None ):
    if resize:
        img = cv2.resize(img, (resize_size, resize_size),interpolation=cv2.INTER_NEAREST)
    for i in range(int(resize_size/size)):
        for j in range(int(resize_size/size)):
            temp_img = img[size*i:size*(i+1),size*j:size*(j+1),:]
            cv2.imwrite(save_loc+'/'+name+'-'+str(i)+'_'+str(j)+'.jpg',temp_img)
data_paths = '/mnt/imgproc/Aman/anamoly detection/anomaly detection data/small_fundus/test/good/' 
img_paths = sorted(glob.glob(data_paths+'*'))
save_path = '/mnt/imgproc/Aman/anamoly detection/anomaly detection data/patch_data/test/good/'
for image in tqdm(img_paths):
    img = cv2.imread(image)
    img_name = os.path.basename(image)
    img_name = img_name.split('.')[0]
    make_patch(img,size = 256, name=img_name,resize=True,resize_size=1024,save_loc= save_path)