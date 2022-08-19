import glob 
import cv2 
import numpy as np 
import os
name = glob.glob('/mnt/imgproc/Aman/anamoly detection/anomaly detection data/small_fundus/test/good/'+'*')
temp_name = name[0]
temp_name = os.path.basename(temp_name)
temp_name = temp_name.split('.')[0]
temp_name = '121'
patch_path = '/home/amans/data177/to_copy_file/anomaly detection/stpm/patch_data/lightning_logs/version_29/sample/'
orgnl_img = glob.glob(patch_path+'*'+'_'+temp_name+'-'+'*'+'_'+'*'+'o.jpg')
orgnl_img = orgnl_img[0].split('-')[0]
temp_np = np.zeros((1024,1024,3))
# print(np.unique(temp_np))
for i in range(4):
    for j in range(4):
        check = orgnl_img+'-'+str(i)+'_'+str(j)+'_o.jpg'
        img_path = str(check)
        img = cv2.imread(img_path)
        # print(np.unique(img))
        temp_np[i*256:(i+1)*256 , j*256:(j+1)*256 , :] = img
        # print(np.unique(temp_np))
cv2.imwrite('/home/amans/data177/to_copy_file/anomaly detection/stpm/patch_data/lightning_logs/version_29/joined_patch/'+temp_name+'.png',temp_np) 
temp_np = np.zeros((1024,1024,3))
for i in range(4):
    for j in range(4):
        check = orgnl_img+'-'+str(i)+'_'+str(j)+'_amap.jpg'
        img_path = str(check)
        img = cv2.imread(img_path)
        # print(np.unique(img))
        temp_np[i*256:(i+1)*256 , j*256:(j+1)*256 , :] = img
        # print(np.unique(temp_np))
cv2.imwrite('/home/amans/data177/to_copy_file/anomaly detection/stpm/patch_data/lightning_logs/version_29/joined_patch/'+temp_name+'_amap.png',temp_np) 
temp_np = np.zeros((1024,1024,3))
for i in range(4):
    for j in range(4):
        check = orgnl_img+'-'+str(i)+'_'+str(j)+'_amap_on_img.jpg'
        img_path = str(check)
        img = cv2.imread(img_path)
        # print(np.unique(img))
        temp_np[i*256:(i+1)*256 , j*256:(j+1)*256 , :] = img
        # print(np.unique(temp_np))
cv2.imwrite('/home/amans/data177/to_copy_file/anomaly detection/stpm/patch_data/lightning_logs/version_29/joined_patch/'+temp_name+'_amap_on_img.png',temp_np) 
# print(orgnl_img)