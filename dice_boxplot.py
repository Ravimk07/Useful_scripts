
import numpy as np
import os
from tensorflow import keras
from PIL import Image
import glob
import cv2
import pandas as pd 
from matplotlib import pyplot as plt
import xlrd 
from matplotlib.patches import Polygon

# Z:\Ravi K\Other\ISBI_2019_challenge\iChallenge_AMD\Task_2\data\New\Augmented\resized\Fovea_detection
# original_path = 'Z:/Ravi K/Other/ISBI_2019_challenge/iChallenge_AMD/Task_2/data/New/Augmented/resized/Fovea_detection/'

# data_path = 'Z:/Ravi K/Other/ISBI_2019_challenge/iChallenge_AMD/Task_2/data/New/Augmented/resized/Fovea_detection/val_images/'
# save_path = 'Z:/Ravi K/Other/ISBI_2019_challenge/iChallenge_AMD/Task_2/data/New/Augmented/resized/Fovea_detection/val_images/'

data_path = 'Z:/Ravi K/Other/ISBI_2019_challenge/ADAM Validation-400-images/Task_2/final/'
save_path = 'Z:/Ravi K/Other/ISBI_2019_challenge/ADAM Validation-400-images/Task_2/final/'

# Z:\Ravi K\Other\ISBI_2019_challenge\ADAM Validation-400-images\Task_2\final

# data_path = 'Z:/Ravi K/Other/ISBI_2019_challenge/iChallenge_AMD/Task_2/data/'
# save_path = 'Z:/Ravi K/Other/ISBI_2019_challenge/iChallenge_AMD/Task_2/data/'

pre_loc = data_path + 'fovea_preds.xlsx'
 
wb = xlrd.open_workbook(pre_loc) 
sheet = wb.sheet_by_index(0)  
sheet.cell_value(0, 0) 


fx1 = []
fx2 = []
for i in range(sheet.nrows): 
    fx = sheet.cell_value(i, 1)
    fy = sheet.cell_value(i, 2)
    fx1 = fx, fy
    fx1 = list(fx1)
    fx2.append(fx1)

print(fx2)  
  
# Give the location of original  file 
Actu_loc = data_path + 'Fovea_true.xlsx'

wb = xlrd.open_workbook(Actu_loc) 
sheet = wb.sheet_by_index(0)  
sheet.cell_value(0, 0) 

fxactual = [] 
fxactual2 = []
for i in range(sheet.nrows): 
    fx = sheet.cell_value(i, 1)
    fy = sheet.cell_value(i, 2)
    fxactual = fx, fy
    fxactual = list(fxactual)
    fxactual2.append(fxactual)

print(fxactual2)  


xp = [i[0] for i in fx2]
yp = [i[1] for i in fx2]
  
xo = [i[0] for i in fxactual2]
yo = [i[1] for i in fxactual2]

# np.sqrt(  np.sum((xo - xp) ** 2) + np.sum((yo - yp) ** 2))
xo = np.array(xo)
xp = np.array(xp)
yo = np.array(yo)
yp = np.array(yp)

# ED = np.sqrt(np.sum((xo - xp) ** 2) + np.sum((yo - yp) ** 2))    
# print(ED)   

pt1 = fx2
pt2 = fxactual2

# ED = np.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)

ED = np.sqrt((xo - xp) ** 2 + (yo - yp) ** 2)
print(ED)

# import seaborn as sns

# fig, axs = plt.subplots(2, 2)
# axs[0, 0].boxplot(ED)
# axs[0, 0].set_title('')  
# axs.set_ylabel('Euclidean Distance in pixels') 
# plt.show()
fig = plt.figure()
# fig.suptitle('bold figure suptitle', fontsize=14, fontweight='bold')
ax = fig.add_subplot(111)
ax.boxplot(ED)

ax.set_title('')
ax.set_xlabel('')
ax.set_ylabel('Euclidean distance (in pixels)')
plt.show()
# import numpy as np
# from scipy.spatial.distance import pdist, squareform
# 
# # Compute the Euclidean distance between all rows of x.
# # d[i, j] is the Euclidean distance between x[i, :] and x[j, :],
# # and d is the following array:
# # [[ 0.          1.41421356  2.23606798]
# #  [ 1.41421356  0.          1.        ]
# #  [ 2.23606798  1.          0.        ]]
# d = squareform(pdist(fx2, 'euclidean'))
# print(d)