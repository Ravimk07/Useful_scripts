import os
import numpy as np
import torch
import cv2
from PIL import Image
import albumentations as A
from skimage import color
import random
np.random.seed(37)

def stain_agumentation(img,theta=0.02):
    th = 0.9
    alpha = np.random.uniform(1 - theta, 1 + theta, (1, 3))
    beta = np.random.uniform(-theta, theta, (1, 3))
    img = np.array(img)
    gray_img = color.rgb2gray((img))
    background = (gray_img > th)  # * (gray_img-self.th)/(1-self.th)
    background = background[:, :, np.newaxis]

    s = color.rgb2hed(img)
    ns = alpha * s + beta  # perturbations on HED color space
    nimg = color.hed2rgb(ns)

    imin = nimg.min()
    imax = nimg.max()
    rsimg = ((nimg - imin) / (imax - imin))  # rescale
    rsimg = (1 - background) * rsimg + background * img / 255

    rsimg = (255 * rsimg).astype('uint8')
    return rsimg

aug = A.Compose([
    # A.OneOf([
        # A.RandomSizedCrop(min_max_height=(50, 101), height=original_height, width=original_width, p=0.5),
        # A.PadIfNeeded(min_height=original_height, min_width=original_width, p=0.5)
    # ],p=1),
    A.VerticalFlip(p=0.5),
    A.HorizontalFlip(p=0.5),#p=1
    A.RandomRotate90(p=0.5),
    # A.OneOf([
        # A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
        # # A.GridDistortion(p=0.5),
        # # A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1)
        # ], p=0.8),
    A.CLAHE(p=0.8),
    A.RandomBrightnessContrast(p=0.2),
    A.HueSaturationValue(p=0.8),#p=1
    A.RandomGamma(p=0.8)])


def data_mean_normalization(im):
    im_  = im.astype("float32")
    #Individual channel-wise mean substraction
    im_ -= np.array((0.485, 0.456, 0.406))
    #Individual channel-wise standard deviation division
    im_ /= np.array((0.229, 0.224, 0.225))
    return im_

def sampleMeanStdExcludeWhite(img):


    img  = img.astype("float32")

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    x, y = np.where(imgGray <240)

    b_ch_mean=np.mean(img[x,y,0])
    g_ch_mean=np.mean(img[x,y,1])
    r_ch_mean=np.mean(img[x,y,2])

    b_ch_std=np.std(img[x,y,0])
    g_ch_std=np.std(img[x,y,1])
    r_ch_std=np.std(img[x,y,2])

    img[:, :, 0] = (img[:, :, 0] - b_ch_mean)/b_ch_std
    img[:, :, 1] = (img[:, :, 1] - g_ch_mean)/g_ch_std
    img[:, :, 2] = (img[:, :, 2] - r_ch_mean)/r_ch_std

    return img

def local_mean_normalization(im):
    #=====================Local_mean====================================
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

    return im_

class DataGenerator(torch.utils.data.Dataset):
    'Generates data for pytorch'
    def __init__(self, root, split, batch_size=32, dim=(32,32), n_channels=3,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.root = root
        self.split = split
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle


        self.image_dir = os.path.join(self.root, self.split, self.split + '_data_all')
        self.label_dir = os.path.join(self.root, self.split, self.split + '_labels_all')

        file_list = os.path.join(self.root, self.split, self.split + '_segmentation', self.split + ".txt")
        self.files = [line.rstrip() for line in tuple(open(file_list, "r"))]
        self.indexes = np.arange(len(self.files))
        #self.on_epoch_end()

    def __len__(self):
        return len(self.files)

    def __getitem__(self,
                    index: int ):
        # Select the sample
        indexes = self.indexes[index]
        list_IDs_temp = self.files[indexes]

        image_path = os.path.join(self.image_dir, list_IDs_temp + '.png')
        label_path = os.path.join(self.label_dir, list_IDs_temp+ '.png')
        image = cv2.imread(image_path)
        label = cv2.imread(label_path,0)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #label = np.asarray(Image.open(label_path), dtype=np.int)
        # label = cv2.imread(label_path)
        # apply augmentations
        augmented = aug(image=image, mask=label)
        image = augmented['image']

        rdm = random.randint(1, 3)
        if rdm == 1:
            image = stain_agumentation(image)
        label = augmented['mask']
        image= image/255
        image = data_mean_normalization(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image= image.astype(np.float32)
        label = label.astype(np.long)
        image = np.rollaxis(image,-1,0)
        #store class
        label = binarylabel(label, self.n_classes)
        label= np.rollaxis(label, -1,0)
        #label = np.reshape(label, (self.dim[0] * self.dim[1], self.n_classes))

        # Typecasting
        x,y =  torch.tensor(image), torch.tensor(label)
        return x, y


def binarylabel(im_label,classes):

#     print(im_label.shape)

    im_dims = im_label.shape

    lab=np.zeros([im_dims[0],im_dims[1],classes],dtype="uint8")
    for class_index in range(classes):

        lab[im_label==class_index, class_index] = 1

    return lab


class ValDataGenerator(torch.utils.data.Dataset):
    'Generates data for pytorch'

    def __init__(self, root, split, batch_size=32, dim=(32, 32), n_channels=3,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.root = root
        self.split = split
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle

        self.image_dir = os.path.join(self.root, self.split, self.split + '_data_all')
        self.label_dir = os.path.join(self.root, self.split, self.split + '_labels_all')

        file_list = os.path.join(self.root, self.split, self.split + '_segmentation', self.split + ".txt")
        self.files = [line.rstrip() for line in tuple(open(file_list, "r"))]
        self.indexes = np.arange(len(self.files))
        # self.on_epoch_end()

    def __len__(self):
        return len(self.files)

    def __getitem__(self,
                    index: int):
        # Select the sample
        indexes = self.indexes[index]
        list_IDs_temp = self.files[indexes]

        image_path = os.path.join(self.image_dir, list_IDs_temp + '.png')
        label_path = os.path.join(self.label_dir, list_IDs_temp + '.png')
        image = cv2.imread(image_path)
        label = cv2.imread(label_path, 0)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        image = image / 255
        image = data_mean_normalization(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = image.astype(np.float32)
        label = label.astype(np.long)
        image = np.rollaxis(image, -1, 0)

        # store class
        label = binarylabel(label, self.n_classes)
        label = np.rollaxis(label, -1, 0)
        #label = np.reshape(label, (self.dim[0] * self.dim[1], self.n_classes))

        # Typecasting

        x, y = torch.tensor(image), torch.tensor(label)
        return x, y

