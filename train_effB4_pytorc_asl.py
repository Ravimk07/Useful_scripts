import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from losses import AsymmetricLoss
# from torchsummary import summary 
from torchinfo import summary   
from torch.utils.data import SubsetRandomSampler,ConcatDataset
from sklearn.model_selection import KFold
from pathlib import Path

class ImageDataset(Dataset):
    def __init__(self, csv, train, test):
        self.csv = csv
        self.train = train
        self.test = test
        self.all_image_names = self.csv[:]['ID']
        self.all_labels = np.array(self.csv.drop(['ID','Disease_Risk'], axis=1))
        self.train_ratio = int(0.85 * len(self.csv))
        self.valid_ratio = len(self.csv) - self.train_ratio
        # set the training data images and labels
        if self.train == True:
            print(f"Number of training images: {self.train_ratio}")
            self.image_names = list(self.all_image_names[:self.train_ratio])
            self.labels = list(self.all_labels[:self.train_ratio])
            # define the training transforms
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((1024, 1024)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=45),
                transforms.ToTensor(),
            ])
        # set the validation data images and labels
        elif self.train == False and self.test == False:
            print(f"Number of validation images: {self.valid_ratio}")
            self.image_names = list(self.all_image_names[-self.valid_ratio:-10])
            self.labels = list(self.all_labels[-self.valid_ratio:])
            # define the validation transforms
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
            ])
        # set the test data images and labels, only last 10 images
        # this, we will use in a separate inference script
        elif self.test == True and self.train == False:
            self.image_names = list(self.all_image_names[-10:])
            self.labels = list(self.all_labels[-10:])
             # define the test transforms
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ])
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, index):
        image = cv2.imread(f"/mnt/imgproc/Ravi K/RIDD/New_data/crop1/{self.image_names[index]}.png")
        # convert the image from BGR to RGB color format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # apply image transforms
        image = self.transform(image)
        targets = self.labels[index]
        
        return {
            'image': torch.tensor(image, dtype=torch.float32),
            'label': torch.tensor(targets, dtype=torch.float32)
        }
        
        
from torchvision import models as models
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
# import timm 
def stpm_classification(pretrained, requires_grad):
    # model = EfficientNet.from_name('efficientnet-b4', num_classes=28)
    model = models.resnet50(progress=True, pretrained='/mnt/imgproc/Aman/anamoly detection/anomaly detection data/small_fundus/test/small_fundus/lightning_logs/version_10/checkpoints/epoch=564-step=19209.ckpt')
    # model = timm.create_model('inception_resnet_v2', pretrained=pretrained, num_classes=28)
    # model = EfficientNet.from_pretrained('efficientnet-b1', num_classes=28)    models.resnext50_32x4d(pretrained=True)
    # model = models.resnet50(progress=True, pretrained=True)
    # model = models.resnext50_32x4d(progress=True, pretrained=True)
    print(model.layer1[-1])
    print("##########################")
    print(model.layer2[-1])
    print("##########################") 
    print(model.layer3[-1])
    # to freeze the hidden layers 
    if requires_grad == False:
        for param in model.parameters():
            param.requires_grad = False
    # to train the hidden layers
    elif requires_grad == True:
        for param in model.parameters():
            param.requires_grad = True
    # make the classification layer learnable
    # we have 25 classes in total
    model.fc = nn.Linear(2048, 28)
    return model   


import torch
from tqdm import tqdm
# training function
def train(model, dataloader, optimizer, criterion, train_data, device):
    print('Training')
    model.train()
    counter = 0
    train_running_loss = 0.0
    for i, data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
        counter += 1
        data, target = data['image'].to(device), data['label'].to(device)
        optimizer.zero_grad()
        outputs = model(data)
        # apply sigmoid activation to get all the outputs between 0 and 1
        outputs = torch.sigmoid(outputs)
        loss = criterion(outputs, target)
        train_running_loss += loss.item()
        # backpropagation
        loss.backward()
        # update optimizer parameters
        optimizer.step()
        
    train_loss = train_running_loss / counter
    return train_loss




# validation function
def validate(model, dataloader, criterion, val_data, device):
    print('Validating')
    model.eval()
    counter = 0
    val_running_loss = 0.0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(val_data)/dataloader.batch_size)):
            counter += 1
            data, target = data['image'].to(device), data['label'].to(device)
            outputs = model(data)
            # apply sigmoid activation to get all the outputs between 0 and 1
            outputs = torch.sigmoid(outputs)
            loss = criterion(outputs, target)
            val_running_loss += loss.item()
        
        val_loss = val_running_loss / counter
        return val_loss
    
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib 
from torch.utils.data import DataLoader
matplotlib.style.use('ggplot')
# initialize the computation device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        

# read the training csv file
train_csv = pd.read_csv('/mnt/imgproc/Ravi K/RIDD/New_data/RFMiD_Training_Labels_new.csv')


# train dataset
train_data = ImageDataset(
    train_csv, train=True, test=False
)
# validation dataset
valid_data = ImageDataset(
    train_csv, train=False, test=False
)
dataset = ConcatDataset([train_data,valid_data])
# train data loader
# train_loader = DataLoader(
#     train_data, 
#     batch_size=batch_size,
#     shuffle=True
# )
# # validation data loader
# valid_loader = DataLoader(
#     valid_data, 
#     batch_size=batch_size,
#     shuffle=False
# )


# start the training and validation
train_loss = []
valid_loss = []
splits = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(dataset)))):
    print('Fold {}'.format(fold+1))
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

#intialize the model

    model = stpm_classification(pretrained=True, requires_grad=True).to(device)

    # learning parameters
    lr = 0.0001 
    epochs = 10
    batch_size = 4
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # criterion = nn.BCELoss()  
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    # criterion = nn.CrossEntropyLoss().cuda('2') 
    valid_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler) 


    for epoch in range(epochs):
        print(f"Epoch {epoch+1} of {epochs}")
        train_epoch_loss = train(
            model, train_loader, optimizer, criterion, train_data, device
        )
        valid_epoch_loss = validate(
            model, valid_loader, criterion, valid_data, device
        )
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        print(f"Train Loss: {train_epoch_loss:.4f}")
        print(f'Val Loss: {valid_epoch_loss:.4f}')
        
        if epoch %3 ==0:
            print('saving the model')
            # save the trained model to disk
            torch.save({
                        'epoch': epochs,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': criterion,}, 
                Path('/mnt/imgproc/Ravi K/RIDD/New_data/pytorch/model_resnet_stpm/resnet_stpm_ssl_epoch_'+str(epoch)+'_fold_'+str(fold+1)+'.pth'))
    print('saving the model after the epoch')
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,},
                Path('/mnt/imgproc/Ravi K/RIDD/New_data/pytorch/model_resnet_stpm/resnet_stpm_ssl_fold_'+str(fold+1)+'.pth'))

    # plot and save the train and validation line graphs
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.plot(valid_loss, color='red', label='validataion loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(Path('/mnt/imgproc/Ravi K/RIDD/New_data/pytorch/model_resnet_stpm/resnet_stpm_ssl_fold_'+str(fold+1)+'.png'))
    plt.show()



    
        