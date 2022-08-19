import torch 
from torchvision import models as models
import torch.nn as nn
import numpy as np 
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader 

from torch.utils.data import Dataset 

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
            self.image_names = list(self.all_image_names[0:640])
            self.labels = list(self.all_labels[0:640])
             # define the test transforms
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
            ])
    def __len__(self): 
        return len(self.image_names)
    
    def __getitem__(self, index):
        image = cv2.imread(f"/mnt/imgproc/Aman/ridd/data/test_dataset/{self.image_names[index]}.jpg")
        # convert the image from BGR to RGB color format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # apply image transforms
        image = self.transform(image)
        targets = self.labels[index]
        
        return {
            'image': torch.tensor(image, dtype=torch.float32),
            'label': torch.tensor(targets, dtype=torch.float32)
        }
 
# initialize the computation device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
from efficientnet_pytorch import EfficientNet

def stpm_classification(pretrained, requires_grad):
    # model = EfficientNet.from_name('efficientnet-b4', num_classes=28)
    model = models.resnet50(progress=True, pretrained='/mnt/imgproc/Aman/anamoly detection/anomaly detection data/small_fundus/test/small_fundus/lightning_logs/version_10/checkpoints/epoch=564-step=19209.ckpt')
    
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

#intialize the model
model = stpm_classification(pretrained=True, requires_grad=True).to(device)
# model = models.model(pretrained=False, requires_grad=False).to(device)
# model = EfficientNet.from_name('efficientnet-b4', num_classes=28)
# load the model checkpoint
# model.load_state_dict(torch.load('/mnt/X/Ravi K/RIDD/New_data/pytorch/model_effi/effib4.pth'))
checkpoint = torch.load('/mnt/imgproc/Ravi K/RIDD/New_data/pytorch/model_resnet_stpm/resnet_stpm_ssl_epoch_6_fold_1.pth')

# load model weights state_dict
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
out_path = '/mnt/imgproc/Ravi K/RIDD/New_data/pytorch/resnet_sptm/' 

train_csv = pd.read_csv('/mnt/imgproc/Aman/ridd/data/RFMiD_Testing_Labels.csv')
genres = train_csv.columns.values[2:]
# prepare the test dataset and dataloader
test_data = ImageDataset(
    train_csv, train=False, test=True
)
test_loader = DataLoader(
    test_data, 
    batch_size=1,
    shuffle=False
)

resized_coordinates = {'ID': [],'Disease_Risk':[],'DR':[],'ARMD':[],'MH':[],'DN':[],'MYA':[],'BRVO':[],'TSLN':[],
                       'ERM':[],'LS':[],'MS':[],'CSR':[],'ODC':[],
                       'CRVO':[],'TV':[],'AH':[],'ODP':[],'ODE':[],
                       'ST':[],'AION':[],'PT':[],'RT':[],'RS':[],
                       'CRS':[],'EDN':[],'RPEC':[],'MHL':[],'RP':[],
                       'OTHER':[]}
num_classes = 28

for counter, data in enumerate(test_loader):
    image, target = data['image'].to(device), data['label']
    # get all the index positions where value == 1
    target_indices = [i for i in range(len(target[0])) if target[0][i] == 1]
    # get the predictions by passing the image through the model
    outputs = model(image)
    # imageid = target_indices[0] 
    outputs = torch.sigmoid(outputs)
    pred = outputs.detach().cpu()
    pred = pred.detach().numpy()

    resized_coordinates['ID'].append(str(counter+1))
    print(str(counter+1))
#
    resized_coordinates['Disease_Risk'].append(0)
    resized_coordinates['DR'].append(pred[0][0])
    resized_coordinates['ARMD'].append(pred[0][1])
    resized_coordinates['MH'].append(pred[0][2])
    resized_coordinates['DN'].append(pred[0][3])
    resized_coordinates['MYA'].append(pred[0][4])
    resized_coordinates['BRVO'].append(pred[0][5])
    resized_coordinates['TSLN'].append(pred[0][6])
    resized_coordinates['ERM'].append(pred[0][7])
    resized_coordinates['LS'].append(pred[0][8])
    resized_coordinates['MS'].append(pred[0][9])
    resized_coordinates['CSR'].append(pred[0][10])
    resized_coordinates['ODC'].append(pred[0][11])
    resized_coordinates['CRVO'].append(pred[0][12])
    resized_coordinates['TV'].append(pred[0][13])
    resized_coordinates['AH'].append(pred[0][14])
    resized_coordinates['ODP'].append(pred[0][15])
    resized_coordinates['ODE'].append(pred[0][16])
    resized_coordinates['ST'].append(pred[0][17])
    resized_coordinates['AION'].append(pred[0][18])
    resized_coordinates['PT'].append(pred[0][19])
    resized_coordinates['RT'].append(pred[0][20])
    resized_coordinates['RS'].append(pred[0][21])
    resized_coordinates['CRS'].append(pred[0][22])
    resized_coordinates['EDN'].append(pred[0][23])
    resized_coordinates['RPEC'].append(pred[0][24])
    resized_coordinates['MHL'].append(pred[0][25])
    resized_coordinates['RP'].append(pred[0][26])
    resized_coordinates['OTHER'].append(pred[0][27])
#
coords_df = pd.DataFrame(data=resized_coordinates,columns=['ID','Disease_risk','DR','ARMD','MH','DN','MYA','BRVO','TSLN','ERM','LS','MS','CSR','ODC','CRVO','TV','AH','ODP','ODE','ST','AION','PT','RT','RS','CRS','EDN','RPEC','MHL','RP','OTHER'])
coords_df.to_csv(out_path + 'resnet_sptm_asl.csv')
    
    
    
    
    
    
    








#
# for counter, data in enumerate(test_loader):
#     image, target = data['image'].to(device), data['label']
#     # get all the index positions where value == 1
#     target_indices = [i for i in range(len(target[0])) if target[0][i] == 1]
#     # get the predictions by passing the image through the model
#     outputs = model(image)
#     outputs = torch.sigmoid(outputs)
#     outputs = outputs.detach().cpu()
#     sorted_indices = np.argsort(outputs[0])
#     best = sorted_indices[-3:]
#     string_predicted = ''
#     string_actual = ''
#     for i in range(len(best)):
#         string_predicted += f"{genres[best[i]]}    "
#     for i in range(len(target_indices)):
#         string_actual += f"{genres[target_indices[i]]}    "
#     image = image.squeeze(0)
#     image = image.detach().cpu().numpy()
#     image = np.transpose(image, (1, 2, 0))
#     plt.imshow(image)
#     plt.axis('off')
#     plt.title(f"PREDICTED: {string_predicted}\nACTUAL: {string_actual}")
#     plt.savefig(f"/mnt/X/Aman/ridd/data/inference/inference_{counter}.jpg")
#     plt.show()
    
    