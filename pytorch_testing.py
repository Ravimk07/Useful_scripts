import torch
import segmentation_models_pytorch as smp
import cv2
import os
import numpy as np
#from transformations import normalize_01, re_normalize
#from sklearn.externals._pilutil import bytescale


def data_mean_normalization(im):
    im_  = im.astype("float32")
    #Individual channel-wise mean substraction
    im_ -= np.array((0.485, 0.456, 0.406))
    #Individual channel-wise standard deviation division
    im_ /= np.array((0.229, 0.224, 0.225))
    return im_

def preprocess(img: np.ndarray):
    #img = np.moveaxis(img, -1, 0)  # from [H, W, C] to [C, H, W]

    img = np.expand_dims(img, axis=0)  # add batch dimension [B, C, H, W]
    img = img.astype(np.float32)  # typecasting to float32
    return img

def postprocess(img: torch.tensor, grlvl):
    img = torch.argmax(img, dim=1)  # perform argmax to generate 1 channel
    img = img.cpu().numpy()  # send to cpu and transform to numpy.ndarray
    img = np.squeeze(img)  # remove batch dim and channel dim -> [H, W]
    #img = img*255 #re_normalize(img)  # scale it to the range [0-255]
    img=np.uint8(grlvl * (img.astype('float32')))
    #img=bytescale(img, low=0, high=255)
    return img

def predict(img,
            model,
            preprocess,
            postprocess,
            device,
            grlvl,
            ):
    model.eval()
    model.to(device)
    img = preprocess(img)  # preprocess image
    x = torch.from_numpy(img).to(device)  # to torch, send to device
    with torch.no_grad():
        out = model(x)  # send through model/network

    out_softmax = torch.softmax(out, dim=1)  # perform softmax on outputs
    result = postprocess(out_softmax, grlvl)  # postprocess outputs

    return result
# device
device=2
modelpath = '/home/adarshk/Desktop/Kidney/mineralization/artifacts_upd/effib3_3/bestmodel430.pth'
data_path = '/mnt/imgproc/Adarsh/Kidney/Mineralization/training_data/mineraliazation_with_black_artefacts_upd/train/train_data_all/'
path2write= '/mnt/imgproc/Adarsh/Kidney/Mineralization/training_data/mineraliazation_with_black_artefacts_upd/train/b3out_lr3/'
new_classes= 2

if not os.path.exists(path2write):
    os.makedirs(path2write)

# model
model = smp.Unet(
        encoder_name='timm-efficientnet-b3',  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7 or timm-efficientnet-b4
        encoder_weights='noisy-student',  # use `imagenet` or 'noisy-student' pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=new_classes,  # model output channels (number of classes in your dataset)
    )

#model_weights = torch.load(modelpath)
model_weights=torch.load(modelpath, map_location='cuda:' + str(device))
model.load_state_dict(model_weights)


all_files = [f for f in os.listdir(data_path)]
from tqdm import tqdm
for filename in tqdm(all_files):
    fname = filename
    OIm = image = cv2.imread(os.path.join(data_path, fname))
      # normalize_01(img)  # linear scaling to range [0-1]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255
    image = data_mean_normalization(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = image.astype(np.float32)
    image = np.rollaxis(image, -1, 0)
    grlvl= 255/(new_classes-1)
    out_img = predict(image, model, preprocess, postprocess, device, grlvl)

    cv2.imwrite(path2write  + '/' + fname[0:-4] + '_modlabel.png', out_img)

print("END")
