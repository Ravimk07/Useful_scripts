import torch
import torch.nn as nn
import torch.nn.functional as F

class vgg16(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.n_channels= n_channels
        self.n_classes= n_classes

