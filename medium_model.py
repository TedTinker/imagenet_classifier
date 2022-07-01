#%%
import torch
from torch import nn 
from torch.optim import Adam
from torchinfo import summary as torch_summary

from utils import device, image_size, init_weights
from train_test import train

class Medium(nn.Module):
    def __init__(self):
        
        super(Medium, self).__init__()
                
        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels = 3,
                out_channels= 64,
                kernel_size = (3,3),
                stride = 1,
                padding = (1,1),
                padding_mode = "reflect"),
            nn.PReLU(),
            nn.MaxPool2d(
                kernel_size = (3,3), 
                stride = (2,2),
                padding = (1,1)),
            nn.Conv2d(
                in_channels = 64,
                out_channels= 64,
                kernel_size = (3,3),
                stride = 1,
                padding = (1,1),
                padding_mode = "reflect"),
            nn.PReLU(),
            nn.MaxPool2d(
                kernel_size = (3,3), 
                stride = (2,2),
                padding = (1,1)),
            nn.Conv2d(
                in_channels = 64,
                out_channels= 64,
                kernel_size = (3,3),
                stride = 1,
                padding = (1,1),
                padding_mode = "reflect"),
            nn.PReLU(),
            nn.MaxPool2d(
                kernel_size = (3,3), 
                stride = (2,2),
                padding = (1,1)))
        
        example = torch.zeros((1, 3, image_size, image_size))
        example = self.cnn(example).flatten(1)
        quantity = example.shape[-1]
        
        self.lin = nn.Sequential(
            nn.Linear(
                in_features = quantity, 
                out_features = 1028),
            nn.PReLU(),
            nn.Linear(
                in_features = 1028, 
                out_features = 1000),
            nn.LogSoftmax(1))
                
        self.lin.apply(init_weights)
        self.to(device)
        self.opt = Adam(self.parameters())
        
    def forward(self, images):
        images = images.to(device)
        #images = images*2 - 1
        images = images.permute(0,-1,1,2)
        images = self.cnn(images)
        images = images.flatten(1)
        output = self.lin(images)
        return(output.cpu())
    
medium = Medium()

print(medium)
print()
print(torch_summary(medium, (10, image_size, image_size, 3)))

train(medium)
# %%
