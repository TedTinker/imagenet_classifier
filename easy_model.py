#%%
import torch
from torch import nn 
from torch.optim import Adam
from torchinfo import summary as torch_summary

from utils import device, image_size, init_weights
from train_test import train

class Easy(nn.Module):
    def __init__(self):
        
        super(Easy, self).__init__()
        
        self.lin = nn.Sequential(
            nn.Linear(
                in_features = image_size*image_size*3, 
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
        images = images.flatten(1)
        output = self.lin(images)
        return(output.cpu())
    
easy = Easy()

print(easy)
print()
print(torch_summary(easy, (10, image_size, image_size, 3)))

train(easy)
# %%
