#%%%
import torch

import os
import numpy as np
from random import sample
try:    from keras.preprocessing.image import load_img
except: from PIL import Image

from utils import data_folder, image_size, val_dict, train_dict, show_image



def get_images(files, size, test):
    os.chdir(data_folder)
    images = []
    if(test): files = ["images/val/{}".format(file) for file in files]
    else:     files = ["images/train/{}/{}".format(file[:9], file) for file in files]
    for file in files:
        try:    images.append(np.array(load_img(file, target_size=(size, size)))/255)
        except: images.append(np.array(Image.open(file).resize((size, size)))/255)
    images = [torch.from_numpy(image).unsqueeze(0) for image in images]
    for i, image in enumerate(images):
        if(image.shape == (1, size, size)): 
            image = image.unsqueeze(-1).repeat((1,1,1,3)) 
            images[i] = image
    return(torch.cat(images, 0))

def get_data(batch_size = 64, size = image_size, test = False):
    if(test):
        image_names = sample(val_dict.keys(), batch_size)
        solutions = torch.cat([val_dict[n] for n in image_names])
    else:
        image_names = sample(train_dict.keys(), batch_size)
        solutions = torch.cat([train_dict[n] for n in image_names])
    images = get_images(image_names, size, test)
    return(images.float(), solutions.float())
        
        
        
if __name__ == "__main__":

    images, solutions = get_data(test = True)

    print()
    print(images.shape)
    print(solutions.shape)
    print()

    images, solutions = get_data(test = False)

    print()
    print(images.shape)
    print(solutions.shape)
    print()
    
    show_image(images[0], solutions[0])
    
# %%
