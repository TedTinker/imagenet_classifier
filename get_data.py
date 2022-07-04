#%%%
import torch

import os
import numpy as np
try:    from keras.preprocessing.image import load_img
except: from PIL import Image
import imagesize
from random import sample

from utils import data_folder, program_folder, image_size, val_dict, train_dict, show_image



def get_images(files, test):
    os.chdir(data_folder)
    if(test): files = ["Data/val/{}".format(file) for file in files]
    else:     files = ["Data/train/{}/{}".format(file[:9], file) for file in files]
    images = []
    image_sizes = []
    for file in files:
        image_sizes.append(imagesize.get(file))
        try:    images.append(np.array(load_img(file, target_size=(image_size, image_size)))/255)
        except: images.append(np.array(Image.open(file).resize((image_size, image_size)))/255)
    images = [torch.from_numpy(image).unsqueeze(0) for image in images]
    for i, image in enumerate(images):
        if(image.shape == (1, image_size, image_size)): 
            image = image.unsqueeze(-1).repeat((1,1,1,3)) 
            images[i] = image
    os.chdir(program_folder)
    return(torch.cat(images, 0).float(), image_sizes)

def get_data(batch_size = 64, size = image_size, test = False):
    if(test):
        image_names = sample(list(val_dict.keys()), batch_size)
        solutions = [val_dict[n] for n in image_names]
        classifications = [torch.tensor([s[0] for s in solution]) for solution in solutions]
        positions = [torch.tensor([s[1] for s in solution]).float() for solution in solutions]
    else:
        image_names = sample(list(train_dict.keys()), batch_size)
        solutions = [train_dict[n] for n in image_names]
        classifications = [torch.tensor([s[0] for s in solution]) for solution in solutions]
        positions = [torch.tensor([s[1] for s in solution]).float() for solution in solutions]
    images, image_sizes = get_images(image_names, test)
    for p, s in zip(positions, image_sizes):
        p[:,0] = 2*(p[:,0]/s[0]) - 1
        p[:,1] = 2*(p[:,1]/s[1]) - 1
        p[:,2] = 2*(p[:,2]/s[0]) - 1
        p[:,3] = 2*(p[:,3]/s[1]) - 1
    return(images, classifications, positions)
        
        
        
if __name__ == "__main__":

    print("\n\nVAL:\n\n")
    images, classifications, positions = get_data(test = True)
    for i in range(10):
        show_image(images[i], classifications[i], positions[i])

    print("\n\nTRAIN:\n\n")
    images, classifications, positions = get_data(test = False)
    for i in range(10):
        show_image(images[i], classifications[i], positions[i])
    
# %%
