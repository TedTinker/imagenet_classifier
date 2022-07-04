#%%
from tqdm import tqdm
import pandas as pd
import itertools


import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # Without this, pyplot crashes the kernal
#data_folder    = r"C:\Users\tedjt\Desktop\imagenet"
#program_folder = r"C:\Users\tedjt\Desktop\imagenet_classifier"
data_folder     = r"/home/ted/Desktop/imagenet"
program_folder  = r"/home/ted/Desktop/imagenet_classifier"

import torch 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if(device.type == "cpu"):   print("\n\nAAAAAAAUGH! CPU! >:C\n")
else:                       print("\n\nUsing CUDA! :D\n")



image_size = 64
max_classes = 20



os.chdir(data_folder)
mapping = pd.read_csv('mapping.txt', sep='\0',header=None)
mapping = {c[:9] : c[10:].split(',')[0] for c in mapping[0]}
all_solutions = list(mapping.keys())
all_solutions.sort()
all_solutions = {s : i for i, s in enumerate(all_solutions)}
inverted    = {v : k for k, v in all_solutions.items()}
mapping = {h : mapping[inverted[h]] for h in inverted.keys()}
    
    

try:
    print("Loading dictionaries...")
    train_dict = torch.load("train_dict.pt")
    val_dict   = torch.load("val_dict.pt")
except:
    print("Can't load dictionaries. Making and saving dictionaries...")
    
    val_image_names = os.listdir("Data/val")
    val_image_names.sort()
    val_solutions = pd.read_csv('val_solution.csv', sep=',',header=0)
    val_solutions["ImageId"] = val_solutions["ImageId"] + ".JPEG"
    val_solutions["PredictionString"] = val_solutions["PredictionString"].str.split()
    print("\nVal image names: {}. Val solutions: {}.".format(len(val_image_names), len(val_solutions)))
    val_solutions = val_solutions.iloc[pd.Index(val_solutions['ImageId']).get_indexer(val_image_names)]["PredictionString"].values.tolist()
    val_dict = {image : [solution[i:i+5] for i in range(0, len(solution), 5)] for image, solution in zip(
        val_image_names,
        val_solutions)}
        
    train_image_names = []
    for folder in os.listdir("Data/train"):
        train_image_names.append(os.listdir("Data/train/" + folder))
    train_image_names = list(itertools.chain.from_iterable(train_image_names))
    train_image_names.sort()
    train_solutions = pd.read_csv('train_solution.csv', sep=',',header=0)
    train_solutions["ImageId"] = train_solutions["ImageId"] + ".JPEG"
    train_solutions["PredictionString"] = train_solutions["PredictionString"].str.split()
    print("Train image names: {}. Train solutions: {}.".format(len(train_image_names), len(train_solutions)))
    train_solutions = train_solutions.iloc[pd.Index(train_solutions['ImageId']).get_indexer(train_image_names)]["PredictionString"].values.tolist()
    train_dict = {image : [solution[i:i+5] for i in range(0, len(solution), 5)] for image, solution in zip(
        train_image_names,
        train_solutions)}

    print("\nVal image names: {}. Val solutions: {}.".format(len(val_image_names), len(val_solutions)))
    print("Train image names: {}. Train solutions: {}.".format(len(train_image_names), len(train_solutions)))
    
    def finalize_solutions(k, v, test):
        hots = []
        for s in v:
            hots.append([
                all_solutions[s[0]], 
                [int(s[1]), int(s[2]), int(s[3]), int(s[4])]])
        return(hots)
    
    for k, v in tqdm(train_dict.items()):
        train_dict[k] = finalize_solutions(k, v, False)
    for k, v in tqdm(val_dict.items()):
        val_dict[k]   = finalize_solutions(k, v, True)
    
    os.chdir(data_folder)
    torch.save(train_dict, "train_dict.pt")
    torch.save(val_dict,   "val_dict.pt")
os.chdir(program_folder)

if __name__ == "__main__":
    print("\nVal image names: \t{}. \t\tVal solutions: \t\t{}.".format(len(val_dict.keys()), len(val_dict.values())))
    print("Training image names: \t{}. \tTraining solutions: \t{}.".format(len(train_dict.keys()), len(train_dict.values())))
    print("Solutions: {}.".format(len(all_solutions)))
    
    
    
def init_weights(m):
    try:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)
    except: pass



import matplotlib.pyplot as plt
def show_image(image_name, image, classifications, positions):
    plt.imshow(image)
    title = image_name + ":\n"
    for i, (c, p) in enumerate(zip(classifications, positions)):
        p[0] = image_size * p[0]
        p[1] = image_size * p[1]
        p[2] = image_size * p[2]
        p[3] = image_size * p[3]
        title += mapping[c.item()] + ": "
        title += ", ".join([str(round(p_)) for p_ in p.tolist()]) + "."
        if(i < len(positions)):
            title += "\n"
        box = [p_ for p_ in p.tolist()]
        x = [box[1], box[2]]
        y = [box[1], box[3]]
        x = [x[0], x[0], x[1], x[1], x[0]]
        y = [y[0], y[1], y[1], y[0], y[0]]
        plt.plot(x, y, color = "yellow")
    plt.title(title)
    plt.axis('off')
    plt.show()
    plt.close()
    
import numpy as np
try:    from keras.preprocessing.image import load_img
except: from PIL import Image
    
def show_this_image(image_name, test):
    os.chdir(data_folder)
    if(test): file = "Data/val/{}".format(image_name)
    else:     file = "Data/train/{}/{}".format(image_name[:9], image_name)
    if(test): solutions = val_dict[image_name]
    else:     solutions = train_dict[image_name]
    classifications = [torch.tensor([solution[0]]) for solution in solutions]
    positions = torch.tensor([solution[1] for solution in solutions]).float() 
    try:    image = np.array(load_img(file))/255
    except: image = np.array(Image.open(file))/255
    plt.imshow(image)
    title = image_name + ":\n"
    for i, (c, p) in enumerate(zip(classifications, positions)):
        title += mapping[c.item()] + ": "
        title += ", ".join([str(round(p_)) for p_ in p.tolist()]) + "."
        if(i < len(positions)):
            title += "\n"
        box = [p_ for p_ in p.tolist()]
        x = [box[1], box[2]]
        y = [box[1], box[3]]
        x = [x[0], x[0], x[1], x[1], x[0]]
        y = [y[0], y[1], y[1], y[0], y[0]]
        plt.plot(x, y, color = "yellow")
    plt.title(title)
    plt.axis('off')
    plt.show()
    plt.close()
    
def plot_trials(e, images, solutions, predictions):
    titles = []
    for i in range(len(images)):
        pred_dict = {mapping[j] : 100*torch.exp(predictions[i,j]).item() for j in range(len(all_solutions))}
        pred_sorted = [(k, v) for k, v in pred_dict.items()]
        pred_sorted.sort(key=lambda p: p[1], reverse=True)
        titles.append("Answer:\n{} ({}%).\n\nGuesses:\n{} ({}%),\n{} ({}%),\n{} ({}%).".format(
            mapping[torch.argmax(solutions[i]).item()], 
            round(pred_dict[mapping[torch.argmax(solutions[i]).item()]],3),
            pred_sorted[0][0], round(pred_sorted[0][1],3), 
            pred_sorted[1][0], round(pred_sorted[1][1],3), 
            pred_sorted[2][0], round(pred_sorted[2][1],3)))
    
    fig = plt.figure(figsize=(20,30))
    fig.suptitle("\nExamples from Epoch {}\n".format(str(e)), fontsize = 20)
    for i in range(1, int(len(images))+1):
        fig.add_subplot(int(len(images)**.5), int(len(images)**.5), i)
        plt.imshow(images[i-1].squeeze(-1))
        plt.title(titles[i-1])
        plt.axis("off")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    plt.close()
    
def plot_losses(e, train_losses, test_losses):
    plt.plot(train_losses, color = "blue", label = "Train")
    plt.plot(test_losses, color = "red", label = "Test")
    plt.title("Losses on Epoch " + str(e))
    plt.legend(loc = 'upper left')
    plt.show()
    plt.close()
# %%
