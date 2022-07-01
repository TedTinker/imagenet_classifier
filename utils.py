#%%
data_folder = r"/home/ted/Desktop/imagenet"
program_folder = r"/home/ted/Desktop/imagenet_classifier"

import torch 
import torch.nn.functional as F 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if(device.type == "cpu"):   print("\n\nAAAAAAAUGH! CPU! >:C\n")
else:                       print("\n\nUsing CUDA! :D\n")

import os
import pandas as pd
import itertools

image_size = 64

os.chdir(data_folder)
train_folders = os.listdir("images/train")
train_image_names = []
for folder in train_folders:
    train_image_names.append(os.listdir("images/train/" + folder))
train_image_names = list(itertools.chain.from_iterable(train_image_names))
train_image_names.sort()
train_solutions = [n[:9] for n in train_image_names]
train_dict = {image : solution for image, solution in zip(train_image_names, train_solutions)}

val_image_names = os.listdir("images/val")
val_image_names.sort()
val_solutions = pd.read_csv('val_solution.csv', sep=',',header=0)
val_solutions["ImageId"] = val_solutions["ImageId"] + ".JPEG"
val_solutions["PredictionString"] = val_solutions["PredictionString"].str.split().str[0]
val_solutions = val_solutions.iloc[pd.Index(val_solutions['ImageId']).get_indexer(val_image_names)]["PredictionString"].values.tolist()
val_dict = {image : solution for image, solution in zip(
    val_image_names,
    val_solutions)}

mapping = pd.read_csv('mapping.txt', sep='\n',header=None)
mapping = {c[:9] : c[10:].split(',')[0] for c in mapping[0]}
all_solutions = list(train_dict.values()) + list(val_dict.values())
all_solutions = list(set(all_solutions))
all_solutions.sort()
hot_solutions = {s : F.one_hot(torch.tensor([i]), num_classes = len(all_solutions)) \
                               for i, s in enumerate(all_solutions)}
inverted = {v : k for k, v in hot_solutions.items()}
hot_mapping = {torch.argmax(h).item() : mapping[inverted[h]] for h in inverted.keys()}

train_dict = {k : hot_solutions[v] for k,v in train_dict.items()}
val_dict = {k : hot_solutions[v] for k,v in val_dict.items()}

if __name__ == "__main__":
    print("Training image names: \t{}. \tTraining solutions: \t{}.".format(len(train_dict.keys()), len(train_dict.values())))
    print("Val image names: \t{}. \t\tVal solutions: \t\t{}.".format(len(val_dict.keys()), len(val_dict.values())))
    print("Solutions: {}.".format(len(all_solutions)))
    
    
    
def init_weights(m):
    try:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)
    except: pass



import matplotlib.pyplot as plt
def show_image(image, solution):
    plt.imshow(image)
    plt.title(hot_mapping[torch.argmax(solution).item()])
    plt.axis('off')
    plt.show()
    plt.close()
    
def plot_trials(e, images, solutions, predictions):
    titles = []
    for i in range(len(images)):
        pred_dict = {hot_mapping[j] : 100*torch.exp(predictions[i,j]).item() for j in range(len(all_solutions))}
        pred_sorted = [(k, v) for k, v in pred_dict.items()]
        pred_sorted.sort(key=lambda p: p[1], reverse=True)
        titles.append("Answer:\n{} ({}%).\n\nGuesses:\n{} ({}%),\n{} ({}%),\n{} ({}%).".format(
            hot_mapping[torch.argmax(solutions[i]).item()], 
            round(pred_dict[hot_mapping[torch.argmax(solutions[i]).item()]],3),
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
