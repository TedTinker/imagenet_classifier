#%%
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # Without this, pyplot crashes the kernal
#data_folder    = r"C:\Users\tedjt\Desktop\imagenet"
#program_folder = r"C:\Users\tedjt\Desktop\imagenet_classifier"
data_folder     = r"/home/ted/Desktop/imagenet"
program_folder  = r"/home/ted/Desktop/imagenet_classifier"

import pandas as pd
import itertools
os.chdir(data_folder)

val_image_names = os.listdir("Data/val")
val_image_names.sort()
val_solutions = pd.read_csv('val_solution.csv', sep=',',header=0)
val_solutions["ImageId"] = val_solutions["ImageId"] + ".JPEG"
val_solutions["PredictionString"] = val_solutions["PredictionString"].str.split()
print("\nVal image names: {}. Val solutions: {}.".format(len(val_image_names), len(val_solutions)))
print("Val image names, but not val solutions: {}.".format(len(list(set(val_image_names) - set(val_solutions["ImageId"])))))
print("Val solutions, but not val image names: {}.".format(len(list(set(val_solutions["ImageId"]) - set(val_image_names)))))
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
print("\nTrain image names: {}. Train solutions: {}.".format(len(train_image_names), len(train_solutions)))
print("Train image names, but not train solutions: {}.".format(len(list(set(train_image_names) - set(train_solutions["ImageId"])))))
print("Train solutions, but not train image names: {}.".format(len(list(set(train_solutions["ImageId"]) - set(train_image_names)))))
train_solutions = train_solutions.iloc[pd.Index(train_solutions['ImageId']).get_indexer(train_image_names)]["PredictionString"].values.tolist()
train_dict = {image : [solution[i:i+5] for i in range(0, len(solution), 5)] for image, solution in zip(
    train_image_names,
    train_solutions)}
# %%
