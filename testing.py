#%%
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # Without this, pyplot crashes the kernal
#data_folder    = r"C:\Users\tedjt\Desktop\imagenet"
#program_folder = r"C:\Users\tedjt\Desktop\imagenet_classifier"
from utils import data_folder, program_folder

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

missed_solutions = list(set(train_image_names) - set(train_solutions["ImageId"]))
total_classes = [i[:9] for i in train_image_names]
classes = [i[:9] for i in missed_solutions]
print("\nClasses in missed solutions: {}.".format(len(list(set(classes)))))
for c in list(set(classes)):
    print("\tClass {}: missing {} out of {}.".format(c, classes.count(c), total_classes.count(c)))
    

# %%
