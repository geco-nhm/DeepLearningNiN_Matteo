#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 15:30:36 2024

@author: matteo_crespinjouan
"""

import os
import shutil
from sklearn.model_selection import train_test_split

#%%
print(os.getcwd())

os.makedirs("/Users/matteo_crespinjouan/Desktop/NHM/ORTHO_ENDEAVOUR/32_split", exist_ok=True)

#%%
# Define paths
root_path = "/Users/matteo_crespinjouan/Desktop/NHM/ORTHO_ENDEAVOUR/32_tiffs"

train_path = os.path.join("/Users/matteo_crespinjouan/Desktop/NHM/ORTHO_ENDEAVOUR/32_split", 'train')
val_path = os.path.join("/Users/matteo_crespinjouan/Desktop/NHM/ORTHO_ENDEAVOUR/32_split", 'val')

#%%
# Here we create the directories where we'll store the datasets
os.makedirs(train_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)

#%% Gather image paths and labels
image_paths = []
labels = []
for class_dir in os.listdir(root_path): # We iterate over the folders in the directory where the data is.
    class_path = os.path.join(root_path, class_dir)
    if os.path.isdir(class_path):
        for img_file in os.listdir(class_path):
            image_paths.append(os.path.join(class_path, img_file))
            labels.append(class_dir)

#%%
# Now we split dataset into train and validation sets
train_paths, val_paths, train_labels, val_labels = train_test_split(
    image_paths, labels, test_size=0.2, stratify=labels)

def copy_files(file_paths, file_labels, destination_root):
    for path, label in zip(file_paths, file_labels):
        # Create a directory for the class if it doesn't exist
        destination_dir = os.path.join(destination_root, label)
        os.makedirs(destination_dir, exist_ok=True)
        # Copy file to the respective class directory
        shutil.copy(path, destination_dir)

# Copy files to the respective directories
copy_files(train_paths, train_labels, train_path)
copy_files(val_paths, val_labels, val_path)

#%%
# Checking intersection is null
all_sets = [set(train_paths), set(val_paths)] # Every element is there one single time since sets can't duplicate
disjoint = all(len(set1.intersection(set2)) == 0 for i, set1 in enumerate(all_sets) for set2 in all_sets if i != all_sets.index(set2))
print(disjoint) # True if disjoint