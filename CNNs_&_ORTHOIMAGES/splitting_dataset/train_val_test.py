
import os
import shutil
from sklearn.model_selection import train_test_split




#%%
print(os.getcwd())

os.makedirs("data_split", exist_ok=True)

#%%
# Define paths
root_path = "/Users/matteo_crespinjouan/Desktop/NHM/DATASET_BETA"

train_path = os.path.join("data_split", 'train')

val_path = os.path.join("data_split", 'val')
test_path = os.path.join("data_split", 'test')
#%%
# here we create the directories where we'll store the datasets
os.makedirs(train_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

#%% Gather image paths and labels
image_paths = []
labels = []
for class_dir in os.listdir(root_path): #we iterate over the folders in the directory where the data is.
    class_path = os.path.join(root_path, class_dir)
    if os.path.isdir(class_path):
        for img_file in os.listdir(class_path):
            image_paths.append(os.path.join(class_path, img_file))
            labels.append(class_dir)

            ##so here, image_paths contains all the paths of the images
            ##labels is a list of the same size containing the respetive labels (since class dir is a label)

#%%
# now i split dataset into train and test+val first, then split test+val into test and val
train_paths, test_val_paths, train_labels, test_val_labels = train_test_split(
    image_paths, labels, test_size=0.2, stratify=labels)

val_paths, test_paths, val_labels, test_labels = train_test_split(
    test_val_paths, test_val_labels, test_size=0.5, stratify=test_val_labels)

def copy_files(file_paths, file_labels, destination_root):
    for path, label in zip(file_paths, file_labels):
        # Create a directory for the class if it doesn't exist
        destination_dir = os.path.join(destination_root, label)
        os.makedirs(destination_dir, exist_ok=True)
        # Copy file to the respective class directory
        shutil.copy(path, destination_dir)

# copy files to the respective directories (train_paths are the lists of the paths of all the files we want to be


copy_files(train_paths, train_labels, train_path)
copy_files(val_paths, val_labels, val_path)
copy_files(test_paths, test_labels, test_path)

#%%
# checking intersection is null
all_sets = [set(train_paths), set(val_paths), set(test_paths)] #every element is there one single time since sets cant duplicat
disjoint = all(len(set1.intersection(set2)) == 0 for i, set1 in enumerate(all_sets) for set2 in all_sets if i != all_sets.index(set2))
print(disjoint) #true si c disjoint=True)