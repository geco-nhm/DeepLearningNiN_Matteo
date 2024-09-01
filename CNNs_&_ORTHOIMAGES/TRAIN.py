#imports
from torchvision import datasets, transforms
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
import torchvision.models
import torch.nn as nn
import torch.optim
import os
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import torchvision.models as models
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR


# Define paths
root_path = "/Users/matteo_crespinjouan/Desktop/NHM/ORTHO_ENDEAVOUR/32_compare_224"
train_path = os.path.join(root_path, 'train_jpg')
val_path = os.path.join(root_path, 'val_jpg')
#test_path = os.path.join(root_path, 'test') ########LATER WHEN I WANT TO TEST########

# CREATING A DATALOADER:
# Transform images into tensors
transform = v2.Compose([
    #v2.RandomCrop(size=(224, 224)),
    #v2.RandomRotation(degrees=(0, 180)),
    v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    v2.RandomVerticalFlip(p=0.5),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToTensor()

])

# Create dataset objects
train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
#test_dataset = datasets.ImageFolder(root=test_path, transform=transform)########LATER WHEN I WANT TO TEST########
val_dataset = datasets.ImageFolder(root=val_path, transform=transform)

# Create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
#test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)
val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False)

# IMPORT A MODEL
model = torchvision.models.resnet18(pretrained=True)
#model = models.swin_v2_t(weights="IMAGENET1K_V1")  # Load pretrained Swin V2 Tiny model
model_name = type(model).__name__

# Freeze all layers or not 
for param in model.parameters():
    param.requires_grad = True

# tête de classification 
model.fc = nn.Linear(model.fc.in_features, 7)
#model.head = nn.Linear(model.head.in_features, 7)

criterion = nn.CrossEntropyLoss() # Loss function

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
# Define learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

# METRICS

# TRAINING & VALIDATION LOOP
device = torch.device("mps")
model.to(device)

class_names = train_dataset.classes  # For printing the scores later
#%%
# Number of epochs to train for
num_epochs = 20

# Training & Validation Loop
train_losses = []
val_losses = []
macro_f1_scores = []

#%%
for epoch in range(num_epochs):

    # TRAINING LOOP
    model.train()  # Set the model to training mode
    batch_losses = []
    for i, (inputs, labels) in enumerate(train_dataloader):
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())
        
        #on récup le learning rate
        current_lr = current_lr = optimizer.param_groups[0]['lr'] # get_last_lr() returns a list

        print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(train_dataloader)}], Learning Rate: {current_lr}, training loss: {loss.item()}')
    # END OF TRAINING LOOP

    epoch_loss = np.mean(batch_losses)
    train_losses.append(epoch_loss)

    # VALIDATION LOOP & METRICS
    model.eval()  # Model to evaluation mode
    all_probs = []
    all_labels = []
    val_batch_losses = []

    with torch.no_grad():  # No need to keep track of gradients
        for inputs, labels in val_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_loss = criterion(outputs, labels)
            print(f'val loss du batch : {val_loss}')
            val_batch_losses.append(val_loss.item())

            probs = F.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_epoch_loss = np.mean(val_batch_losses[:-1]) #other wise the last can change the mean a lot becasue there can be only one in the batcbh
    val_epoch_loss = np.mean(val_batch_losses)
    scheduler.step(val_epoch_loss) #ici on donne l'info au scheduler pour qu'il décide si il changera le lr ou pas

    val_losses.append(val_epoch_loss)

    probability_scores = np.concatenate(all_probs, axis=0)
    all_labels = np.array(all_labels)

    # Calculate precision and recall
    preds = np.argmax(probability_scores, axis=1)
    precision = precision_score(all_labels, preds, average=None, labels=np.arange(len(class_names)))
    recall = recall_score(all_labels, preds, average=None, labels=np.arange(len(class_names)))
    f1 = f1_score(all_labels, preds, average=None, labels=np.arange(len(class_names)))
    
    macro_f1 = f1_score(all_labels, preds, average='macro')
    # here we keep track of the macro f1
    macro_f1_scores.append(macro_f1)
    
    # here we calulate the moving average of the macro-averaged f1 scores across the past 10 peochs 
    
    window_size = 10
    if len(macro_f1_scores) >= window_size:
        moving_avg_macro_f1 = np.mean(macro_f1_scores[-window_size:])
    else:
        moving_avg_macro_f1 = np.mean(macro_f1_scores)
    
    #now we print all this at each epoch 
    
    
    print(f'Epoch [{epoch + 1}/{num_epochs}] - Validation Loss: {val_epoch_loss}')
    for idx, class_name in enumerate(class_names):
        print(f'{class_name} - Precision: {precision[idx]:.4f}, Recall: {recall[idx]:.4f}, F1 Score: {f1[idx]:.4f}')

    # Print the moving average of macro-averaged F1 score with some emphasis
    print("="*50)
    print(f'MOVING AVERAGE MACRO-AVERAGED F1 SCORE (last {min(window_size, len(macro_f1_scores))} epochs): {moving_avg_macro_f1:.4f}')
    print("="*50)
    
    # Plot confusion matrix
    # Plot confusion matrix
    cm = confusion_matrix(all_labels, preds, labels=np.arange(len(class_names)))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_normalized, annot=cm, fmt='d', cmap='rocket_r', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{model_name} - Epoch {epoch + 1}\nMoving Average Macro-averaged F1 Score: {moving_avg_macro_f1:.4f}')
    plt.show()
#%%
# Plot training and validation losses and macro F1 score on the same plot
plt.figure(figsize=(10, 5))

# Plot training and validation losses
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Losses with Macro F1 Score')

# Create a second y-axis to plot macro F1 score
ax1 = plt.gca()
ax2 = ax1.twinx()
ax2.plot(macro_f1_scores, label='Macro F1 Score', color='green')
ax2.set_ylabel('Macro F1 Score')

# Combine legends from both y-axes
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

plt.show()

