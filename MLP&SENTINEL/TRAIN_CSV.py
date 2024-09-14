#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 16:10:36 2024

@author: matteo_crespinjouan
"""


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

# File paths
# train_csv = '/Users/matteo_crespinjouan/Desktop/NHM/SENTINEL_MLP_ENDEAVOUR/SENTINEL_FINAL/MLP_final/hovedtyper_train.csv'
# validation_csv = '/Users/matteo_crespinjouan/Desktop/NHM/SENTINEL_MLP_ENDEAVOUR/SENTINEL_FINAL/MLP_final/hovedtyper_val.csv'
train_csv = '/Users/matteo_crespinjouan/Desktop/NHM/SENTINEL_MLP_ENDEAVOUR/SENTINEL_FINAL/MLP_FINAL_all_gruntyper/gruntyper_train.csv'
validation_csv = '/Users/matteo_crespinjouan/Desktop/NHM/SENTINEL_MLP_ENDEAVOUR/SENTINEL_FINAL/MLP_FINAL_all_gruntyper/gruntyper_val.csv'
# Load data
train_df = pd.read_csv(train_csv)
val_df = pd.read_csv(validation_csv)

# Extract features and labels
X_train = train_df.drop(columns=['class']).values
y_train = train_df['class'].values
X_val = val_df.drop(columns=['class']).values
y_val = val_df['class'].values

# Encode labels to integers
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_val = label_encoder.transform(y_val)

# Normalize the input data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)

# Create DataLoader
batch_size = 32
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# Initialize model, criterion, optimizer, scheduler
input_size = X_train.shape[1]
num_classes = len(np.unique(y_train))

model = MLP(input_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training and validation loop
num_epochs = 100
train_losses = []
val_losses = []
macro_f1_scores = []
#%%

for epoch in range(num_epochs):
    # Training loop
    model.train()
    batch_train_losses = []
    for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        batch_train_losses.append(loss.item())
    train_losses.append(np.mean(batch_train_losses))

    # Validation loop
    model.eval()
    batch_val_losses = []
    val_preds = []
    with torch.no_grad():
        for batch_idx, (batch_X, batch_y) in enumerate(val_loader):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            val_outputs = model(batch_X)
            val_loss = criterion(val_outputs, batch_y)
            batch_val_losses.append(val_loss.item())
            
            val_probs = F.softmax(val_outputs, dim=1)
            val_preds.append(torch.argmax(val_probs, dim=1))
            
        val_losses.append(np.mean(batch_val_losses))
        val_preds = torch.cat(val_preds)

        # Metrics
        precision = precision_score(y_val.cpu(), val_preds.cpu(), average=None)
        recall = recall_score(y_val.cpu(), val_preds.cpu(), average=None)
        f1 = f1_score(y_val.cpu(), val_preds.cpu(), average=None)
        macro_f1 = f1_score(y_val.cpu(), val_preds.cpu(), average='macro')
        macro_f1_scores.append(macro_f1)

        window_size = 10
        if len(macro_f1_scores) >= window_size:
            moving_avg_macro_f1 = np.mean(macro_f1_scores[-window_size:])
        else:
            moving_avg_macro_f1 = np.mean(macro_f1_scores)

        # Adjust learning rate based on validation loss
        #scheduler.step(np.mean(batch_val_losses))

    # Print results after each epoch
    print(f'Epoch [{epoch + 1}/{num_epochs}]')
    print("="*50)
    print(f'LEARNING RATE: {optimizer.param_groups[0]["lr"]}')
    print("="*50)
    print(f'Validation Loss: {np.mean(batch_val_losses)}')
    print("="*50)
    print(f'MOVING AVERAGE MACRO-AVERAGED F1 SCORE (last {min(window_size, len(macro_f1_scores))} epochs): {moving_avg_macro_f1:.4f}')
    print("="*50)
    # for idx, class_name in enumerate(label_encoder.classes_):
    #     print(f'Class {class_name} - Precision: {precision[idx]:.4f}, Recall: {recall[idx]:.4f}, F1 Score: {f1[idx]:.4f}')
    # print("="*50)
    
#%%
    # Plot confusion matrix
    cm = confusion_matrix(y_val.cpu(), val_preds.cpu())
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 7))
    heatmap = sns.heatmap(
        cm_normalized, 
        #annot=cm, 
        fmt='d', 
        cmap='rocket_r', 
        xticklabels=label_encoder.classes_, 
        yticklabels=label_encoder.classes_,
        cbar_kws={'label': 'Scale'}
    )
    
    # Set the font size of the labels
    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=60, horizontalalignment='right', fontsize=7)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, fontsize=7)
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'MLP - Epoch {epoch + 1}\nMoving Average Macro-averaged F1 Score: {moving_avg_macro_f1:.4f}')
    plt.show()

#%%
# Plot training and validation losses and macro F1 score
plt.figure(figsize=(10, 5))
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