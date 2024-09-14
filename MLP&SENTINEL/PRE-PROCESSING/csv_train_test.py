#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 11:38:03 2024

@author: matteo_crespinjouan
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# File paths
input_csv = "/Users/matteo_crespinjouan/Desktop/NHM/SENTINEL_MLP_ENDEAVOUR/SENTINEL_FINAL/ALL_SENTINEL2_CLEANED.csv"
train_csv = '/Users/matteo_crespinjouan/Desktop/NHM/SENTINEL_MLP_ENDEAVOUR/SENTINEL_FINAL/MLP_FINAL_all_gruntyper/gruntyper_train.csv'
validation_csv = '/Users/matteo_crespinjouan/Desktop/NHM/SENTINEL_MLP_ENDEAVOUR/SENTINEL_FINAL/MLP_FINAL_all_gruntyper/gruntyper_val.csv'

# Read the input CSV file
df = pd.read_csv(input_csv)

# Split the data into training and validation sets (80-20 split)
train_df, validation_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['class'])

# Save the training and validation sets to separate CSV files
train_df.to_csv(train_csv, index=False)
validation_df.to_csv(validation_csv, index=False)

print(f"Training data saved to {train_csv}")
print(f"Validation data saved to {validation_csv}")