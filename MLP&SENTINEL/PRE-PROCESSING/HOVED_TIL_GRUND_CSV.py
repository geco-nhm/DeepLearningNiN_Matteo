#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 20:23:13 2024

@author: matteo_crespinjouan
"""

import pandas as pd

# Path to the CSV file
file_path = '/Users/matteo_crespinjouan/Desktop/NHM/SENTINEL_MLP_ENDEAVOUR/SENTINEL_FINAL/vegaABC_cleaned.csv'

# Read the CSV file
df = pd.read_csv(file_path)

# Rename the classes by keeping only the part before the first hyphen
df['class'] = df['class'].apply(lambda x: x.split('-')[0])

# Save the modified data to a new CSV file
df.to_csv('/Users/matteo_crespinjouan/Desktop/NHM/SENTINEL_MLP_ENDEAVOUR/SENTINEL_FINAL/VEGA_hoved.csv', index=False)

print("Classes renamed and new CSV file saved.")