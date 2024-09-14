#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 11:27:11 2024

@author: matteo_crespinjouan
"""

import rasterio
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import shape
from rasterio.features import geometry_mask

# File paths
tiff_file = '/Users/matteo_crespinjouan/Desktop/NHM/SENTINEL_MLP_ENDEAVOUR/SENTINEL_FINAL/vegaA.tiff'
shapefile_path = '/Users/matteo_crespinjouan/Desktop/NHM/shp_endeavour/anders_vigga.shp'
output_csv = '/Users/matteo_crespinjouan/Desktop/NHM/SENTINEL_MLP_ENDEAVOUR/SENTINEL_FINAL/vegaA.csv'

# Read the shapefile using geopandas
gdf = gpd.read_file(shapefile_path)

# Ensure CRS match between raster and shapefile
with rasterio.open(tiff_file) as src:
    raster_crs = src.crs
    gdf = gdf.to_crs(raster_crs)

# Read the raster file
with rasterio.open(tiff_file) as src:
    bands = src.read()
    transform = src.transform

# Initialize lists to store the data
data = []

# Iterate over each shape and its corresponding record
for idx, row in gdf.iterrows():
    polygon = row.geometry
    class_label = row['gtype1']  # Adjust this key based on your shapefile schema

    # Create a mask for the current polygon
    mask = geometry_mask([polygon], transform=transform, invert=True, out_shape=(src.height, src.width))

    # Get the indices of the pixels that intersect with the polygon
    rows, cols = np.where(mask)

    for row, col in zip(rows, cols):
        pixel_values = bands[:, row, col]
        data.append([class_label] + pixel_values.tolist())

# Convert the data to a pandas DataFrame
df = pd.DataFrame(data, columns=['class'] + [f'band_{i+1}' for i in range(bands.shape[0])])

# Save the DataFrame to a CSV file
df.to_csv(output_csv, index=False)

print(f"Data saved to {output_csv}")