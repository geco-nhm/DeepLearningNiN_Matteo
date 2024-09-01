#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 16:11:35 2024

@author: matteo_crespinjouan
"""

import rasterio
from rasterio.windows import Window
import geopandas as gpd
from shapely.geometry import box
import os
import numpy as np

# Define the input files and output directory
raster_path = '/Users/matteo_crespinjouan/Desktop/NHM/ORTHO_ENDEAVOUR/tif_ortho/Eldsfjellet-01-Vestlandet CIR 2020.tif'
shapefile_path = '/Users/matteo_crespinjouan/Desktop/NHM/shp_endeavour/fusion.shp'
output_dir = '/Users/matteo_crespinjouan/Desktop/NHM/ORTHO_ENDEAVOUR/32_tiffs'
#%%
# Read the shapefile
try:
    gdf = gpd.read_file(shapefile_path)
except Exception as e:
    print(f"Error reading shapefile: {e}")
    raise

# Check if the column exists
if 'gtype1' not in gdf.columns:
    print("Column 'gtype1' does not exist in the shapefile. Available columns are:")
    print(gdf.columns)
    raise ValueError("Column 'gtype1' not found.")

# Ensure CRS match between raster and shapefile
with rasterio.open(raster_path) as src:
    raster_crs = src.crs
    gdf = gdf.to_crs(raster_crs)

# Define the size of the patches
patch_size = 32

# Create the output directories if they don't exist
classes = gdf['gtype1'].unique()
for class_name in classes:
    class_dir = os.path.join(output_dir, class_name)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

print("Directories created successfully.")

# Open the raster file
with rasterio.open(raster_path) as src:
    for i in range(0, src.width, patch_size):
        for j in range(0, src.height, patch_size):
            # Define the window
            window = Window(i, j, patch_size, patch_size)
            
            # Read the patch
            patch = src.read(window=window)
            
            if patch.size == 0:
                print(f"Empty patch at ({i}, {j})")
                continue

            # Get the bounding box of the patch
            bbox = box(*src.window_bounds(window))
            
            print(f"Patch at ({i}, {j}) with bbox {bbox.bounds}")

            # Find intersecting polygons and their classes
            intersecting_polygons = gdf[gdf.intersects(bbox)]
            
            if not intersecting_polygons.empty:
                # Determine if there is only one unique class
                unique_classes = intersecting_polygons['gtype1'].unique()
                
                if len(unique_classes) == 1:
                    predominant_class = unique_classes[0]
                    
                    # Ensure class directory exists
                    class_dir = os.path.join(output_dir, predominant_class)
                    if not os.path.exists(class_dir):
                        os.makedirs(class_dir)
                    
                    # Save the patch
                    patch_path = os.path.join(class_dir, f'{i}_{j}.tif')
                    with rasterio.open(patch_path, 'w', driver='GTiff', height=patch.shape[1], width=patch.shape[2],
                                       count=src.count, dtype=patch.dtype, crs=src.crs, transform=src.window_transform(window)) as dst:
                        dst.write(patch)
                    print(f"Saved patch to {patch_path}")
                else:
                    print(f"Patch at ({i}, {j}) intersects multiple classes: {unique_classes}")
            else:
                print(f"No intersecting polygons for patch at ({i}, {j})")

print("Patches have been saved successfully.")