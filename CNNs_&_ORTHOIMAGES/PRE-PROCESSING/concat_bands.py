#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 14:52:31 2024

@author: matteo_crespinjouan
"""

import rasterio
from rasterio.enums import Resampling
import numpy as np
#%%
# Paths to the input .tif images
nir_path = '/Users/matteo_crespinjouan/Desktop/NHM/ORTHO_ENDEAVOUR/tif_ortho/2023_Landsvik_multispectral_index_nir.tif'
blue_path = '/Users/matteo_crespinjouan/Desktop/NHM/ORTHO_ENDEAVOUR/tif_ortho/2023_Landsvik_multispectral_index_red.tif'
green_path = '/Users/matteo_crespinjouan/Desktop/NHM/ORTHO_ENDEAVOUR/tif_ortho/2023_Landsvik_multispectral_index_green.tif'

# Open the images and read the data
with rasterio.open(nir_path) as nir:
    nir_band = nir.read(1, resampling=Resampling.nearest)
    meta = nir.meta

with rasterio.open(blue_path) as blue:
    blue_band = blue.read(1, resampling=Resampling.nearest)

with rasterio.open(green_path) as green:
    green_band = green.read(1, resampling=Resampling.nearest)
    
print("marjane")

# Stack the bands along the third dimension
stacked_array = np.stack([nir_band, blue_band, green_band], axis=0)


# Update the metadata to reflect the number of layers
meta.update({"count": 3})


# Path for the output .tif
output_path = '/Users/matteo_crespinjouan/Desktop/NHM/ORTHO_ENDEAVOUR/tif_ortho/nir_red_green.tif'

print("noisette")
# Write the stacked array to a new .tif file
with rasterio.open(output_path, 'w', **meta) as dst:
    dst.write(stacked_array)

print(f"Successfully created {output_path}")