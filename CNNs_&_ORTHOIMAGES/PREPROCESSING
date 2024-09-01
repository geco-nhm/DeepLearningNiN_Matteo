Here are various snippets of code that i used to build the validation datasets that i later fed to the TRAIN file. 

Concat_bands.py concatenates tiffs along the bands dimension (you can eg feed red and green in two diffrent files, and it stacks them together)

raster_tiling.py takes a raster and a shapefile with the classes of the mappers,it tiles the raster in tiles of a chosen size (eg 64x64 pixels), and adds them in a folder of the class it intersects from the shapefile ONLY IF IT INTERSECTS ONLY ONE CLASS (a tile overlaping mutliple classes won't be saved anywhere). 

tiff_to_jpg.py is to be applied to a final validation dataset, and write the same, with jpg files instead of tif so they can be passed to the TRAIN.py code
