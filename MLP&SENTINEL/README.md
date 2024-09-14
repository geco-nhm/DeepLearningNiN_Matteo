In this subfolder, you will find the code that i used to process the sentinel data, convert it to CSV with the values of each band for each pixel, and it's class,and then pass it to a Multilayer perceptron. 
You will also find the code that i used when i tried to predict lime richness.
**Train.csv** is the piece of code that contains the MLP


**ALL_SENTINEL2_CLEANED.csv** is an example of development dataset that has sentinel 2 pixels labeled wherever there were polygons (both vega and landsvik) based on that of anders. It can be split in a train and validation dataset using the ad hoc snippet of code in the preprocessing section. 
