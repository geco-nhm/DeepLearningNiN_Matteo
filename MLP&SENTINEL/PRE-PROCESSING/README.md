here, **sentinel_to_csv.py** takes a shapefile of the classes and a sentinel multichanel .tif image, and it writes on a csv, for each pixel, its class followed by the activation across all the chanels : 
for example : 

class,Blue 490nm,Green 560nm,Red 665nm,VRE 705nm,VRE 740nm,VRE 783nm,NIR 843nm,SWIR 1610nm,SWIR 2190nm,VRE 865nm
T12-C-1,0.0232999995350837,0.0531000010669231,0.0423000007867813,0.1190000027418136,0.2614000141620636,0.3061999976634979,0.3043999969959259,0.19480000436306,0.09740000218153,0.3425000011920929
T12-C-1,0.0186999998986721,0.0529000014066696,0.0381000004708766,0.1190000027418136,0.2614000141620636,0.3061999976634979,0.3424000144004822,0.19480000436306,0.09740000218153,0.3425000011920929
etc 

**DUPLICATE_REMOVER.py** removes all the lines that occur twice, to make sure the model isn't shown an example in both the train and validation dataset. 

**csv_train_test.py** takes a csv like above and splits it into a development and a training dataset



