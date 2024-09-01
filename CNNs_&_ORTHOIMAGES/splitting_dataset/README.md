Here, is used mainly the train/val script. 
it requires a folder that contains subfolders of all the classes filled with the data for each class. 
for instance, it would start like : 

dataset/
│
├── forest/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
│
├── desert/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
│
└── glacier/
    ├── image1.jpg
    ├── image2.jpg
    └── ...

  and output : 

dataset/
│
├── train/
│   ├── forest/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── desert/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── glacier/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
│
├── val/
│   ├── forest/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── desert/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── glacier/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
│
└── test/
    ├── forest/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── desert/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── glacier/
        ├── image1.jpg
        ├── image2.jpg
        └── ...
