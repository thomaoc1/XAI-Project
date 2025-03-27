# XAI-Project
## Venv
```
python -m venv venv  
source venv/bin/activate  
pip install -r requirements.txt
```

## Dataset
To start, download dataset from [here](https://www.kaggle.com/datasets/itamargr/dfdc-faces-of-the-train-sample/data). Next, create directory `dataset` at the project root level, move the dataset into the folder. Should look like this:
```
dataset
└── deepfake-dataset
    ├── train
    │   ├── fake
    │   └── real
    └── validation
        ├── fake
        └── real
```
Verify using `python verify_dataset.py`