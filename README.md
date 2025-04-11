# XAI-Project
## Venv
```
python -m venv venv  
source venv/bin/activate  
pip install -r requirements.txt
```

## Datasets
To start, download the [DeepFake](https://www.kaggle.com/datasets/itamargr/dfdc-faces-of-the-train-sample/data) and [Dogs-vs-Cats](https://www.kaggle.com/c/dogs-vs-cats/data) datasets. Next, create directory `dataset` at the project root level, move the datasets into the folder. Should look like this:
```
dataset
├── dogs-vs-cats
└── deepfake-dataset
```

Next format the datasets by running `python format_datasets.py`

Your `dataset` directory should now look like so: 

```
dataset
├── deepfake
│   ├── train
│   │   ├── fake
│   │   └── real
│   └── validation
│       ├── fake
│       └── real
└── dogs-vs-cats
    ├── train
    │   ├── cat
    │   └── dog
    └── validation
        ├── cat
        └── dog
```
    
Verify using `python verify_dataset.py`

## Execution
The pipeline is as follows:
1. Train original classifier
2. Generate heatmap dataset
3. Train VAE on heatmap dataset
4. Evaluate VAE anomaly detection performance

This can be done by running the following command 

```bash
python -m src.run_full_pipeline <dataset> [--target_class <class>] [--classifier_batch_size <int>] [--vae_batch_size <int>] [--classifier_epochs <int>] [--vae_epochs <int>]
```