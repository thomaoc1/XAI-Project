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

## Reproducing Using Model Weights
If you have been given the model weights for each model you can create the `model/` directory at the project root and move them all into it.
The directory should look like this: 
```
model
├── deepfake_classifier.pt
├── deepfake_fake_hm_vae.pt
├── deepfake_hm_vae.pt
├── deepfake_real_hm_vae.pt
├── dogs_vs_cats_cat_hm_vae.pt
├── dogs_vs_cats_classifier.pt
├── dogs_vs_cats_dog_hm_vae.pt
└── dogs_vs_cats_hm_vae.pt
```

From here you can skip straight to the evaluation of the VAE (Execution - Section 3)

## Execution

### Full Pipeline
The pipeline is as follows:
1. Train target classifier
2. Generate heatmap dataset
3. Train VAE on heatmap dataset
4. Evaluate VAE anomaly detection performance

This can be done by running the following command:

```bash
python -m src.run_full_pipeline <dataset> [--target_class <class>] [--classifier_batch_size <int>] [--vae_batch_size <int>] [--classifier_epochs <int>] [--vae_epochs <int>]
```

### 1. Training/Evaluating the Classifier
Everything that concerns training the target classifier can be found in the module `src.classification`, the following is a brief description of the contents of the module.

#### src.classification.binary_classifier
Contains the implementation of the target classifier for both datasets.

#### src.classification.train
Script used to train the classifier on a given dataset, can be executed as follows: 

```bash
python -m src.classification.train <dataset> [--epochs <int>] [--batch_size <int>] 
```

This will save the model weights to the path `model/{dataset}_classifier.pt`

#### src.classification.eval
Script used to evaluate classification performance of model on a given dataset which can be executed as such:
```bash
python -m src.classification.eval <dataset> [--classifier_batch_size <int>] 
```
Training the model or adding weights under the path `model/{dataset}_classifier.pt` is a prerequisite to this step.

### 2. Heatmap Generation 
Everything concerning the heatmap dataset generate can be found in the module `src.datagen`, the following is a brief description of the contents of the module.

#### src.datagen.dataset
This file contains the PyTorch implementation of the heatmap dataset.

#### src.datagen.generate_dataset
This script generates the clean heatmaps for a given dataset and optionally a specific class. It is executed as follows: 
```bash
python -m src.datagen.generate_dataset <dataset> [--batch_size <int>] [--target_class <class>]
```
Training the model or adding weights under the path `model/{dataset}_classifier.pt` is a prerequisite to this step.

### 3. Training/Evaluating the VAE
Everything that concerns training and evaluating the VAE can be found in the module `src.anomalydetection`, the following is a brief description of the contents of the module.

#### src.anomalydetection.vae
This file contains the PyTorch implementation of the CNN-VAE which expects a 224x224 input image and outputs the same. 

#### src.anomalydetection.train
This script allows for the training of the VAE given a dataset and optionally a target class. If you generated a heatmap dataset using a target class then you must ensure that you pass the correct one to this script. It can be run as follows:
```bash
python -m src.anomalydetection.train <dataset> [--epochs <int>] [--batch_size <int>] [--target_class <class>]
```
Generating a heatmap or copying on to the path `dataset/heatmap/{dataset}_hm_dataset.pt` is a prerequisite to this step.

#### src.anomalydetection.eval_vae
This script allows for the evaluation of a VAE model given a dataset and optionally a target class and attack strength (epsilon). It can be run as follows:
```bash
python -m src.anomalydetection.eval_vae <dataset> [--target_class <class>]
```
Training a VAE or copying its weights to the path `model/{dataset}_hm_vae.pt` is a prerequisite to this step.
