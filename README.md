# Deep learning for automatic segmentation of cerebral infarcts on diffusion-weighted images:  Effects of training data size, domain adaptation, and data features

<div align="center">
  <img width="100%" alt="Research flowchart" src="https://drive.google.com/uc?export=view&id=1B8_EEV68jpadt8_-ATKh2Usnndz9RxMZ">
</div>
  
This is the code repository for our paper titled "Deep Learning for Automatic Segmentation of Cerebral Infarcts on Diffusion-Weighted Images: Effects of Training Data Size, Domain Adaptation, and Data Features". The study examined the effectiveness and generalizability of deep learning for acute infarct segmentation on diffusion-weighted magnetic resonance imaging (DWI).  
  
This study examined the impact of single-site versus multi-site training data, training data size, and domain adaptation on the efficacy of a model using a dataset containing 10,820 DWIs from 10 hospitals.

# Requirements

- Python 3.7
- [Tensorflow](https://www.tensorflow.org/) 2.9.2
- tensorflow-addons 0.18.0
- scikit-image 0.18.1
- scikit-learn 1.1.2
- SimpleITK 2.1.1
- pydicom 2.1.2

Full list of required packages is in requirements.txt. Since the code is rather simple, you will probably be fine with different package versions.

# Data Preparation

You will need B1000 diffusion-weighted MR images of brain to start with. For each patient, DICOM files and lesion mask has to be located in seperate folders named 'dcm' and 'gt'.  
For each lesion mask, the file name should be same as its corresponding DICOM slice and image format such as .jpg or .png is accepted. Also check that the mask is of same size as the image in DICOM file.
```
├── raw_images
│   ├── dataset1
│   │   └── ExampleCase
│   │       ├── dcm
│   │       └── gt
│   └── dataset2
├── preprocessed_images
│   ├── dataset1
│   │   ├── rt_img
│   │   └── gt
│   └── dataset2
├── data_list
│   ├── dataset1
│   └── dataset2
└── result
    ├── dataset1
    └── dataset2
```
Data tree template can be found in resource folder.

# Running codes
Here are some example commands used to run each script in the experiment.

## Image preprocessing
```
python data_preprocessing.py \
--raw_path /path/to/DICOM/and/GT/files \
--save_path /path/to/preprocessed/data \
--img_size 256 256 \
--n_workers 12
```

## Dataset split
```
python data_split.py \
--data_path /path/to/preprocessed/data \
--save_path /path/to/save/data/list \
--val_size 0.2 \
--test_size 0.2
```

## Model training
```
python train.py \
--list_path /path/to/data/list \
--data_path /path/to/preprocessed/data \
--img_size 256 256 \
--n_slices 64 \
--n_channels 1 \
--epochs 1000 \
--batch_size 2 \
--gpu_id 0 \
--ckpt_path /path/to/checkpoints
```

## Model evaluation
```
python evaluation.py \
--list_fpath /path/to/test/list \
--data_path /path/to/preprocessed/data \
--csv_fpath /save/path/to/csv/file \
--gpu_id 0 \
--model_fpath /path/to/checkpoint/for/evaluation
```

## Acknowledgements

None.