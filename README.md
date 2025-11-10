# Minu: Deep learning models for Medical Image Analysis to predict future risk of lung cancer over several subsequent years.

A comprehensive framework for medical image analysis, focusing on feature extraction and data processing for lung cancer risk prediction from LDCT scans.
Sybil is used as pure feature extrator. The preprocessing.py is identical and compatible to Sybil preprocessing pipeline, now is ready to be used for any external LDCT sets as well as NLST dataset.

## Project Overview

This project provides a complete pipeline for processing and analyzing medical imaging data, with a focus on CT scan analysis for lung cancer risk prediction. It includes tools for feature extraction, patient timeline analysis, and comprehensive data processing.

## Installation

1. Clone the repository:
```bash
git git@github.com:hajami0802/Minu.git
cd Minu
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```


## Pipeline Overview

```
Raw DICOM Files
      ↓
[preprocessing.py]
      ↓
Preprocessed Volumes (.pt)
      ↓
 Feature Extraction
      ↓
Resnet features, Attention features, Final embeddings
      ↓
Deep Learning Models
      ↓
BiFusion, TriFusion, Hybrid 
```


## Usage

### 1. DICOM Preprocessing Tool

A robust preprocessing pipeline for CT DICOM series that handles longitudinal time-point scans (T0, T1, T2).

📖 **[Full Documentation](preprocessing/README.md)**

### 2. Feature extraction at multiple depth

Extract features from preprocessed LDCT scans using Sybil's 5-checkpoint ensemble model.

📖 **[Full Documentation](feature_extraction/README.md)**


## Reference and Citation
@article{mikhael2023sybil,
  title={Sybil: a validated deep learning model to predict future lung cancer risk from a single low-dose chest computed tomography},
  author={Mikhael, Peter G and Wohlwend, Jeremy and Yala, Adam and others},
  journal={Journal of Clinical Oncology},
  volume={41},
  number={12},
  pages={2191--2200},
  year={2023},
  publisher={American Society of Clinical Oncology}
}

