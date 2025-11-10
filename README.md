# Medical Image Analysis Framework: Minu

A comprehensive framework for medical image analysis, focusing on feature extraction and data processing for lung cancer risk prediction from CT scans.

## Project Overview

This project provides a complete pipeline for processing and analyzing medical imaging data, with a focus on CT scan analysis for lung cancer risk prediction. It includes tools for feature extraction, patient timeline analysis, and comprehensive data processing.

## Installation

1. Clone the repository:
```bash
git git@github.com:hajami0802/Minu.git
cd Meenu
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. DICOM Preprocessing Tool

A robust preprocessing pipeline for CT DICOM series that handles longitudinal time-point scans (T0, T1, T2).

📖 **[Full Documentation](preprocessing/README.md)**

### 2. Feature extraction at multiple depth

Extract features from preprocessed LDCT scans using Sybil's 5-checkpoint ensemble model.

📖 **[Full Documentation](feature_extraction/README.md)**

## Contact

Hanieh Ajami - [@hajami0802](https://github.com/hajami0802)

Project Link: [https://github.com/hajami0802/Minu.git](https://github.com/hajami0802/Minu.git)
