# DICOM Preprocessing for Lung Cancer Risk Prediction

A comprehensive preprocessing pipeline for LDCT DICOM series that handles longitudinal time-point scans (T0, T1, T2). The preprocessed outputs are compatible with deep learning models for lung cancer risk prediction feature extraction.

## ✨ Features

- ✅ **Single Series Processing**: Preprocess individual DICOM series with one command
- ✅ **Multi-Timepoint Batch Processing**: Process longitudinal data (T0, T1, T2) automatically
- ✅ **Localizer Detection**: Automatically detects and skips scout/localizer scans
- ✅ **Flexible Thickness Acceptance**: Accepts scans up to 10mm thickness and resamples to 2.5mm
- ✅ **Comprehensive JSON Logging**: Detailed logs with metadata, statistics, and rejection reasons
- ✅ **Automatic Resampling**: Resamples to required voxel spacing (0.703125 × 0.703125 × 2.5 mm)
- ✅ **Anatomical Ordering**: Ensures correct slice ordering (abdomen → clavicles)
- ✅ **Progress Tracking**: Automatic saving every 5 patients with resume support
- ✅ **Robust Error Handling**: Detailed logging and graceful failure handling

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Preprocessing DICOM Files

#### Single DICOM Series

```bash
python preprocessing.py single --input /path/to/dicom/folder --output /path/to/output/directory
```

Output files are automatically named as `{input_folder_name}_preprocessed.pt`.

#### Batch Processing

Process multiple series from a CSV file with multi-timepoint support:

```bash
python preprocessing.py batch --csv input_data.csv --output_dir ./preprocessed --output_csv preprocessed_output.csv
```

**Resume interrupted processing:**

```bash
python preprocessing.py batch --csv input_data.csv --output_dir ./preprocessed --output_csv preprocessed_output.csv --resume-from PATIENT_ID
```

> **Note**: After preprocessing, the output volumes can be used for feature extraction with deep learning models. Use the csv file obtained from here for feature extraction section.

## Input Format

### CSV for Batch Processing

The input CSV file must contain these columns:

| Column | Description |
|--------|-------------|
| `pid` | Patient/Series identifier (unique ID) |
| `T0`, `T1`, `T2` | Flags (1 or 0) indicating which timepoints to process |
| `dicom_pathT0`, `dicom_pathT1`, `dicom_pathT2` | Full paths to DICOM directories for each timepoint |
| `labelT0`,...`labelT7` | (1,0) indicating at which year the cancer was diagnosed |

The output CSV will include additional columns:

| Column | Description |
|--------|-------------|
| `volT0`, `volT1`, `volT2` | Paths to the saved preprocessed .pt files |

Additionally, a `preprocessing_log.json` file is created with detailed processing information.

## Output Format

### Output Files

Each preprocessed file is saved as a PyTorch `.pt` file containing:

```python
{
    'volume': torch.Tensor,              # Shape: (1, 3, 200, 256, 256)
    'normalization_mean': float,         # 128.1722
    'normalization_std': float,          # 87.1849
    'original_path': str,                # Path to original DICOM directory
    'num_original_slices': int,          # Number of original DICOM slices
    'patient_id': str,                   # Patient ID (batch mode only)
    'timepoint': str                     # Timepoint (T0, T1, T2)
}
```

## Preprocessing Details

1. **DICOM Loading**: Reads all DICOM files from the specified directory
2. **Localizer Detection**: Automatically skips scout/localizer scans
3. **Thickness Validation**: Accepts scans with slice thickness ≤ 10mm
4. **Anatomical Ordering**: Sorts slices by DICOM position (abdomen → clavicles)
5. **Voxel Spacing**: Resamples to (0.703125, 0.703125, 2.5)mm
6. **Slice Adjustment**: Adjusts to exactly 200 slices (padding or cropping)
7. **Normalization**: Applies Z-score normalization with mean=128.1722, std=87.1849
8. **Output**: Saves as PyTorch tensor with shape (1, 3, 200, 256, 256)

## JSON Processing Log

For batch processing, a detailed `preprocessing_log.json` file is created with status information for each patient and timepoint:

**Status Codes:**
- `success`: Preprocessed successfully ✅
- `failed`: Error during preprocessing ❌
- `skipped`: Localizer scan detected 🚫
- `rejected`: Thickness exceeds threshold ❌
- `not_requested`: Flag not set to 1 ⏭️
- `already_processed`: Output file exists ⏭️
