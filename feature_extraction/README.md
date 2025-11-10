# LDCT Scan Feature Extraction

Extract features from preprocessed LDCT scans using Sybil's 5-checkpoint ensemble model.
Ensure to feed the preprocessed scans obtained from preprocessing.py.

## Features

- Global ResNet Features
- Local Attention Features
- Final Embeddings
- Multi-checkpoint Ensemble
- Resume Support
- Maintains Directory Structure

## Installation

```bash
pip install -r requirements.txt
```

## Usage
--preprocessed_output.csv for all commands are the csv file that output from preprocessing.py 

### Global ResNet Features: Encoded features
```bash
python extract_features.py global --csv preprocessed_output.csv --checkpoint_dir ./checkpoints --output_dir ./resnet_features --output_csv resnet_features.csv
```

### Local Features: Attention features extracted from multi attentions guided mechanism in Sybil.
```bash
python extract_features.py local --csv preprocessed_output.csv --checkpoint_dir ./checkpoints --output_dir ./local_features --output_csv local_features.csv
```

### Final Embeddings
```bash
python extract_features.py final --csv preprocessed_output.csv --checkpoint_dir ./checkpoints --output_dir ./finalembeddings_features --output_csv finalembeddings_features.csv
```

## Input Format

The input CSV must contain these columns:
- `pid`: Patient ID
- `volT0`, `volT1`, `volT2`: Paths to preprocessed `.pt` files from preprocessing.py

## Output Format

### Global Features (`.pt`)
```python
{
    'ensemble_features': torch.Tensor,      # Shape: (1, 512, D', H', W')
    'feature_shape': tuple,
    'input_volume_shape': tuple,
    'num_checkpoints': int,
    'checkpoint_paths': list
}
```

### Local Features (`.pt`)
```python
{
    'ensemble_local_features': torch.Tensor,  # Spatially attended features
    'feature_shape': tuple,
    'input_volume_shape': tuple,
    'description': str
}
```

### Final Embeddings (`.npy`)
NumPy array containing final ensemble embeddings

## Requirements

- Python 3.8+
- PyTorch
- CUDA-capable GPU (recommended)
- Pre-trained Sybil checkpoints

## Troubleshooting

### Common Issues

**"No volT0/volT1/volT2 columns found"**
- Ensure you're using the CSV output from preprocessing.py
- The CSV must contain `volT0`, `volT1`, and/or `volT2` columns

## Author

Hanieh Ajami
