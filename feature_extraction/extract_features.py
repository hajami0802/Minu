#!/usr/bin/env python3
"""
Feature Extraction Tool for CT Volumes
====================================

Extracts features from preprocessed CT volumes using Sybil's 5-checkpoint ensemble.
Supports both global ResNet features and local attention features.

Author: Hanieh Ajami
Repository: https://github.com/hajami0802/Lung-Cancer-Risk-Prediction
License: MIT

Usage:
------
# For ResNet (global) features:
python extract_features.py global --csv input.csv --checkpoint_dir ./checkpoints --output_dir ./features --output_csv results.csv

# For Local attention features:
python extract_features.py local --csv input.csv --checkpoint_dir ./checkpoints --output_dir ./features --output_csv results.csv

Input CSV Format:
-----------------
Must contain columns: pid, volT0, volT1, volT2 (paths to preprocessed .pt files from preprocessing step)
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import argparse
import logging
from pathlib import Path
import numpy as np
from typing import Optional, List, Dict, Any, Union
from types import SimpleNamespace
from torchvision.models.video import r3d_18
from tqdm import tqdm

# For local features and final embeddings
from pooling_layer import MultiAttentionPool

# =============================================================================
# LOGGING SETUP
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set seeds for reproducibility
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# =============================================================================
# CONSTANTS
# =============================================================================
CHECKPOINT_FILENAMES = [
    "28a7cd44f5bcd3e6cc760b65c7e0d54d.ckpt",
    "56ce1a7d241dc342982f5466c4a9d7ef.ckpt",
    "64a91b25f84141d32852e75a3aec7305.ckpt",
    "65fd1f04cb4c5847d86a9ed8ba31ac1a.ckpt",
    "624407ef8e3a2a009f9fa51f9846fe9a.ckpt",
]

DROPOUT_P = 0.2
MAX_FOLLOWUP = 6

# =============================================================================
# MODEL CLASSES
# =============================================================================
class SybilBackbone(torch.nn.Module):
    """ResNet18 3D backbone for feature extraction"""
    def __init__(self):
        super().__init__()
        enc = r3d_18(weights=None)
        self.image_encoder = torch.nn.Sequential(
            enc.stem, enc.layer1, enc.layer2, enc.layer3, enc.layer4
        )
    
    def forward(self, x):
        return self.image_encoder(x)


class SybilHead(torch.nn.Module):
    """Head module for local feature extraction"""
    def __init__(self):
        super().__init__()
        self.pool = MultiAttentionPool()
        self.act = torch.nn.ReLU(inplace=False)
    
    def forward(self, x):
        pool_output = self.pool(x)
        # Extract local attention features (spatially attended per slice)
        local_features = pool_output['multi_image_hidden_1']
        return local_features
    
    def forward_with_embeddings(self, x):
        """Forward pass that returns final embeddings"""
        pool_output = self.pool(x)
        h = self.act(pool_output['hidden'])  # Final embeddings
        return h


# =============================================================================
# MODEL LOADING FUNCTIONS
# =============================================================================
def load_backbone_weights(bb, ckpt):
    """Load weights into backbone"""
    try:
        # First try loading with weights_only=True
        raw = torch.load(ckpt, map_location='cuda', weights_only=True)
    except Exception as e:
        # If that fails, try with weights_only=False for older checkpoints
        raw = torch.load(ckpt, map_location='cuda', weights_only=False)
    
    sd = raw.get('state_dict', raw)
    clean = {k.split('.', 1)[-1]: v for k, v in sd.items()}
    encoder_weights = {
        k.split('image_encoder.', 1)[1]: v
        for k, v in clean.items() if 'image_encoder.' in k
    }
    bb.image_encoder.load_state_dict(encoder_weights, strict=True)


def load_head_weights(head, ckpt):
    """Load weights into head"""
    try:
        # First try loading with weights_only=True
        raw = torch.load(ckpt, map_location='cuda', weights_only=True)
    except Exception as e:
        # If that fails, try with weights_only=False for older checkpoints
        raw = torch.load(ckpt, map_location='cuda', weights_only=False)
    
    sd = raw.get('state_dict', raw)
    clean = {k.split('model.', 1)[-1]: v for k, v in sd.items()}
    head_weights = {k: v for k, v in clean.items() if not k.startswith('image_encoder.')}
    missing, unexpected = head.load_state_dict(head_weights, strict=False)
    return len(missing) == 0 and len(unexpected) == 0


# =============================================================================
# FEATURE EXTRACTION FUNCTIONS
# =============================================================================
def extract_resnet_features(
    preprocessed_pt_path: str,
    backbones: List[SybilBackbone],
    device: torch.device
) -> Dict[str, Any]:
    """Extract ensemble ResNet (global) features from a preprocessed .pt file"""
    try:
        # Load preprocessed volume
        data = torch.load(preprocessed_pt_path, map_location='cpu', weights_only=False)
        
        # Extract volume tensor
        if isinstance(data, dict):
            volume = data['volume']
        else:
            volume = data
        
        # Ensure input is float32
        volume = volume.to(torch.float32)
        
        # Move to device
        volume = volume.to(device).contiguous()
        
        # Extract features from all checkpoints
        all_backbone_features = []
        
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            for backbone in backbones:
                backbone_feats = backbone(volume)
                backbone_feats = backbone_feats.to(torch.float32)
                all_backbone_features.append(backbone_feats.cpu())
        
        # Create ensembled features
        ensemble_features = torch.mean(torch.stack(all_backbone_features), dim=0)
        
        return {
            'success': True,
            'ensemble_features': ensemble_features,
            'feature_shape': ensemble_features.shape,
            'input_shape': volume.shape,
            'num_checkpoints': len(backbones)
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def extract_final_embeddings(
    preprocessed_pt_path: str,
    models: List[tuple],
    device: torch.device
) -> Dict[str, Any]:
    """Extract ensemble final embeddings from a preprocessed .pt file"""
    try:
        # Load preprocessed volume
        data = torch.load(preprocessed_pt_path, map_location='cpu', weights_only=False)
        
        # Extract volume tensor
        if isinstance(data, dict):
            vol = data['volume']
            normalization_mean = data.get('normalization_mean', 'unknown')
            normalization_std = data.get('normalization_std', 'unknown')
        else:
            vol = data
            normalization_mean = 'unknown'
            normalization_std = 'unknown'
        
        vol = vol.to(device)
        all_final_embeddings = []
        
        # Process each checkpoint
        for i, (bb, head) in enumerate(models):
            with torch.no_grad():
                # Extract ResNet backbone features
                backbone_feats = bb(vol)
                
                # Extract final embeddings
                final_embeddings = head.forward_with_embeddings(backbone_feats)
                
                # Store for ensemble averaging
                all_final_embeddings.append(final_embeddings.cpu().numpy())
        
        # Create ensembled embeddings
        ensemble_final_embeddings = np.mean(np.stack(all_final_embeddings), axis=0)
        
        return {
            'success': True,
            'ensemble_final_embeddings': ensemble_final_embeddings,
            'embedding_shape': ensemble_final_embeddings.shape,
            'input_shape': vol.shape,
            'normalization_mean': normalization_mean,
            'normalization_std': normalization_std
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def extract_local_features(
    preprocessed_pt_path: str,
    models: List[tuple],
    device: torch.device
) -> Dict[str, Any]:
    """Extract ensemble local features from a preprocessed .pt file"""
    try:
        # Load the preprocessed data
        data = torch.load(preprocessed_pt_path, weights_only=False)
        
        # Extract the volume tensor
        if isinstance(data, dict):
            if 'volume' in data:
                vol = data['volume']
            elif 'img' in data:
                vol = data['img']
            elif 'image' in data:
                vol = data['image']
            elif 'data' in data:
                vol = data['data']
            else:
                tensor_keys = [k for k, v in data.items() if isinstance(v, torch.Tensor)]
                if tensor_keys:
                    vol_key = max(tensor_keys, key=lambda k: data[k].dim())
                    vol = data[vol_key]
                else:
                    raise ValueError("No volume tensor found in data dictionary")
        else:
            vol = data
        
        vol = vol.to(device)
        
        all_local_features = []
        
        # Process each checkpoint
        for i, (bb, head) in enumerate(models):
            with torch.no_grad():
                # Extract ResNet backbone features
                backbone_feats = bb(vol)
                
                # Extract local attention features
                local_features = head(backbone_feats)
                all_local_features.append(local_features.cpu())
        
        # Create ensembled local features
        ensemble_local_features = torch.mean(torch.stack(all_local_features), dim=0)
        
        result = {
            'success': True,
            'ensemble_local_features': ensemble_local_features,
            'feature_shape': ensemble_local_features.shape,
            'input_shape': vol.shape
        }
        
        return result
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def create_output_path(
    input_path: str,
    base_input_dir: str,
    base_output_dir: str,
    feature_type: str
) -> str:
    """
    Create output path maintaining directory structure.
    
    Args:
        input_path: Path to preprocessed .pt file
        base_input_dir: Base directory of preprocessed files
        base_output_dir: Base directory for output features
        feature_type: Either 'global' or 'local'
    """
    input_path = Path(input_path)
    base_input_dir = Path(base_input_dir)
    base_output_dir = Path(base_output_dir)
    
    # Get relative path
    relative_path = input_path.relative_to(base_input_dir)
    
    # Create output path
    output_path = base_output_dir / relative_path
    
    # Change filename suffix based on feature type
    original_filename = output_path.name
    extension = '.pt' if feature_type in ['global', 'local'] else '.npy'
    
    if original_filename.endswith('_preprocessed.pt'):
        feature_filename = original_filename.replace(
            '_preprocessed.pt', 
            f'_{feature_type}_features{extension}'
        )
    else:
        feature_filename = original_filename.replace(
            '.pt',
            f'_{feature_type}_features{extension}'
        )
    
    output_path = output_path.parent / feature_filename
    
    return str(output_path)


def process_batch_from_csv(
    csv_path: str,
    checkpoint_dir: str,
    output_dir: str,
    output_csv: str,
    feature_type: str,
    base_preprocessed_dir: Optional[str] = None,
    resume: bool = False
) -> None:
    """
    Batch process feature extraction from CSV file.
    
    Args:
        csv_path: Input CSV with preprocessed file paths
        checkpoint_dir: Directory containing Sybil checkpoint files
        output_dir: Output directory for feature files
        output_csv: Output CSV file path
        feature_type: Either 'global', 'local', or 'final'
        base_preprocessed_dir: Base directory of preprocessed files
        resume: Skip existing feature files
    """
    logger.info(f"🚀 Starting {feature_type.title()} Feature Extraction")
    logger.info(f"Input CSV: {csv_path}")
    logger.info(f"Checkpoint directory: {checkpoint_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Output CSV: {output_csv}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Verify checkpoint directory
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    # Load CSV
    logger.info(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Found {len(df)} patients in CSV")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Detect timepoint columns
    timepoints = []
    for t in ['T0', 'T1', 'T2']:
        vol_col = f'vol{t}'
        if vol_col in df.columns:
            timepoints.append(t)
            # Add feature path column if doesn't exist
            feat_col = f'{feature_type}_feat_{t}_path'
            if feat_col not in df.columns:
                df[feat_col] = None
            
    if not timepoints:
        logger.error("No volT0/volT1/volT2 columns found in CSV")
        return
    
    logger.info(f"Found timepoint columns: {timepoints}")
    
    # Determine base directory for relative paths
    if base_preprocessed_dir is None:
        # Try to infer from first valid path
        for _, row in df.iterrows():
            for t in timepoints:
                vol_col = f'vol{t}'
                if vol_col in df.columns and pd.notna(row[vol_col]):
                    sample_path = Path(row[vol_col])
                    if sample_path.exists():
                        base_preprocessed_dir = sample_path.parent.parent.parent
                        break
            if base_preprocessed_dir:
                break
        
        if base_preprocessed_dir is None:
            logger.warning("Could not infer base directory, using input paths as-is")
            base_preprocessed_dir = "/"
    
    logger.info(f"Base preprocessed directory: {base_preprocessed_dir}")
    
    # Load models
    logger.info("Loading models...")
    
    if feature_type == 'global':
        models = []
        for i, ckpt in enumerate(CHECKPOINT_FILENAMES):
            ckpt_path = os.path.join(checkpoint_dir, ckpt)
            bb = SybilBackbone().to(device).eval()
            load_backbone_weights(bb, ckpt_path)
            models.append(bb)
    else:  # local features or final embeddings
        models = []
        for i, ckpt in enumerate(CHECKPOINT_FILENAMES):
            ckpt_path = os.path.join(checkpoint_dir, ckpt)
            bb = SybilBackbone().to(device).eval()
            load_backbone_weights(bb, ckpt_path)
            head = SybilHead().to(device).eval()
            load_head_weights(head, ckpt_path)
            models.append((bb, head))
    
    # Statistics
    processed_count = 0
    error_count = 0
    skipped_count = 0
    
    # Process each patient
    logger.info(f"\n🔄 Processing {len(df)} patients...")
    
    with tqdm(total=len(df), desc="Processing patients") as pbar:
        for idx, row in df.iterrows():
            patient_id = row.get('pid', 'unknown')
            
            # Process each timepoint
            for timepoint in timepoints:
                vol_col = f'vol{timepoint}'
                feat_col = f'{feature_type}_feat_{timepoint}_path'
                
                # Skip if no preprocessed file
                if pd.isna(row[vol_col]):
                    continue
                
                input_path = row[vol_col]
                
                # Skip if input doesn't exist
                if not os.path.exists(input_path):
                    logger.warning(f"Input not found: {input_path}")
                    continue
                
                # Create output path
                output_path = create_output_path(
                    input_path,
                    str(base_preprocessed_dir),
                    output_dir,
                    feature_type
                )
                
                # Create output directory
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Skip if resume mode and file exists
                if resume and os.path.exists(output_path):
                    df.at[idx, feat_col] = output_path
                    skipped_count += 1
                    continue
                
                # Extract features
                if feature_type == 'global':
                    result = extract_resnet_features(input_path, models, device)
                    
                    if result['success']:
                        # Save features for global features
                        feature_data = {
                            'ensemble_features': result['ensemble_features'],
                            'feature_shape': result['feature_shape'],
                            'input_volume_shape': result['input_shape'],
                            'num_checkpoints': len(CHECKPOINT_FILENAMES),
                            'checkpoint_paths': CHECKPOINT_FILENAMES,
                            'feature_type': 'global_resnet',
                            'original_preprocessed_path': input_path,
                            'timepoint': timepoint
                        }
                        torch.save(feature_data, output_path)
                        df.at[idx, feat_col] = output_path
                        processed_count += 1
                    
                elif feature_type == 'final':  # final embeddings
                    result = extract_final_embeddings(
                        input_path, models, device
                    )
                    
                    if result['success']:
                        # Save final embeddings as .npy file
                        np.save(output_path, result['ensemble_final_embeddings'])
                        df.at[idx, feat_col] = output_path
                        processed_count += 1
                        
                else:  # local features
                    result = extract_local_features(
                        input_path, models, device
                    )
                    
                    if result['success']:
                        # Save features
                        feature_data = {
                            'ensemble_local_features': result['ensemble_local_features'],
                            'feature_shape': result['feature_shape'],
                            'input_volume_shape': result['input_shape'],
                            'num_checkpoints': len(CHECKPOINT_FILENAMES),
                            'checkpoint_paths': CHECKPOINT_FILENAMES,
                            'extraction_stage': 'multi_image_hidden_1',
                            'feature_type': 'spatially_attended_per_slice',
                            'original_preprocessed_path': input_path,
                            'timepoint': timepoint,
                            'description': 'Spatially attended features per slice'
                        }
                        torch.save(feature_data, output_path)
                        
                        # Update CSV
                        df.at[idx, feat_col] = output_path
                        processed_count += 1
                        
                if not result['success']:
                    logger.error(
                        f"❌ Error processing {patient_id} {timepoint}: {result['error']}"
                    )
                    error_count += 1
            
            # Update progress
            pbar.set_postfix({
                'Processed': processed_count,
                'Errors': error_count,
                'Skipped': skipped_count
            })
            pbar.update(1)
            
            # Save progress periodically
            if (idx + 1) % 50 == 0:
                df.to_csv(output_csv, index=False)
                logger.info(f"💾 Progress saved at patient {idx+1}")
    
    # Final save
    df.to_csv(output_csv, index=False)
    
    logger.info(f"\n✅ Processing completed!")
    logger.info(f"📊 Summary:")
    logger.info(f"  - Total patients: {len(df)}")
    logger.info(f"  - Successfully processed: {processed_count}")
    logger.info(f"  - Errors: {error_count}")
    logger.info(f"  - Skipped (resume mode): {skipped_count}")
    logger.info(f"  - Output directory: {output_dir}")
    logger.info(f"  - Output CSV: {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract features from preprocessed CT volumes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
---------
# Extract ResNet (global) features:
python extract_features.py global \\
    --csv preprocessed_output.csv \\
    --checkpoint_dir /path/to/sybil/checkpoints \\
    --output_dir ./features \\
    --output_csv features_output.csv

# Extract local attention features:
python extract_features.py local \\
    --csv preprocessed_output.csv \\
    --checkpoint_dir /path/to/sybil/checkpoints \\
    --output_dir ./features \\
    --output_csv features_output.csv

# Extract final embeddings:
python extract_features.py final \\
    --csv preprocessed_output.csv \\
    --checkpoint_dir /path/to/sybil/checkpoints \\
    --output_dir ./features \\
    --output_csv features_output.csv

# Resume interrupted processing:
python extract_features.py global \\
    --csv preprocessed_output.csv \\
    --checkpoint_dir /path/to/sybil/checkpoints \\
    --output_dir ./features \\
    --output_csv features_output.csv \\
    --resume
        """
    )
    
    parser.add_argument(
        'feature_type',
        choices=['global', 'local', 'final'],
        help="Type of features to extract ('global' for ResNet, 'local' for attention, or 'final' for final embeddings)"
    )

    
    parser.add_argument(
        '--csv', '-c',
        required=True,
        help='Input CSV file with preprocessed file paths'
    )
    parser.add_argument(
        '--checkpoint_dir', '-ckpt',
        required=True,
        help='Directory containing Sybil checkpoint files'
    )
    parser.add_argument(
        '--output_dir', '-d',
        required=True,
        help='Output directory for feature files'
    )
    parser.add_argument(
        '--output_csv', '-o',
        required=True,
        help='Output CSV file with feature file paths'
    )
    parser.add_argument(
        '--base_dir', '-b',
        default=None,
        help='Base directory of preprocessed files (auto-detected if not provided)'
    )
    parser.add_argument(
        '--resume', '-r',
        action='store_true',
        help='Resume processing (skip existing feature files)'
    )
    
    args = parser.parse_args()
    
    try:
        
        process_batch_from_csv(
            csv_path=args.csv,
            checkpoint_dir=args.checkpoint_dir,
            output_dir=args.output_dir,
            output_csv=args.output_csv,
            feature_type=args.feature_type,
            base_preprocessed_dir=args.base_dir,
            resume=args.resume
        )
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()