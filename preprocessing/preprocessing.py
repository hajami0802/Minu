#!/usr/bin/env python3
"""
DICOM Preprocessing Tool: Sybil-Compatible
==========================================

A unified tool for preprocessing LDCT DICOM series for longitudinal time-point scans, compatible with Sybil lung cancer risk model.

Supports both single series preprocessing and batch processing from CSV files with multi-timepoint support.

Author: Hanieh Ajami
Repository: https://github.com/hajami0802/Preprocessing
License: MIT

Usage:
------
# Preprocess a single DICOM series:
python sybil_dicom_preprocessor.py single --input /path/to/dicom/folder --output /path/to/output.pt

# Batch process multiple series from a CSV:
python sybil_dicom_preprocessor.py batch --csv input.csv --output_dir ./preprocessed --output_csv results.csv

Requirements:
-------------
- torch
- numpy
- pydicom
- torchio
- opencv-python (cv2)
- pandas (for batch processing)

## Preprocessing Details

The tool applies the following transformations to match Sybil's requirements:

1. **DICOM Loading**: Loads raw DICOM files with modality LUT applied
2. **Windowing**: Applies lung window (center: -600 HU, width: 1500 HU)
3. **Resizing**: Resizes to 256×256 pixels per slice
4. **Normalization**: Z-score normalization using Sybil's fixed mean (128.1722) and std (87.1849)
5. **Channel Replication**: Converts grayscale to 3-channel (RGB) format
6. **Resampling**: Resamples to 0.703125 × 0.703125 × 2.5 mm voxel spacing
7. **Slice Adjustment**: Pads or crops to exactly 200 slices

### Thickness Validation

The preprocessor accepts DICOM series with slice thickness ≤ 10mm:

- ✅ Accepts scans with thickness ≤ 10mm
- ❌ Rejects scans > 10mm
- All accepted scans are automatically resampled to 2.5mm spacing

This allows the tool to work with a wide variety of CT scans from different institutions while ensuring consistent output format for the Sybil model.

"""
import torch
import numpy as np
import pydicom
import torchio as tio
import cv2
import os
import re
import argparse
import logging
import json
from datetime import datetime
from typing import List, Optional, Tuple, NamedTuple, Dict, Any
from pathlib import Path
from pydicom.pixel_data_handlers.util import apply_modality_lut

# Try importing pandas for batch processing
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: pandas not installed. Batch processing will not be available.")

# =============================================================================
# SYBIL MODEL CONSTANTS
# =============================================================================
VOXEL_SPACING = (0.703125, 0.703125, 2.5)   # Target resampling (mm)
IMG_SIZE = [256, 256]                        # Width, Height
NUM_SLICES = 200                             # Number of slices in final volume
NUM_CHANNELS = 3                             # RGB channels
MAX_SLICE_THICKNESS = 10.0                   # Maximum allowed slice thickness (mm)
NORMALIZATION_MEAN = 128.1722                # Sybil's fixed normalization mean
NORMALIZATION_STD = 87.1849                  # Sybil's fixed normalization std

# =============================================================================
# LOGGING SETUP
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================
class PreprocessingConfig:
    """Configuration for DICOM preprocessing"""
    
    def __init__(self, 
                 img_size: List[int] = None,
                 num_images: int = NUM_SLICES,
                 num_chan: int = NUM_CHANNELS,
                 max_slice_thickness: float = MAX_SLICE_THICKNESS,
                 voxel_spacing: Tuple[float, float, float] = VOXEL_SPACING,
                 normalization_mean: float = NORMALIZATION_MEAN,
                 normalization_std: float = NORMALIZATION_STD):
        """
        Initialize preprocessing configuration.
        
        Args:
            img_size: Target image size [width, height]
            num_images: Number of slices in output volume
            num_chan: Number of channels (typically 3 for RGB)
            max_slice_thickness: Maximum allowed slice thickness (mm), default 10mm
            voxel_spacing: Target voxel spacing (mm)
            normalization_mean: Mean for normalization
            normalization_std: Std for normalization
        """
        self.img_size = img_size if img_size is not None else IMG_SIZE
        self.num_images = num_images
        self.num_chan = num_chan
        self.voxel_spacing = voxel_spacing
        self.normalization_mean = normalization_mean
        self.normalization_std = normalization_std
        self.max_slice_thickness = max_slice_thickness


class Meta(NamedTuple):
    """Metadata extracted from DICOM series"""
    paths: List[str]
    thickness: float
    pixel_spacing: List[float]
    manufacturer: str
    slice_positions: List[float]
    voxel_spacing: torch.Tensor


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def order_slices_by_position(img_paths: List[str]) -> List[str]:
    """
    Order DICOM slices by anatomical position using DICOM metadata.
    Sorts by SliceLocation or ImagePositionPatient[2] (Z-coordinate) in INCREASING order.
    This ensures abdomen → clavicles ordering (inferior to superior).
    """
    path_position_pairs = []
    
    for path in img_paths:
        try:
            dcm = pydicom.dcmread(path, stop_before_pixels=True)
            position = None
            
            # Try SliceLocation first (most reliable)
            if hasattr(dcm, 'SliceLocation') and dcm.SliceLocation is not None:
                position = float(dcm.SliceLocation)
            # Fall back to ImagePositionPatient Z coordinate
            elif hasattr(dcm, 'ImagePositionPatient') and dcm.ImagePositionPatient:
                position = float(dcm.ImagePositionPatient[2])
            
            if position is not None:
                path_position_pairs.append((path, position))
            else:
                raise ValueError(f"No position data (SliceLocation or ImagePositionPatient) found in {path}")
                
        except Exception as e:
            logger.error(f"Error reading position from {path}: {e}")
            raise ValueError(f"Could not extract position data from {path}: {e}")
    
    if len(path_position_pairs) != len(img_paths):
        raise ValueError("Could not extract position data from all DICOM files")
    
    # Sort by position in INCREASING order (abdomen→clavicles)
    # Lower Z values are inferior (abdomen), Higher Z values are superior (head).
    sorted_pairs = sorted(path_position_pairs, key=lambda x: x[1], reverse=False)
    
    logger.info(f"Anatomical ordering: {len(sorted_pairs)} slices from Z={sorted_pairs[0][1]:.1f} to Z={sorted_pairs[-1][1]:.1f}")
    return [path for path, _ in sorted_pairs]


def apply_windowing(image: np.ndarray, center: float, width: float, bit_size: int = 16) -> np.ndarray:
    """Apply window center/width to CT image"""
    y_min = 0
    y_max = 2 ** bit_size - 1
    y_range = y_max - y_min

    c = center - 0.5
    w = width - 1

    image = image.astype(np.float64)
    below = image <= (c - w / 2)
    above = image > (c + w / 2)
    between = np.logical_and(~below, ~above)

    image[below] = y_min
    image[above] = y_max
    if between.any():
        image[between] = ((image[between] - c) / w + 0.5) * y_range + y_min
    return image


def is_localizer_scan(dicom_folder: Path) -> Tuple[bool, str]:
    """
    Check if a DICOM folder contains a localizer/scout scan.
    
    Returns:
        Tuple of (is_localizer, reason)
    """
    folder_name = dicom_folder.name.lower()
    localizer_keywords = ['localizer', 'scout', 'topogram', 'surview', 'scanogram']
    
    # Check folder name
    if any(keyword in folder_name for keyword in localizer_keywords):
        return True, f"Folder name contains localizer keyword: {folder_name}"
    
    try:
        dcm_files = list(dicom_folder.glob("*.dcm"))
        if not dcm_files:
            return False, "No DICOM files found"
        
        # Check first few DICOM files for localizer metadata
        sample_files = dcm_files[:min(3, len(dcm_files))]
        for dcm_file in sample_files:
            try:
                dcm = pydicom.dcmread(dcm_file, stop_before_pixels=True)
                
                # Check ImageType field
                if hasattr(dcm, 'ImageType'):
                    image_type_str = ' '.join(str(val).lower() for val in dcm.ImageType)
                    if any(keyword in image_type_str for keyword in localizer_keywords):
                        return True, f"ImageType indicates localizer: {dcm.ImageType}"
                
                # Check SeriesDescription field
                if hasattr(dcm, 'SeriesDescription'):
                    if any(keyword in dcm.SeriesDescription.lower() for keyword in localizer_keywords):
                        return True, f"SeriesDescription indicates localizer: {dcm.SeriesDescription}"
            except Exception as e:
                logger.debug(f"Error reading DICOM metadata from {dcm_file}: {e}")
                continue
    except Exception as e:
        logger.warning(f"Error checking localizer status for {dicom_folder}: {e}")
    
    return False, "Not a localizer scan"


# =============================================================================
# PREPROCESSING COMPONENTS
# =============================================================================
class DicomImageLoader:
    """Loads and windows DICOM images"""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.window_center = -600
        self.window_width = 1500

    def load_dicom_image(self, path: str) -> np.ndarray:
        """Load a single DICOM image with windowing applied"""
        dcm = pydicom.dcmread(path)
        dcm_array = apply_modality_lut(dcm.pixel_array, dcm)
        windowed_array = apply_windowing(dcm_array, self.window_center, self.window_width, bit_size=16)
        # Scale from 16-bit to 8-bit range
        windowed_array = (windowed_array / 256.0).astype(np.float32)
        return windowed_array


class ImageResizer:
    """Resizes images to target dimensions"""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.target_width, self.target_height = config.img_size

    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target size"""
        return cv2.resize(
            image,
            dsize=(self.target_width, self.target_height),
            interpolation=cv2.INTER_LINEAR
        )


# =============================================================================
# MAIN PREPROCESSOR
# =============================================================================
class SybilPreprocessor:
    """Main preprocessing class for CT DICOM series"""
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self.dicom_loader = DicomImageLoader(self.config)
        self.resizer = ImageResizer(self.config)
        
        # TorchIO transforms
        self.resample_transform = tio.transforms.Resample(target=self.config.voxel_spacing)
        self.padding_transform = tio.transforms.CropOrPad(
            target_shape=tuple(self.config.img_size + [self.config.num_images]),
            padding_mode=0
        )

    def load_metadata(self, paths: List[str]) -> Meta:
        """Extract metadata from DICOM series"""
        processed_paths = order_slices_by_position(paths)
        slice_positions = list(range(len(processed_paths)))
        
        dcm = pydicom.dcmread(processed_paths[0], stop_before_pixels=True)
        thickness = float(dcm.SliceThickness) if hasattr(dcm, 'SliceThickness') else None
        pixel_spacing = list(map(float, dcm.PixelSpacing)) if hasattr(dcm, 'PixelSpacing') else [1.0, 1.0]
        manufacturer = dcm.Manufacturer if hasattr(dcm, 'Manufacturer') else "Unknown"
        
        if thickness is None:
            raise ValueError("SliceThickness not found in DICOM header")
        
        voxel_spacing_tensor = torch.tensor(pixel_spacing + [thickness, 1])
        
        return Meta(
            paths=processed_paths,
            thickness=thickness,
            pixel_spacing=pixel_spacing,
            manufacturer=manufacturer,
            slice_positions=slice_positions,
            voxel_spacing=voxel_spacing_tensor,
        )

    def validate_series(self, meta: Meta) -> None:
        """Validate that series meets preprocessing requirements"""
        if meta.thickness is None:
            raise ValueError("Slice thickness not found")
        
        # Print thickness information
        logger.info(f"📏 Slice thickness: {meta.thickness}mm")
        if meta.thickness <= self.config.max_slice_thickness:
            logger.info(f"✅ Thickness ≤ {self.config.max_slice_thickness}mm: ACCEPTED")
        else:
            logger.error(f"❌ Thickness > {self.config.max_slice_thickness}mm: REJECTED")
            raise ValueError(f"Slice thickness {meta.thickness}mm > {self.config.max_slice_thickness}mm (excluded)")
        
        if meta.voxel_spacing is None:
            raise ValueError("Voxel spacing not found or not set")

    def load_single_image(self, path: str) -> torch.Tensor:
        """Load and preprocess a single DICOM slice"""
        image = self.dicom_loader.load_dicom_image(path)
        image = self.resizer.resize_image(image)
        tensor = torch.from_numpy(image).float()
        return tensor

    def preprocess_volume(self, paths: List[str]) -> torch.Tensor:
        """
        Preprocess a complete DICOM series into a Sybil-compatible volume.
        
        Args:
            paths: List of paths to DICOM files
            
        Returns:
            Preprocessed volume tensor of shape (1, C, D, H, W)
        """
        logger.info(f"Processing {len(paths)} DICOM files...")
        
        # Load metadata and validate
        meta = self.load_metadata(paths)
        logger.info(f"Series info - Thickness: {meta.thickness}mm, Pixel spacing: {meta.pixel_spacing}mm")
        self.validate_series(meta)

        # Load all slices
        images = []
        for i, path in enumerate(meta.paths):
            if i % 50 == 0:
                logger.info(f"Loading image {i+1}/{len(meta.paths)}")
            image_tensor = self.load_single_image(path)
            images.append(image_tensor)

        # Stack into volume (T, H, W)
        volume = torch.stack(images, dim=0)
        
        # Normalize using Sybil's fixed mean/std
        volume = (volume - self.config.normalization_mean) / self.config.normalization_std

        # Add channel dim: (T, 1, H, W)
        volume = volume.unsqueeze(1)
        # Repeat channels: (T, 3, H, W)
        volume = volume.repeat(1, self.config.num_chan, 1, 1)
        # Permute to (C, T, H, W)
        volume = volume.permute(1, 0, 2, 3)

        # TorchIO expects (C, H, W, D)
        tio_image = tio.ScalarImage(
            affine=torch.diag(meta.voxel_spacing),
            tensor=volume.permute(0, 2, 3, 1),  # (C, H, W, D)
        )
        
        # Resample
        logger.info(f"Resampling to {self.config.voxel_spacing}mm voxel spacing...")
        tio_image = self.resample_transform(tio_image)
        
        # Pad/truncate to 200 slices
        logger.info(f"Adjusting to {self.config.num_images} slices...")
        tio_image = self.padding_transform(tio_image)
        
        # Back to (C, D, H, W)
        volume = tio_image.data.permute(0, 3, 1, 2)
        
        # Add batch dim (1, C, D, H, W)
        volume = volume.unsqueeze(0)
        
        logger.info(f"✅ Preprocessing complete! Final shape: {volume.shape}")
        return volume


# =============================================================================
# SINGLE SERIES PROCESSING
# =============================================================================
def preprocess_single_series(
    input_path: str,
    output_path: str
) -> bool:
    """
    Preprocess a single DICOM series.
    
    Args:
        input_path: Path to directory containing DICOM files
        output_path: Path where preprocessed .pt file will be saved
        
    Returns:
        True if successful, False otherwise
    """
    try:
        input_dir = Path(input_path)
        if not input_dir.exists():
            logger.error(f"Input directory does not exist: {input_path}")
            return False
        
        # Check if it's a localizer scan
        is_localizer, localizer_reason = is_localizer_scan(input_dir)
        if is_localizer:
            logger.warning(f"🚫 LOCALIZER SCAN DETECTED: NOT ACCEPTED")
            logger.warning(f"   Reason: {localizer_reason}")
            return False
        
        # Find all DICOM files
        dicom_files = list(input_dir.glob("*.dcm"))
        if not dicom_files:
            logger.error(f"No DICOM files found in {input_path}")
            return False
        
        logger.info(f"Found {len(dicom_files)} DICOM files")
        
        # Create preprocessor
        config = PreprocessingConfig()
        preprocessor = SybilPreprocessor(config)
        
        # Preprocess volume
        volume = preprocessor.preprocess_volume([str(f) for f in dicom_files])
        
        # Save output - use input folder name if output path is not specified with a filename
        output_file = Path(output_path)
        
        # If output_path is just a directory or doesn't have .pt extension, 
        # create filename from input directory name
        if output_file.is_dir() or output_file.suffix != '.pt':
            # Get the input folder name
            input_folder_name = input_dir.name
            output_filename = f"{input_folder_name}_preprocessed.pt"
            if output_file.is_dir():
                output_file = output_file / output_filename
            else:
                # If it's a file path without .pt, append _preprocessed.pt
                output_file = output_file.parent / output_filename
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'volume': volume,
            'normalization_mean': NORMALIZATION_MEAN,
            'normalization_std': NORMALIZATION_STD,
            'original_path': str(input_dir),
            'num_original_slices': len(dicom_files)
        }, output_file)
        
        logger.info(f"✅ Saved preprocessed volume to: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Preprocessing failed: {e}", exc_info=True)
        return False


# =============================================================================
# BATCH PROCESSING WITH TIMEPOINTS
# =============================================================================
def process_single_timepoint(
    pid: str,
    timepoint: str,
    dicom_path: str,
    output_dir: Path,
    preprocessor: SybilPreprocessor,
    processing_log: Dict[str, Any]
) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Process a single timepoint for a patient.
    
    Returns:
        Tuple of (output_path, timepoint_log)
    """
    timepoint_log = {
        'timepoint': timepoint,
        'dicom_path': dicom_path,
        'status': 'pending',
        'reason': None,
        'input_info': {},
        'output_info': {}
    }
    
    try:
        # Check if input exists
        if not Path(dicom_path).exists():
            timepoint_log['status'] = 'failed'
            timepoint_log['reason'] = f"DICOM path does not exist: {dicom_path}"
            logger.error(f"❌ {timepoint_log['reason']}")
            return None, timepoint_log
        
        # Check if it's a localizer scan
        is_localizer, localizer_reason = is_localizer_scan(Path(dicom_path))
        if is_localizer:
            timepoint_log['status'] = 'skipped'
            timepoint_log['reason'] = f"Localizer scan: {localizer_reason}"
            logger.warning(f"🚫 LOCALIZER SCAN DETECTED: NOT ACCEPTED")
            logger.warning(f"   Reason: {localizer_reason}")
            return None, timepoint_log
        
        # Find DICOM files
        dicom_files = list(Path(dicom_path).glob("*.dcm"))
        if not dicom_files:
            timepoint_log['status'] = 'failed'
            timepoint_log['reason'] = f"No DICOM files found in {dicom_path}"
            logger.error(f"❌ {timepoint_log['reason']}")
            return None, timepoint_log
        
        logger.info(f"Found {len(dicom_files)} DICOM files")
        timepoint_log['input_info']['num_dicom_files'] = len(dicom_files)
        
        # Load metadata to get thickness info before preprocessing
        try:
            meta = preprocessor.load_metadata([str(f) for f in dicom_files])
            timepoint_log['input_info']['slice_thickness_mm'] = float(meta.thickness)
            timepoint_log['input_info']['pixel_spacing_mm'] = meta.pixel_spacing
            timepoint_log['input_info']['manufacturer'] = meta.manufacturer
            
            # Check thickness acceptance
            if meta.thickness <= 2.5:
                timepoint_log['input_info']['thickness_status'] = 'Standard NLST (≤2.5mm)'
            elif meta.thickness <= 10.0:
                timepoint_log['input_info']['thickness_status'] = 'External validation (2.5-10mm)'
            else:
                timepoint_log['status'] = 'rejected'
                timepoint_log['reason'] = f"Slice thickness {meta.thickness}mm exceeds 10mm threshold"
                logger.error(f"❌ {timepoint_log['reason']}")
                return None, timepoint_log
                
        except Exception as e:
            timepoint_log['status'] = 'failed'
            timepoint_log['reason'] = f"Failed to load metadata: {str(e)}"
            logger.error(f"❌ {timepoint_log['reason']}")
            return None, timepoint_log
        
        # Preprocess
        volume = preprocessor.preprocess_volume([str(f) for f in dicom_files])
        
        # Calculate output statistics
        volume_np = volume.cpu().numpy()
        timepoint_log['output_info']['shape'] = list(volume.shape)
        timepoint_log['output_info']['mean'] = float(np.mean(volume_np))
        timepoint_log['output_info']['std'] = float(np.std(volume_np))
        timepoint_log['output_info']['min'] = float(np.min(volume_np))
        timepoint_log['output_info']['max'] = float(np.max(volume_np))
        
        # Save output - recreate the full directory structure
        dicom_path_obj = Path(dicom_path)
        path_parts = dicom_path_obj.parts
        
        # Find where pid appears in the path
        pid_index = None
        for i, part in enumerate(path_parts):
            if part == str(pid):
                pid_index = i
                break
        
        # If pid found, recreate structure from pid onwards
        if pid_index is not None:
            relative_structure = Path(*path_parts[pid_index:])
            output_file_dir = output_dir / relative_structure.parent
        else:
            output_file_dir = output_dir / str(pid)
        
        # Create the directory structure
        output_file_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename with _preprocessed suffix
        input_folder_name = dicom_path_obj.name
        output_filename = f"{input_folder_name}_preprocessed.pt"
        output_file = output_file_dir / output_filename
        
        torch.save({
            'volume': volume,
            'normalization_mean': NORMALIZATION_MEAN,
            'normalization_std': NORMALIZATION_STD,
            'original_path': str(dicom_path),
            'num_original_slices': len(dicom_files),
            'patient_id': str(pid),
            'timepoint': timepoint,
            'slice_thickness': float(meta.thickness),
            'pixel_spacing': meta.pixel_spacing
        }, output_file)
        
        timepoint_log['status'] = 'success'
        timepoint_log['output_info']['preprocessed_path'] = str(output_file)
        logger.info(f"✅ Successfully saved to: {output_file}")
        
        return str(output_file), timepoint_log
        
    except Exception as e:
        timepoint_log['status'] = 'failed'
        timepoint_log['reason'] = f"Preprocessing error: {str(e)}"
        logger.error(f"❌ Failed to process {timepoint}: {e}", exc_info=True)
        return None, timepoint_log


def preprocess_batch(
    csv_path: str,
    output_dir: str,
    output_csv: str,
    resume_from: Optional[str] = None
) -> None:
    """
    Batch preprocess multiple DICOM series from a CSV file with timepoint support.
    
    CSV Format:
    -----------
    The input CSV must have the following columns:
    - pid: Patient/Series ID
    - T0, T1, T2: Flags (1 or 0) indicating which timepoints to process
    - dicom_pathT0, dicom_pathT1, dicom_pathT2: Paths to DICOM directories
    - volT0, volT1, volT2: Will be populated with preprocessed file paths
    
    Additionally creates a JSON log file with detailed processing information.
    
    Args:
        csv_path: Path to input CSV file
        output_dir: Directory where preprocessed files will be saved
        output_csv: Path to output CSV file with results
        resume_from: Patient ID to resume from (optional)
    """
    if not PANDAS_AVAILABLE:
        logger.error("Pandas is required for batch processing. Please install: pip install pandas")
        return
    
    try:
        # Load CSV
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded CSV with {len(df)} entries")
        
        # Validate required columns
        required_cols = ['pid']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"CSV is missing required columns: {missing_cols}")
            logger.error(f"Required columns: {required_cols}")
            return
        
        # Detect available timepoints
        timepoints = []
        for t in ['T0', 'T1', 'T2']:
            if t in df.columns and f'dicom_path{t}' in df.columns:
                timepoints.append(t)
                # Add vol column if it doesn't exist
                if f'vol{t}' not in df.columns:
                    df[f'vol{t}'] = None
        
        if not timepoints:
            logger.error("No valid timepoint columns found (need T0/T1/T2 flags and corresponding dicom_pathT0/T1/T2)")
            return
        
        logger.info(f"Found timepoint columns: {timepoints}")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Normalize output_csv: allow passing a directory
        # If the user passed a directory for --output_csv, create a sensible default filename inside it.
        output_csv_path = Path(output_csv)
        if output_csv_path.exists() and output_csv_path.is_dir():
            output_csv_path = output_csv_path / "preprocessing_results.csv"
        else:
            # Ensure parent directory exists for the requested CSV file
            output_csv_path.parent.mkdir(parents=True, exist_ok=True)

        # Create JSON log file path
        json_log_path = output_path / "preprocessing_log.json"
        processing_logs = {}
        
        # Load existing logs if resuming
        if json_log_path.exists() and resume_from:
            with open(json_log_path, 'r') as f:
                processing_logs = json.load(f)
            logger.info(f"Loaded existing processing logs from {json_log_path}")
        
        # Create preprocessor
        config = PreprocessingConfig()
        preprocessor = SybilPreprocessor(config)
        
        # Statistics
        total_processed = 0
        total_failed = 0
        total_skipped = 0
        
        # Find starting index if resuming
        start_idx = 0
        if resume_from:
            try:
                start_idx = df[df['pid'] == resume_from].index[0]
                logger.info(f"Resuming from patient {resume_from} (index {start_idx})")
            except IndexError:
                logger.warning(f"Resume patient {resume_from} not found, starting from beginning")
        
        # Process each patient
        for idx, row in df.iterrows():
            if idx < start_idx:
                continue
            
            pid = str(row['pid'])
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {idx+1}/{len(df)}: Patient {pid}")
            logger.info(f"{'='*60}")
            
            # Initialize patient log
            patient_log = {
                'pid': pid,
                'processing_date': datetime.now().isoformat(),
                'timepoints': {}
            }
            
            # Process each timepoint
            for timepoint in timepoints:
                flag_col = timepoint
                dicom_col = f'dicom_path{timepoint}'
                vol_col = f'vol{timepoint}'
                
                # Check if this timepoint should be processed
                if pd.isna(row[flag_col]) or row[flag_col] != 1:
                    logger.info(f"⏭️  {timepoint}: Flag not set (skipping)")
                    patient_log['timepoints'][timepoint] = {
                        'status': 'not_requested',
                        'reason': 'Flag not set to 1'
                    }
                    continue
                
                # Check if already processed
                if pd.notna(row[vol_col]) and Path(str(row[vol_col])).exists():
                    logger.info(f"⏭️  {timepoint}: Already processed, skipping...")
                    patient_log['timepoints'][timepoint] = {
                        'status': 'already_processed',
                        'preprocessed_path': str(row[vol_col])
                    }
                    total_skipped += 1
                    continue
                
                # Get DICOM path
                if pd.isna(row[dicom_col]):
                    logger.warning(f"⚠️  {timepoint}: No DICOM path provided")
                    patient_log['timepoints'][timepoint] = {
                        'status': 'skipped',
                        'reason': 'No DICOM path provided'
                    }
                    total_skipped += 1
                    continue
                
                dicom_path = str(row[dicom_col])
                
                logger.info(f"\n--- Processing {timepoint} ---")
                logger.info(f"DICOM path: {dicom_path}")
                
                # Process this timepoint
                output_file, timepoint_log = process_single_timepoint(
                    pid=pid,
                    timepoint=timepoint,
                    dicom_path=dicom_path,
                    output_dir=output_path,
                    preprocessor=preprocessor,
                    processing_log=patient_log
                )
                
                # Update patient log
                patient_log['timepoints'][timepoint] = timepoint_log
                
                # Update dataframe
                if output_file:
                    df.at[idx, vol_col] = output_file
                    total_processed += 1
                else:
                    if timepoint_log['status'] == 'failed':
                        total_failed += 1
                    else:
                        total_skipped += 1
            
            # Save patient log
            processing_logs[pid] = patient_log
            
            # Save progress periodically (every 5 patients)
            if (idx + 1) % 5 == 0:
                df.to_csv(str(output_csv_path), index=False)
                with open(json_log_path, 'w') as f:
                    json.dump(processing_logs, f, indent=2)
                logger.info(f"💾 Progress saved to {output_csv_path} and {json_log_path}")
        
        # Save final results
        df.to_csv(str(output_csv_path), index=False)
        with open(json_log_path, 'w') as f:
            json.dump(processing_logs, f, indent=2)
        
        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info(f"BATCH PROCESSING COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"✅ Successfully processed: {total_processed} timepoints")
        logger.info(f"❌ Failed: {total_failed} timepoints")
        logger.info(f"⏭️  Skipped: {total_skipped} timepoints")
        logger.info(f"📊 Results saved to: {output_csv_path}")
        logger.info(f"📋 Processing log saved to: {json_log_path}")
        
    except Exception as e:
        logger.error(f"❌ Batch processing failed: {e}", exc_info=True)


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Sybil-Compatible DICOM Preprocessing Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
---------
# Preprocess a single DICOM series:
python sybil_dicom_preprocessor.py single --input /path/to/dicoms --output preprocessed.pt

# Batch process from CSV:
python sybil_dicom_preprocessor.py batch --csv input.csv --output_dir ./preprocessed --output_csv results.csv

CSV Format (for batch processing):
-----------------------------------
The input CSV must contain:
- pid: Patient/Series identifier
- T0, T1, T2: Flags (1 or 0) indicating which timepoints to process
- dicom_pathT0, dicom_pathT1, dicom_pathT2: Paths to DICOM directories

Example CSV:
pid,T0,T1,T2,dicom_pathT0,dicom_pathT1,dicom_pathT2
patient001,1,1,0,/data/patient001/T0,/data/patient001/T1,
patient002,1,0,1,/data/patient002/T0,,/data/patient002/T2

Output CSV will add:
- volT0, volT1, volT2: Paths to saved .pt files
- preprocessing_log.json: Detailed processing information for each patient and timepoint

Note:
-----
The preprocessor accepts DICOM series with slice thickness ≤ 10mm and resamples them to 2.5mm spacing.
        """
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='Processing mode', required=True)
    
    # Single series processing
    single_parser = subparsers.add_parser('single', help='Preprocess a single DICOM series')
    single_parser.add_argument('--input', '-i', required=True, help='Path to directory containing DICOM files')
    single_parser.add_argument('--output', '-o', required=True, help='Output path for preprocessed .pt file')
    
    # Batch processing
    batch_parser = subparsers.add_parser('batch', help='Batch preprocess from CSV with timepoint support')
    batch_parser.add_argument('--csv', '-c', required=True, help='Input CSV file with columns: pid, T0, T1, T2, dicom_pathT0, dicom_pathT1, dicom_pathT2')
    batch_parser.add_argument('--output_dir', '-d', required=True, help='Directory to save preprocessed files and JSON log')
    batch_parser.add_argument('--output_csv', '-oc', required=True, help='Output CSV file with results')
    batch_parser.add_argument('--resume-from', help='Patient ID to resume from (optional)')
    
    args = parser.parse_args()
    
    # Execute appropriate mode
    if args.mode == 'single':
        success = preprocess_single_series(
            input_path=args.input,
            output_path=args.output
        )
        exit(0 if success else 1)
        
    elif args.mode == 'batch':
        preprocess_batch(
            csv_path=args.csv,
            output_dir=args.output_dir,
            output_csv=args.output_csv,
            resume_from=args.resume_from
        )


if __name__ == "__main__":
    main()
