#!/usr/bin/env python3
"""
Grad-CAM Demo Script for Spectral Data Analysis

This script demonstrates how to use the Grad-CAM interpretability tools
with trained spectral analysis models.

Usage:
    python gradcam_demo.py --model_path path/to/model.pth --data_path path/to/data.npz
    python gradcam_demo.py --config config.json --checkpoint model.pth --data_path data.npz
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.interpretability.model_wrapper import SpectralModelAnalyzer, create_analyzer_from_checkpoint
from src.regression.models.unet import UNET
from src.regression.models.dualunet import DualUNet
from src.regression.models.dualsimplecnn import DualSimpleCNN
from src.utils.data_io.load_data import load_spectral_data


def load_sample_data(data_path: str, sample_idx: int = 0) -> tuple:
    """
    Load sample spectral data for demonstration.
    
    Args:
        data_path: Path to data file
        sample_idx: Index of sample to load
        
    Returns:
        Tuple of (input_data, wavelengths, metadata)
    """
    if data_path.endswith('.npz'):
        data = np.load(data_path)
        
        # Try different common key names
        input_keys = ['data', 'X', 'input', 'spectra', 'images']
        input_data = None
        
        for key in input_keys:
            if key in data:
                input_data = data[key]
                break
        
        if input_data is None:
            # Use first array found
            input_data = data[list(data.keys())[0]]
        
        # Get wavelengths if available
        wavelengths = None
        wavelength_keys = ['wavelengths', 'wl', 'lambda', 'frequencies']
        for key in wavelength_keys:
            if key in data:
                wavelengths = data[key]
                break
        
        # Select sample
        if input_data.ndim > 2:
            input_data = input_data[sample_idx]
        
        metadata = {
            'file_keys': list(data.keys()),
            'data_shape': input_data.shape,
            'sample_idx': sample_idx
        }
        
        return input_data, wavelengths, metadata
    
    elif data_path.endswith('.npy'):
        input_data = np.load(data_path)
        if input_data.ndim > 2:
            input_data = input_data[sample_idx]
        
        return input_data, None, {'data_shape': input_data.shape, 'sample_idx': sample_idx}
    
    else:
        raise ValueError(f"Unsupported data format: {data_path}")


def create_demo_model(model_type: str = 'unet') -> torch.nn.Module:
    """
    Create a demo model for testing (if no checkpoint provided).
    
    Args:
        model_type: Type of model to create
        
    Returns:
        PyTorch model
    """
    if model_type.lower() == 'unet':
        model = UNET(is_norm=True, in_channels=1, out_channels=2, features=[16, 32, 64])
    elif model_type.lower() == 'dualunet':
        model = DualUNet(is_norm=True, in_channels=1, out_channels=2, branch_number=2)
    elif model_type.lower() == 'cnn':
        model = DualSimpleCNN(in_channels=1, out_channels=2)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Initialize with random weights for demo
    print(f"Created demo {model_type} model with random weights")
    return model


def main():
    parser = argparse.ArgumentParser(description='Grad-CAM Demo for Spectral Analysis')
    
    # Model loading options
    parser.add_argument('--model_path', type=str, help='Path to saved model checkpoint')
    parser.add_argument('--config_path', type=str, help='Path to model configuration file')
    parser.add_argument('--model_type', type=str, default='unet', 
                       choices=['unet', 'dualunet', 'cnn'], help='Model type for demo model')
    
    # Data options
    parser.add_argument('--data_path', type=str, required=True, help='Path to spectral data')
    parser.add_argument('--sample_idx', type=int, default=0, help='Sample index to analyze')
    parser.add_argument('--wavelengths_file', type=str, help='Path to wavelengths file')
    
    # Analysis options
    parser.add_argument('--target_output', type=int, default=0, 
                       help='Target output index for multi-output models')
    parser.add_argument('--layers', type=str, nargs='+', 
                       help='Specific layers to analyze (if not provided, auto-detect)')
    
    # Output options
    parser.add_argument('--output_dir', type=str, default='gradcam_results', 
                       help='Output directory for results')
    parser.add_argument('--sample_name', type=str, default='demo_sample', 
                       help='Name for the analyzed sample')
    parser.add_argument('--save_figures', action='store_true', 
                       help='Save visualization figures')
    
    # Device options
    parser.add_argument('--device', type=str, default='auto', 
                       choices=['auto', 'cpu', 'cuda'], help='Device to use')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load or create model
    if args.model_path:
        print(f"Loading model from {args.model_path}")
        
        if args.config_path:
            # Load from config and checkpoint
            with open(args.config_path, 'r') as f:
                config = json.load(f)
            
            model_name = config.get('model_name', 'UNET').lower()
            model_params = config.get('model_params', {})
            
            if model_name == 'unet':
                model = UNET(**model_params)
            elif model_name == 'dualunet':
                model = DualUNet(**model_params)
            elif model_name == 'dualsimplecnn':
                model = DualSimpleCNN(**model_params)
            else:
                raise ValueError(f"Unsupported model in config: {model_name}")
            
            # Load checkpoint
            checkpoint = torch.load(args.model_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            # Try to load directly
            model = torch.load(args.model_path, map_location='cpu')
            if isinstance(model, dict):
                # It's a state dict, need to create model first
                print("Checkpoint contains state dict, creating demo model...")
                model = create_demo_model(args.model_type)
                checkpoint = torch.load(args.model_path, map_location='cpu')
                model.load_state_dict(checkpoint)
    else:
        print("No model path provided, creating demo model...")
        model = create_demo_model(args.model_type)
    
    # Create analyzer
    analyzer = SpectralModelAnalyzer(model, model_type=args.model_type, device=device)
    
    # Override target layers if specified
    if args.layers:
        analyzer.target_layers = args.layers
        analyzer.gradcam.target_layers = args.layers
        print(f"Using specified layers: {args.layers}")
    
    # Load data
    print(f"Loading data from {args.data_path}")
    input_data, wavelengths, metadata = load_sample_data(args.data_path, args.sample_idx)
    
    # Load wavelengths from separate file if provided
    if args.wavelengths_file:
        wavelengths = np.load(args.wavelengths_file)
        print(f"Loaded wavelengths from {args.wavelengths_file}")
    
    print(f"Data shape: {input_data.shape}")
    print(f"Wavelengths: {'Available' if wavelengths is not None else 'Not available'}")
    
    # Analyze sample
    print("\\nRunning Grad-CAM analysis...")
    analysis_results = analyzer.analyze_sample(
        input_data, 
        target_output_idx=args.target_output,
        wavelengths=wavelengths
    )
    
    # Print analysis summary
    print("\\n" + "="*50)
    print("ANALYSIS SUMMARY")
    print("="*50)
    
    model_info = analyzer.get_model_info()
    print(f"Model: {model_info['model_class']}")
    print(f"Parameters: {model_info['total_parameters']:,}")
    print(f"Target layers: {len(analyzer.target_layers)}")
    
    # Layer importance ranking
    importance_ranking = analysis_results['importance_ranking']
    print("\\nLayer Importance Ranking:")
    for i, (layer_name, score) in enumerate(importance_ranking[:5]):
        print(f"  {i+1}. {layer_name}: {score:.4f}")
    
    # CAM statistics
    cam_results = analysis_results['cam_results']
    print("\\nCAM Statistics:")
    for layer_name, cam_data in list(cam_results.items())[:3]:
        print(f"  {layer_name}:")
        print(f"    Peak intensity: {cam_data['peak_intensity']:.4f}")
        print(f"    Mean intensity: {cam_data['mean_intensity']:.4f}")
        print(f"    CAM shape: {cam_data['cam_shape']}")
    
    # Create visualizations
    print("\\nCreating visualizations...")
    
    output_dir = args.output_dir if args.save_figures else None
    figures = analyzer.visualize_analysis(
        input_data,
        analysis_results,
        wavelengths=wavelengths,
        save_dir=output_dir,
        sample_name=args.sample_name
    )
    
    if args.save_figures:
        print(f"\\nSaved visualizations to {args.output_dir}")
    else:
        print("\\nDisplaying visualizations...")
        plt.show()
    
    # Additional analysis for spectral data
    if wavelengths is not None and input_data.ndim == 1:
        print("\\n" + "="*50)
        print("SPECTRAL ANALYSIS")
        print("="*50)
        
        # Find most important wavelengths
        for layer_name, cam_data in list(cam_results.items())[:2]:
            if 'important_wavelengths' in cam_data:
                important_wl = cam_data['important_wavelengths']
                print(f"\\n{layer_name} - Important wavelengths:")
                for wl in important_wl[:5]:
                    print(f"  {wl:.1f} nm")
            
            if 'dominant_wavelength' in cam_data:
                dom_wl = cam_data['dominant_wavelength']
                print(f"  Dominant wavelength: {dom_wl:.1f} nm")
    
    print("\\n" + "="*50)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("="*50)


if __name__ == '__main__':
    main()