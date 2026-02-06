"""
Model wrapper for integrating Grad-CAM with existing spectral analysis models.
Provides easy-to-use interfaces for UNet, CNN, and other architectures.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import os
import sys

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(project_root)

from .gradcam import SpectralGradCAM, get_model_target_layers
from .visualization import GradCAMVisualizer


class SpectralModelAnalyzer:
    """
    High-level interface for analyzing spectral models with Grad-CAM.
    Supports UNet, CNN, and other architectures used in the project.
    """
    
    def __init__(self, model: nn.Module, model_type: str = 'auto', 
                 device: Optional[torch.device] = None):
        """
        Initialize the model analyzer.
        
        Args:
            model: PyTorch model to analyze
            model_type: Type of model ('unet', 'cnn', 'resnet', 'auto')
            device: Device to run analysis on
        """
        self.model = model
        self.model_type = model_type
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
        
        # Auto-detect target layers
        self.target_layers = get_model_target_layers(model, model_type)
        
        # Initialize Grad-CAM
        self.gradcam = SpectralGradCAM(model, self.target_layers, use_cuda=self.device.type == 'cuda')
        
        # Initialize visualizer
        self.visualizer = GradCAMVisualizer()
        
        print(f"Initialized analyzer for {model.__class__.__name__}")
        print(f"Target layers: {self.target_layers}")
    
    def analyze_sample(self, input_data: Union[np.ndarray, torch.Tensor],
                      target_output_idx: int = 0,
                      wavelengths: Optional[np.ndarray] = None,
                      return_raw: bool = False) -> Dict[str, Any]:
        """
        Analyze a single sample with Grad-CAM.
        
        Args:
            input_data: Input spectral data
            target_output_idx: Target output index for multi-output models
            wavelengths: Wavelength values for spectral data
            return_raw: Whether to return raw CAM values
            
        Returns:
            Dictionary containing analysis results
        """
        # Convert to tensor if needed
        if isinstance(input_data, np.ndarray):
            input_tensor = torch.from_numpy(input_data).float()
        else:
            input_tensor = input_data.float()
        
        # Add batch dimension if needed
        if input_tensor.dim() == 2:  # (H, W) -> (1, 1, H, W)
            input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
        elif input_tensor.dim() == 3:  # (C, H, W) -> (1, C, H, W)
            input_tensor = input_tensor.unsqueeze(0)
        
        # Set spectral information
        if wavelengths is not None:
            self.gradcam.set_spectral_info(wavelengths=wavelengths)
        
        # Generate CAM analysis
        cam_results = self.gradcam.generate_spectral_cam(
            input_tensor, target_output_idx=target_output_idx, return_raw=return_raw
        )
        
        # Get layer importance ranking
        importance_ranking = self.gradcam.get_layer_importance_ranking(
            input_tensor, target_output_idx=target_output_idx
        )
        
        return {
            'cam_results': cam_results,
            'importance_ranking': importance_ranking,
            'input_shape': input_tensor.shape,
            'model_type': self.model_type,
            'target_layers': self.target_layers
        }
    
    def visualize_analysis(self, input_data: Union[np.ndarray, torch.Tensor],
                          analysis_results: Optional[Dict] = None,
                          wavelengths: Optional[np.ndarray] = None,
                          save_dir: Optional[str] = None,
                          sample_name: str = "sample") -> Dict[str, Any]:
        """
        Create visualizations for the analysis.
        
        Args:
            input_data: Original input data
            analysis_results: Pre-computed analysis results (if None, will compute)
            wavelengths: Wavelength values
            save_dir: Directory to save visualizations
            sample_name: Name for the sample (used in filenames)
            
        Returns:
            Dictionary containing matplotlib figures
        """
        if analysis_results is None:
            analysis_results = self.analyze_sample(input_data, wavelengths=wavelengths)
        
        cam_results = analysis_results['cam_results']
        importance_ranking = analysis_results['importance_ranking']
        
        # Convert input to numpy if needed
        if isinstance(input_data, torch.Tensor):
            input_np = input_data.cpu().numpy()
        else:
            input_np = input_data
        
        # Remove batch dimension for visualization
        if input_np.ndim == 4:
            input_np = input_np[0]  # (1, C, H, W) -> (C, H, W)
        if input_np.ndim == 3 and input_np.shape[0] == 1:
            input_np = input_np[0]  # (1, H, W) -> (H, W)
        
        figures = {}
        
        # Main analysis plot
        if input_np.ndim == 1:
            # 1D spectral data
            fig_main = self.visualizer.plot_1d_gradcam(
                input_np, cam_results, wavelengths=wavelengths,
                title=f"Grad-CAM Analysis - {sample_name}"
            )
        else:
            # 2D spectral data
            fig_main = self.visualizer.plot_2d_gradcam(
                input_np, cam_results,
                title=f"Grad-CAM Analysis - {sample_name}"
            )
        
        figures['main_analysis'] = fig_main
        
        # Layer importance plot
        if importance_ranking:
            fig_importance = self.visualizer.plot_layer_importance(
                importance_ranking,
                title=f"Layer Importance - {sample_name}"
            )
            figures['layer_importance'] = fig_importance
        
        # Summary report
        fig_summary = self.visualizer.create_summary_report(
            input_np, cam_results, self.model.__class__.__name__
        )
        figures['summary_report'] = fig_summary
        
        # Save figures if directory provided
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
            for fig_name, fig in figures.items():
                save_path = os.path.join(save_dir, f"{sample_name}_{fig_name}.png")
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Saved {fig_name} to {save_path}")
        
        return figures
    
    def batch_analyze(self, input_data_list: List[Union[np.ndarray, torch.Tensor]],
                     sample_names: Optional[List[str]] = None,
                     wavelengths: Optional[np.ndarray] = None,
                     save_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Analyze multiple samples in batch.
        
        Args:
            input_data_list: List of input data samples
            sample_names: Names for each sample
            wavelengths: Wavelength values
            save_dir: Directory to save results
            
        Returns:
            List of analysis results for each sample
        """
        if sample_names is None:
            sample_names = [f"sample_{i:03d}" for i in range(len(input_data_list))]
        
        results = []
        
        for i, (input_data, sample_name) in enumerate(zip(input_data_list, sample_names)):
            print(f"Analyzing sample {i+1}/{len(input_data_list)}: {sample_name}")
            
            # Analyze sample
            analysis_result = self.analyze_sample(input_data, wavelengths=wavelengths)
            
            # Create visualizations
            if save_dir:
                sample_save_dir = os.path.join(save_dir, sample_name)
                figures = self.visualize_analysis(
                    input_data, analysis_result, wavelengths, sample_save_dir, sample_name
                )
                analysis_result['figures'] = figures
            
            results.append(analysis_result)
        
        return results
    
    def compare_models(self, input_data: Union[np.ndarray, torch.Tensor],
                      other_analyzers: List['SpectralModelAnalyzer'],
                      model_names: Optional[List[str]] = None,
                      layer_name: Optional[str] = None,
                      wavelengths: Optional[np.ndarray] = None,
                      save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Compare Grad-CAM results across multiple models.
        
        Args:
            input_data: Input data to analyze
            other_analyzers: List of other model analyzers
            model_names: Names for each model
            layer_name: Specific layer to compare (if None, use first common layer)
            wavelengths: Wavelength values
            save_path: Path to save comparison plot
            
        Returns:
            Comparison results
        """
        all_analyzers = [self] + other_analyzers
        
        if model_names is None:
            model_names = [analyzer.model.__class__.__name__ for analyzer in all_analyzers]
        
        # Analyze with each model
        all_results = []
        for analyzer in all_analyzers:
            result = analyzer.analyze_sample(input_data, wavelengths=wavelengths)
            all_results.append(result['cam_results'])
        
        # Find common layer if not specified
        if layer_name is None:
            common_layers = set(all_results[0].keys())
            for result in all_results[1:]:
                common_layers &= set(result.keys())
            
            if common_layers:
                layer_name = list(common_layers)[0]
            else:
                raise ValueError("No common layers found across models")
        
        # Convert input for visualization
        if isinstance(input_data, torch.Tensor):
            input_np = input_data.cpu().numpy()
        else:
            input_np = input_data
        
        # Remove batch dimension
        if input_np.ndim == 4:
            input_np = input_np[0]
        if input_np.ndim == 3 and input_np.shape[0] == 1:
            input_np = input_np[0]
        
        # Create comparison plot
        fig = self.visualizer.plot_comparative_analysis(
            input_np, all_results, model_names, layer_name, wavelengths, save_path
        )
        
        return {
            'comparison_figure': fig,
            'all_results': all_results,
            'model_names': model_names,
            'compared_layer': layer_name
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model and analysis setup."""
        return {
            'model_class': self.model.__class__.__name__,
            'model_type': self.model_type,
            'target_layers': self.target_layers,
            'device': str(self.device),
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }


def create_analyzer_from_checkpoint(checkpoint_path: str, model_class: type,
                                  model_kwargs: Dict[str, Any],
                                  model_type: str = 'auto') -> SpectralModelAnalyzer:
    """
    Create a model analyzer from a saved checkpoint.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        model_class: Model class to instantiate
        model_kwargs: Keyword arguments for model initialization
        model_type: Type of model
        
    Returns:
        SpectralModelAnalyzer instance
    """
    # Load model
    model = model_class(**model_kwargs)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Create analyzer
    analyzer = SpectralModelAnalyzer(model, model_type)
    
    print(f"Loaded model from {checkpoint_path}")
    
    return analyzer


def analyze_model_from_config(config_path: str, checkpoint_path: str) -> SpectralModelAnalyzer:
    """
    Create analyzer from configuration file and checkpoint.
    
    Args:
        config_path: Path to model configuration
        checkpoint_path: Path to model checkpoint
        
    Returns:
        SpectralModelAnalyzer instance
    """
    import json
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Extract model information
    model_name = config.get('model_name', 'UNET')
    model_params = config.get('model_params', {})
    
    # Import appropriate model class
    if model_name.lower() == 'unet':
        from src.regression.models.unet import UNET
        model_class = UNET
        model_type = 'unet'
    elif model_name.lower() == 'dualunet':
        from src.regression.models.dualunet import DualUNet
        model_class = DualUNet
        model_type = 'unet'
    elif model_name.lower() == 'dualsimplecnn':
        from src.regression.models.dualsimplecnn import DualSimpleCNN
        model_class = DualSimpleCNN
        model_type = 'cnn'
    elif model_name.lower() == 'resnet18':
        from src.regression.models.resnet18 import ResNet18
        model_class = ResNet18
        model_type = 'resnet'
    else:
        raise ValueError(f"Unsupported model type: {model_name}")
    
    return create_analyzer_from_checkpoint(checkpoint_path, model_class, model_params, model_type)