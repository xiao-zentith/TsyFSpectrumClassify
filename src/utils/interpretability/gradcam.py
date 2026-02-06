"""
Grad-CAM implementation for spectral data analysis.
Supports both 1D and 2D spectral data with specialized handling for UNet and CNN architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from scipy.ndimage import zoom


class GradCAM:
    """
    Generic Grad-CAM implementation for PyTorch models.
    """
    
    def __init__(self, model: nn.Module, target_layers: List[str], use_cuda: bool = True):
        """
        Initialize Grad-CAM.
        
        Args:
            model: PyTorch model
            target_layers: List of layer names to compute gradients for
            use_cuda: Whether to use CUDA if available
        """
        self.model = model
        self.target_layers = target_layers
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.gradients = {}
        self.activations = {}
        self.hooks = []
        
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks for target layers."""
        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach()
            return hook
        
        def get_gradient(name):
            def hook(model, input, output):
                self.gradients[name] = output[0].detach()
            return hook
        
        for name, module in self.model.named_modules():
            if name in self.target_layers:
                # Forward hook for activations
                handle_f = module.register_forward_hook(get_activation(name))
                # Backward hook for gradients
                handle_b = module.register_backward_hook(get_gradient(name))
                self.hooks.extend([handle_f, handle_b])
    
    def generate_cam(self, input_tensor: torch.Tensor, target_class: Optional[int] = None, 
                     target_output_idx: int = 0) -> Dict[str, np.ndarray]:
        """
        Generate Grad-CAM heatmaps.
        
        Args:
            input_tensor: Input tensor (batch_size, channels, height, width)
            target_class: Target class index (for classification)
            target_output_idx: Target output index (for multi-output models)
            
        Returns:
            Dictionary mapping layer names to CAM heatmaps
        """
        self.model.eval()
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad_(True)
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Handle multi-output models (like DualUNet)
        if isinstance(output, (list, tuple)):
            output = output[target_output_idx]
        
        # For regression tasks, use the mean of all outputs as the target
        if target_class is None:
            if len(output.shape) > 2:  # Spatial output
                target = output.mean()
            else:  # Vector output
                target = output.mean()
        else:
            target = output[0, target_class]
        
        # Backward pass
        self.model.zero_grad()
        target.backward(retain_graph=True)
        
        # Generate CAMs for each target layer
        cams = {}
        for layer_name in self.target_layers:
            if layer_name in self.gradients and layer_name in self.activations:
                gradients = self.gradients[layer_name]
                activations = self.activations[layer_name]
                
                # Global average pooling of gradients
                weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
                
                # Weighted combination of activation maps
                cam = torch.sum(weights * activations, dim=1, keepdim=True)
                cam = F.relu(cam)
                
                # Normalize CAM
                cam = cam.squeeze().cpu().numpy()
                if cam.ndim == 2:
                    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
                
                cams[layer_name] = cam
        
        return cams
    
    def __del__(self):
        """Clean up hooks when object is destroyed."""
        if hasattr(self, 'hooks'):
            for hook in self.hooks:
                hook.remove()


class SpectralGradCAM(GradCAM):
    """
    Specialized Grad-CAM for spectral data analysis.
    
    Extends the base GradCAM class with spectral-specific functionality
    such as wavelength analysis and spectral region identification.
    """
    
    def __init__(self, model: nn.Module, target_layers: Optional[List[str]] = None, 
                 model_type: Optional[str] = None, use_cuda: bool = True):
        """
        Initialize SpectralGradCAM.
        
        Args:
            model: PyTorch model
            target_layers: List of layer names to analyze
            model_type: Type of model for automatic layer detection
            use_cuda: Whether to use CUDA if available
        """
        # Auto-detect target layers if not provided
        if target_layers is None and model_type is not None:
            target_layers = get_model_target_layers(model, model_type)
        
        super().__init__(model, target_layers, use_cuda)
        self.model_type = model_type
        self.spectral_bands = None
        self.wavelengths = None
    
    def generate_cam(self, input_data: Union[torch.Tensor, np.ndarray], 
                     target_class: Optional[int] = None,
                     target_output_idx: int = 0) -> Dict[str, Dict]:
        """
        Generate Grad-CAM for spectral data.
        
        Args:
            input_data: Input spectral data (numpy array or torch tensor)
            target_class: Target class for analysis
            target_output_idx: Target output index for multi-output models
            
        Returns:
            Dictionary with CAM analysis results
        """
        # Convert numpy array to tensor if needed
        if isinstance(input_data, np.ndarray):
            # Handle different input shapes
            if input_data.ndim == 1:  # 1D spectral data
                # For UNet, we need 4D input: [batch, channel, height, width]
                # Reshape 1D data to 2D for UNet compatibility
                input_tensor = torch.FloatTensor(input_data).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, length]
            elif input_data.ndim == 2:  # 2D spectral data
                input_tensor = torch.FloatTensor(input_data).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            elif input_data.ndim == 3:  # Multi-channel 2D data
                input_tensor = torch.FloatTensor(input_data).unsqueeze(0)  # [1, C, H, W]
            else:
                input_tensor = torch.FloatTensor(input_data)
        else:
            input_tensor = input_data
        
        # Generate standard CAMs using parent method
        cams = super().generate_cam(input_tensor, target_class, target_output_idx)
        
        # Add spectral-specific analysis
        results = {}
        for layer_name, cam in cams.items():
            layer_results = {
                'cam': cam,
                'cam_shape': cam.shape,
                'peak_intensity': float(np.max(cam)),
                'mean_intensity': float(np.mean(cam))
            }
            
            # Add spectral-specific analysis based on data dimensionality
            if input_data.ndim == 1 or (input_data.ndim == 2 and min(input_data.shape) == 1):
                # 1D spectral data
                layer_results.update(self._analyze_1d_spectral_cam(cam, input_tensor))
            elif input_data.ndim >= 2:
                # 2D spectral data
                layer_results.update(self._analyze_2d_spectral_cam(cam, input_tensor))
            
            results[layer_name] = layer_results
        
        return results
    
    def get_important_wavelengths(self, cam: np.ndarray, wavelengths: np.ndarray, 
                                 top_k: int = 5) -> List[float]:
        """
        Get the most important wavelengths based on CAM values.
        
        Args:
            cam: CAM array
            wavelengths: Wavelength array
            top_k: Number of top wavelengths to return
            
        Returns:
            List of important wavelengths
        """
        if len(cam) != len(wavelengths):
            # Interpolate CAM to match wavelengths
            from scipy.interpolate import interp1d
            f = interp1d(np.linspace(0, 1, len(cam)), cam, kind='linear')
            cam = f(np.linspace(0, 1, len(wavelengths)))
        
        # Get indices of top CAM values
        top_indices = np.argsort(cam)[-top_k:][::-1]
        return wavelengths[top_indices].tolist()
    
    def rank_layer_importance(self, cam_results: Dict[str, Dict]) -> List[Tuple[str, float]]:
        """
        Rank layers by their importance based on CAM statistics.
        
        Args:
            cam_results: Dictionary of CAM results
            
        Returns:
            List of (layer_name, importance_score) tuples, sorted by importance
        """
        importance_scores = []
        
        for layer_name, layer_data in cam_results.items():
            # Calculate importance score based on peak and mean intensity
            peak = layer_data.get('peak_intensity', 0)
            mean = layer_data.get('mean_intensity', 0)
            
            # Weighted combination of peak and mean
            importance_score = 0.7 * peak + 0.3 * mean
            importance_scores.append((layer_name, importance_score))
        
        # Sort by importance score (descending)
        importance_scores.sort(key=lambda x: x[1], reverse=True)
        return importance_scores
    
    def generate_spectral_cam(self, input_tensor: torch.Tensor,
                            return_raw: bool = False) -> Dict[str, Dict]:
        """
        Generate spectral-specific Grad-CAM analysis.
        
        Args:
            input_tensor: Input spectral tensor
            target_class: Target class for analysis
            target_output_idx: Target output index for multi-output models
            return_raw: Whether to return raw CAM values
            
        Returns:
            Dictionary with CAM analysis results
        """
        # Generate standard CAMs
        cams = self.generate_cam(input_tensor, target_class, target_output_idx)
        
        results = {}
        for layer_name, cam in cams.items():
            layer_results = {
                'cam': cam,
                'input_shape': input_tensor.shape,
                'cam_shape': cam.shape
            }
            
            # Add spectral-specific analysis
            if cam.ndim == 2:  # 2D spectral data
                layer_results.update(self._analyze_2d_spectral_cam(cam, input_tensor))
            elif cam.ndim == 1:  # 1D spectral data
                layer_results.update(self._analyze_1d_spectral_cam(cam, input_tensor))
            
            if not return_raw:
                # Resize CAM to match input size for visualization
                layer_results['cam_resized'] = self._resize_cam_to_input(
                    cam, input_tensor.shape[-2:])
            
            results[layer_name] = layer_results
        
        return results
    
    def _analyze_2d_spectral_cam(self, cam: np.ndarray, input_tensor: torch.Tensor) -> Dict:
        """Analyze 2D spectral CAM for important regions."""
        analysis = {}
        
        # Find peak regions
        threshold = np.percentile(cam, 90)
        important_regions = cam > threshold
        
        analysis['important_regions'] = important_regions
        analysis['peak_intensity'] = cam.max()
        analysis['mean_intensity'] = cam.mean()
        
        # Spatial analysis
        if cam.shape[0] > 1 and cam.shape[1] > 1:
            # Find center of mass
            y_indices, x_indices = np.indices(cam.shape)
            total_intensity = cam.sum()
            if total_intensity > 0:
                center_y = (y_indices * cam).sum() / total_intensity
                center_x = (x_indices * cam).sum() / total_intensity
                analysis['center_of_mass'] = (center_y, center_x)
        
        return analysis
    
    def _analyze_1d_spectral_cam(self, cam: np.ndarray, input_tensor: torch.Tensor) -> Dict:
        """Analyze 1D spectral CAM for important wavelengths."""
        analysis = {}
        
        # Find peak wavelengths
        threshold = np.percentile(cam, 90)
        important_indices = np.where(cam > threshold)[0]
        
        analysis['important_indices'] = important_indices
        analysis['peak_intensity'] = cam.max()
        analysis['mean_intensity'] = cam.mean()
        
        # Wavelength analysis if available
        if self.wavelengths is not None and len(self.wavelengths) == len(cam):
            important_wavelengths = self.wavelengths[important_indices]
            analysis['important_wavelengths'] = important_wavelengths
            
            # Find dominant wavelength
            max_idx = np.argmax(cam)
            analysis['dominant_wavelength'] = self.wavelengths[max_idx]
        
        return analysis
    
    def _resize_cam_to_input(self, cam: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize CAM to match input tensor size."""
        if cam.shape == target_size:
            return cam
        
        # Use scipy zoom for high-quality resizing
        if len(cam.shape) == 2:
            zoom_factors = (target_size[0] / cam.shape[0], target_size[1] / cam.shape[1])
            resized_cam = zoom(cam, zoom_factors, order=3)  # order=3 for cubic interpolation
        else:
            resized_cam = cam
        
        return resized_cam
    
    def get_layer_importance_ranking(self, input_tensor: torch.Tensor, 
                                   target_output_idx: int = 0) -> List[Tuple[str, float]]:
        """
        Rank layers by their importance based on gradient magnitude.
        
        Args:
            input_tensor: Input tensor
            target_output_idx: Target output index
            
        Returns:
            List of (layer_name, importance_score) tuples sorted by importance
        """
        cams = self.generate_cam(input_tensor, target_output_idx=target_output_idx)
        
        importance_scores = []
        for layer_name, cam in cams.items():
            # Calculate importance as mean activation strength
            importance = float(np.mean(cam))
            importance_scores.append((layer_name, importance))
        
        # Sort by importance (descending)
        importance_scores.sort(key=lambda x: x[1], reverse=True)
        
        return importance_scores


def get_model_target_layers(model: nn.Module, model_type: str = 'auto') -> List[str]:
    """
    Automatically detect suitable target layers for Grad-CAM based on model architecture.
    
    Args:
        model: PyTorch model
        model_type: Model type ('unet', 'cnn', 'resnet', 'auto')
        
    Returns:
        List of suitable layer names for Grad-CAM
    """
    target_layers = []
    
    if model_type == 'auto':
        model_name = model.__class__.__name__.lower()
        if 'unet' in model_name:
            model_type = 'unet'
        elif 'resnet' in model_name:
            model_type = 'resnet'
        elif 'cnn' in model_name:
            model_type = 'cnn'
    
    for name, module in model.named_modules():
        if model_type == 'unet':
            # For UNet, target the last layer of each encoder block and decoder block
            if ('downs' in name and 'conv' in name) or ('ups' in name and 'conv' in name):
                if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                    target_layers.append(name)
        elif model_type == 'resnet':
            # For ResNet, target the last conv layer of each block
            if 'layer' in name and 'conv2' in name:
                target_layers.append(name)
        elif model_type == 'cnn':
            # For CNN, target all conv layers
            if isinstance(module, nn.Conv2d):
                target_layers.append(name)
    
    # If no specific layers found, use the last few conv layers
    if not target_layers:
        conv_layers = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                conv_layers.append(name)
        
        # Take the last 3 conv layers
        target_layers = conv_layers[-3:] if len(conv_layers) >= 3 else conv_layers
    
    return target_layers