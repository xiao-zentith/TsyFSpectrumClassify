"""
Visualization tools for Grad-CAM analysis of spectral data.
Provides specialized plotting functions for 1D and 2D spectral data interpretation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Optional, Tuple, Union
from scipy import ndimage


class GradCAMVisualizer:
    """
    Visualization tools for Grad-CAM analysis of spectral data.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 100):
        """
        Initialize the visualizer.
        
        Args:
            figsize: Default figure size
            dpi: Figure DPI
        """
        self.figsize = figsize
        self.dpi = dpi
        
        # Set up custom colormaps for spectral data
        self.setup_colormaps()
        
        # Configure matplotlib for better plots
        plt.style.use('default')
    
    def setup_colormaps(self):
        """Setup custom colormaps for spectral visualization."""
        # Custom colormap for spectral data (blue to red)
        colors = ['#000080', '#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF8000', '#FF0000']
        self.spectral_cmap = LinearSegmentedColormap.from_list('spectral', colors)
        
        # Heatmap colormap
        self.heatmap_cmap = plt.cm.hot
    
    def plot_2d_gradcam(self, input_data: np.ndarray, cam_results: Dict, 
                       layer_names: Optional[List[str]] = None,
                       save_path: Optional[str] = None,
                       title: str = "Grad-CAM Analysis") -> plt.Figure:
        """
        Plot 2D Grad-CAM results for spectral data.
        
        Args:
            input_data: Original input data (H, W) or (C, H, W)
            cam_results: Dictionary of CAM results from SpectralGradCAM
            layer_names: Specific layers to plot (if None, plot all)
            save_path: Path to save the figure
            title: Figure title
            
        Returns:
            matplotlib Figure object
        """
        if layer_names is None:
            layer_names = list(cam_results.keys())
        
        n_layers = len(layer_names)
        
        # Create subplot layout
        if n_layers <= 2:
            rows, cols = 1, n_layers + 1  # +1 for original image
        elif n_layers <= 4:
            rows, cols = 2, 3
        else:
            rows = int(np.ceil((n_layers + 1) / 3))
            cols = 3
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), dpi=self.dpi)
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        # Plot original input
        if input_data.ndim == 3:
            # Take the first channel or average across channels
            display_input = input_data[0] if input_data.shape[0] <= 3 else np.mean(input_data, axis=0)
        else:
            display_input = input_data
        
        im0 = axes[0, 0].imshow(display_input, cmap=self.spectral_cmap)
        axes[0, 0].set_title('Original Input')
        axes[0, 0].axis('off')
        plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)
        
        # Plot CAMs for each layer
        for idx, layer_name in enumerate(layer_names):
            row = (idx + 1) // cols
            col = (idx + 1) % cols
            
            if row >= rows:
                break
            
            cam_data = cam_results[layer_name]
            cam = cam_data.get('cam_resized', cam_data['cam'])
            
            # Create overlay
            overlay = self._create_overlay(display_input, cam, alpha=0.6)
            
            axes[row, col].imshow(overlay)
            axes[row, col].set_title(f'{layer_name}\nPeak: {cam_data["peak_intensity"]:.3f}')
            axes[row, col].axis('off')
            
            # Add importance information if available
            if 'center_of_mass' in cam_data:
                cy, cx = cam_data['center_of_mass']
                axes[row, col].plot(cx, cy, 'w+', markersize=10, markeredgewidth=2)
        
        # Hide unused subplots
        for idx in range(len(layer_names) + 1, rows * cols):
            row = idx // cols
            col = idx % cols
            if row < rows and col < cols:
                axes[row, col].axis('off')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_1d_gradcam(self, input_data: np.ndarray, cam_results: Dict,
                       wavelengths: Optional[np.ndarray] = None,
                       layer_names: Optional[List[str]] = None,
                       save_path: Optional[str] = None,
                       title: str = "1D Spectral Grad-CAM Analysis") -> plt.Figure:
        """
        Plot 1D Grad-CAM results for spectral data.
        
        Args:
            input_data: Original 1D spectral data
            cam_results: Dictionary of CAM results
            wavelengths: Wavelength values for x-axis
            layer_names: Specific layers to plot
            save_path: Path to save the figure
            title: Figure title
            
        Returns:
            matplotlib Figure object
        """
        if layer_names is None:
            layer_names = list(cam_results.keys())
        
        n_layers = len(layer_names)
        
        fig, axes = plt.subplots(n_layers + 1, 1, figsize=(12, (n_layers + 1) * 3), dpi=self.dpi)
        if n_layers == 0:
            axes = [axes]
        
        # Prepare x-axis
        if wavelengths is not None:
            x_axis = wavelengths
            x_label = 'Wavelength (nm)'
        else:
            x_axis = np.arange(len(input_data))
            x_label = 'Spectral Index'
        
        # Plot original spectrum
        axes[0].plot(x_axis, input_data, 'b-', linewidth=2, label='Original Spectrum')
        axes[0].set_title('Original Spectral Data')
        axes[0].set_xlabel(x_label)
        axes[0].set_ylabel('Intensity')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Plot CAM for each layer
        for idx, layer_name in enumerate(layer_names):
            ax = axes[idx + 1]
            cam_data = cam_results[layer_name]
            cam = cam_data['cam']
            
            # Resize CAM to match input length if needed
            if len(cam) != len(input_data):
                from scipy.interpolate import interp1d
                f = interp1d(np.linspace(0, 1, len(cam)), cam, kind='linear')
                cam = f(np.linspace(0, 1, len(input_data)))
            
            # Plot spectrum with CAM overlay
            ax.plot(x_axis, input_data, 'b-', alpha=0.5, linewidth=1, label='Spectrum')
            
            # Create filled area for important regions
            ax.fill_between(x_axis, 0, input_data * cam, alpha=0.7, 
                          color='red', label='Important Regions')
            
            # Plot CAM as line
            ax2 = ax.twinx()
            ax2.plot(x_axis, cam, 'r-', linewidth=2, alpha=0.8, label='Grad-CAM')
            ax2.set_ylabel('Grad-CAM Intensity', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            
            # Highlight important wavelengths
            if 'important_indices' in cam_data:
                important_indices = cam_data['important_indices']
                if len(important_indices) > 0:
                    important_x = x_axis[important_indices]
                    important_y = input_data[important_indices]
                    ax.scatter(important_x, important_y, color='orange', s=50, 
                             zorder=5, label='Key Wavelengths')
            
            ax.set_title(f'{layer_name} - Peak Intensity: {cam_data["peak_intensity"]:.3f}')
            ax.set_xlabel(x_label)
            ax.set_ylabel('Spectral Intensity')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_layer_importance(self, importance_scores: List[Tuple[str, float]],
                            save_path: Optional[str] = None,
                            title: str = "Layer Importance Ranking") -> plt.Figure:
        """
        Plot layer importance ranking.
        
        Args:
            importance_scores: List of (layer_name, importance_score) tuples
            save_path: Path to save the figure
            title: Figure title
            
        Returns:
            matplotlib Figure object
        """
        if not importance_scores:
            raise ValueError("No importance scores provided")
        
        layer_names, scores = zip(*importance_scores)
        
        fig, ax = plt.subplots(figsize=(10, max(6, len(layer_names) * 0.5)), dpi=self.dpi)
        
        # Create horizontal bar plot
        y_pos = np.arange(len(layer_names))
        bars = ax.barh(y_pos, scores, color=plt.cm.viridis(np.linspace(0, 1, len(scores))))
        
        # Customize plot
        ax.set_yticks(y_pos)
        ax.set_yticklabels(layer_names)
        ax.invert_yaxis()  # Top layer at the top
        ax.set_xlabel('Importance Score')
        ax.set_title(title, fontweight='bold')
        
        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax.text(bar.get_width() + max(scores) * 0.01, bar.get_y() + bar.get_height()/2,
                   f'{score:.3f}', ha='left', va='center', fontweight='bold')
        
        ax.grid(True, axis='x', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_comparative_analysis(self, input_data: np.ndarray, 
                                cam_results_list: List[Dict],
                                model_names: List[str],
                                layer_name: str,
                                wavelengths: Optional[np.ndarray] = None,
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Compare Grad-CAM results across different models.
        
        Args:
            input_data: Original input data
            cam_results_list: List of CAM results from different models
            model_names: Names of the models
            layer_name: Layer to compare
            wavelengths: Wavelength values
            save_path: Path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        n_models = len(cam_results_list)
        
        if input_data.ndim == 1:
            # 1D comparison
            fig, axes = plt.subplots(n_models + 1, 1, figsize=(12, (n_models + 1) * 3), dpi=self.dpi)
            if n_models == 0:
                axes = [axes]
            
            # Prepare x-axis
            x_axis = wavelengths if wavelengths is not None else np.arange(len(input_data))
            x_label = 'Wavelength (nm)' if wavelengths is not None else 'Spectral Index'
            
            # Plot original
            axes[0].plot(x_axis, input_data, 'b-', linewidth=2)
            axes[0].set_title('Original Spectrum')
            axes[0].set_xlabel(x_label)
            axes[0].set_ylabel('Intensity')
            axes[0].grid(True, alpha=0.3)
            
            # Plot each model's CAM
            for idx, (cam_results, model_name) in enumerate(zip(cam_results_list, model_names)):
                ax = axes[idx + 1]
                
                if layer_name in cam_results:
                    cam = cam_results[layer_name]['cam']
                    
                    # Plot with CAM overlay
                    ax.plot(x_axis, input_data, 'b-', alpha=0.5, linewidth=1)
                    ax.fill_between(x_axis, 0, input_data * cam, alpha=0.7, color='red')
                    
                    ax2 = ax.twinx()
                    ax2.plot(x_axis, cam, 'r-', linewidth=2)
                    ax2.set_ylabel('Grad-CAM', color='red')
                    ax2.tick_params(axis='y', labelcolor='red')
                
                ax.set_title(f'{model_name} - {layer_name}')
                ax.set_xlabel(x_label)
                ax.set_ylabel('Intensity')
                ax.grid(True, alpha=0.3)
        
        else:
            # 2D comparison
            fig, axes = plt.subplots(1, n_models + 1, figsize=((n_models + 1) * 4, 4), dpi=self.dpi)
            if n_models == 0:
                axes = [axes]
            
            # Plot original
            display_input = input_data[0] if input_data.ndim == 3 else input_data
            axes[0].imshow(display_input, cmap=self.spectral_cmap)
            axes[0].set_title('Original')
            axes[0].axis('off')
            
            # Plot each model's CAM
            for idx, (cam_results, model_name) in enumerate(zip(cam_results_list, model_names)):
                if layer_name in cam_results:
                    cam = cam_results[layer_name]['cam']
                    overlay = self._create_overlay(display_input, cam)
                    axes[idx + 1].imshow(overlay)
                
                axes[idx + 1].set_title(model_name)
                axes[idx + 1].axis('off')
        
        plt.suptitle(f'Model Comparison - {layer_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def _create_overlay(self, input_image: np.ndarray, cam: np.ndarray, 
                       alpha: float = 0.6) -> np.ndarray:
        """
        Create overlay of input image and CAM heatmap.
        
        Args:
            input_image: Original input image
            cam: CAM heatmap
            alpha: Transparency of the heatmap
            
        Returns:
            Overlay image
        """
        # Normalize input image to [0, 1]
        input_norm = (input_image - input_image.min()) / (input_image.max() - input_image.min() + 1e-8)
        
        # Resize CAM to match input size if needed
        if cam.shape != input_image.shape:
            from scipy.ndimage import zoom
            zoom_factors = (input_image.shape[0] / cam.shape[0], input_image.shape[1] / cam.shape[1])
            cam = zoom(cam, zoom_factors, order=3)
        
        # Convert to RGB
        input_rgb = cm.gray(input_norm)[:, :, :3]
        cam_rgb = self.heatmap_cmap(cam)[:, :, :3]
        
        # Create overlay
        overlay = (1 - alpha) * input_rgb + alpha * cam_rgb
        
        return np.clip(overlay, 0, 1)
    
    def create_summary_report(self, input_data: np.ndarray, cam_results: Dict,
                            model_name: str, save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a comprehensive summary report of Grad-CAM analysis.
        
        Args:
            input_data: Original input data
            cam_results: CAM analysis results
            model_name: Name of the analyzed model
            save_path: Path to save the report
            
        Returns:
            matplotlib Figure object
        """
        fig = plt.figure(figsize=(16, 12), dpi=self.dpi)
        
        # Create grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle(f'Grad-CAM Analysis Report - {model_name}', 
                    fontsize=20, fontweight='bold')
        
        # Original data
        ax1 = fig.add_subplot(gs[0, 0])
        if input_data.ndim == 1:
            ax1.plot(input_data)
            ax1.set_title('Original Spectrum')
        else:
            display_input = input_data[0] if input_data.ndim == 3 else input_data
            ax1.imshow(display_input, cmap=self.spectral_cmap)
            ax1.set_title('Original Input')
        ax1.axis('off')
        
        # Layer importance
        layer_names = list(cam_results.keys())
        importance_scores = [(name, cam_results[name]['peak_intensity']) 
                           for name in layer_names]
        importance_scores.sort(key=lambda x: x[1], reverse=True)
        
        ax2 = fig.add_subplot(gs[0, 1:3])
        names, scores = zip(*importance_scores) if importance_scores else ([], [])
        if names:
            bars = ax2.bar(range(len(names)), scores, color=plt.cm.viridis(np.linspace(0, 1, len(scores))))
            ax2.set_xticks(range(len(names)))
            ax2.set_xticklabels(names, rotation=45, ha='right')
            ax2.set_title('Layer Importance (Peak Intensity)')
            ax2.set_ylabel('Peak Intensity')
        
        # Statistics table
        ax3 = fig.add_subplot(gs[0, 3])
        ax3.axis('off')
        
        stats_data = []
        for layer_name in layer_names[:5]:  # Top 5 layers
            cam_data = cam_results[layer_name]
            stats_data.append([
                layer_name[:15] + '...' if len(layer_name) > 15 else layer_name,
                f"{cam_data['peak_intensity']:.3f}",
                f"{cam_data['mean_intensity']:.3f}"
            ])
        
        if stats_data:
            table = ax3.table(cellText=stats_data,
                            colLabels=['Layer', 'Peak', 'Mean'],
                            cellLoc='center',
                            loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.5)
            ax3.set_title('Top Layers Statistics')
        
        # Individual layer visualizations
        for idx, layer_name in enumerate(layer_names[:6]):  # Show top 6 layers
            row = 1 + idx // 3
            col = idx % 3
            
            if row >= 3:
                break
            
            ax = fig.add_subplot(gs[row, col])
            cam_data = cam_results[layer_name]
            cam = cam_data['cam']
            
            if input_data.ndim == 1:
                ax.plot(input_data, alpha=0.5, label='Spectrum')
                ax.fill_between(range(len(input_data)), 0, input_data * cam, 
                              alpha=0.7, color='red', label='Important')
                ax.legend()
            else:
                display_input = input_data[0] if input_data.ndim == 3 else input_data
                overlay = self._create_overlay(display_input, cam)
                ax.imshow(overlay)
            
            ax.set_title(f'{layer_name}\nPeak: {cam_data["peak_intensity"]:.3f}')
            ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig