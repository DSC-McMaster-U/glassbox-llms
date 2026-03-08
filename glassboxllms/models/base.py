from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
import numpy as np


class ModelWrapper(ABC):
    """
    Abstract base class for all model backends.
    
    Provides a consistent API for:
    - Forward passes
    - Activation extraction
    - Activation patching (via hooks)
    - Model introspection
    
    Designed to support: HuggingFace, GGUF, custom PyTorch/TF architectures.
    
    This interface is stable for dependent modules:
    - glassboxllms.instrumentation.hooks
    - glassboxllms.instrumentation.activation_patching
    - glassboxllms.primitives.probes
    """
    
    def __init__(self):
        """Initialize the wrapper."""
        self._model_info: Dict[str, Any] = {}
    
    @abstractmethod
    def forward(self, inputs: Any, **kwargs) -> Any:
        """
        Execute forward pass on inputs.
        
        Args:
            inputs: Can be text (str), tokens (List/np.ndarray), or embeddings
            **kwargs: Framework-specific options (attention_mask, etc.)
            
        Returns:
            Model output (framework-dependent)
        """
        ...
    
    @abstractmethod
    def get_activations(
        self, 
        inputs: Any, 
        layers: List[str],
        return_type: str = "numpy"
    ) -> Dict[str, Any]:
        """
        Extract activations from specified layers.
        
        Args:
            inputs: Model inputs (text, tokens, embeddings, etc.)
            layers: List of layer identifiers (e.g., ["layer.0.attention", "layer.5.mlp"])
            return_type: "numpy" or "torch" — format for return values
            
        Returns:
            Dict mapping layer names to activation arrays/tensors
            Shape should be (batch_size, seq_len, hidden_dim) or (batch_size, hidden_dim)
        """
        ...
    
    @abstractmethod
    def get_layer_module(self, layer: str) -> Any:
        """
        Get the actual module/layer object for hooking.
        
        Required for activation_patching to attach hooks.
        
        Args:
            layer: Layer identifier (e.g., "layer.5.mlp")
            
        Returns:
            Framework-native module object (torch.nn.Module, etc.)
        """
        ...
    
    @abstractmethod
    def get_layer_shape(self, layer: str) -> Tuple[int, ...]:
        """
        Get output shape signature of a layer.
        
        Returns:
            Tuple of dimensions, e.g., (hidden_dim,) or (seq_len, hidden_dim)
            Used by probes for validation and activation patching for shape matching.
        """
        ...
    
    @property
    @abstractmethod
    def layer_names(self) -> List[str]:
        """
        Return all available layer identifiers.
        
        Must follow consistent naming across backends:
        - "layer.{i}.attention" for attention layers
        - "layer.{i}.mlp" for feedforward layers
        - "embedding" for embedding layer
        """
        ...
    
    @property
    @abstractmethod
    def device(self) -> str:
        """Return device the model is on (e.g., 'cuda:0', 'cpu')."""
        ...
    
    @property
    @abstractmethod
    def model_config(self) -> Dict[str, Any]:
        """
        Return model metadata for probes and analysis.
        
        Should include:
        - hidden_size: Model hidden dimension
        - num_layers: Number of layers
        - vocab_size: Vocabulary size
        - model_type: "transformers" | "gguf" | "custom"
        """
        ...
    
    def set_eval_mode(self):
        """Set model to evaluation mode (no dropout, batch norm fixed)."""
        pass
    
    def set_train_mode(self):
        """Set model to training mode (for if needed)."""
        pass