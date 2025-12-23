"""
LlamaFactory - Unified Model Factory for Namdm
Provides a factory pattern for creating and training different neural network models
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Type
from models.amgru import AMGRU
from models.amlstm import AMLSTM
from models.bilstm import BILSTMModel
from models.gru import GRUModel
from models.lstm import LSTMModel


class ModelFactory:
    """Factory class for creating different model architectures"""
    
    # Registry of available models
    _models: Dict[str, Type[nn.Module]] = {
        'amgru': AMGRU,
        'amlstm': AMLSTM,
        'bilstm': BILSTMModel,
        'gru': GRUModel,
        'lstm': LSTMModel,
    }
    
    # Parameter mapping for models with different parameter names
    _param_mapping: Dict[str, Dict[str, str]] = {
        'amgru': {'input_dim': 'input_size', 'hidden_dim': 'hidden_size'},
        'amlstm': {'input_dim': 'input_size', 'hidden_dim': 'hidden_size'},
        'bilstm': {'input_dim': 'input_dim', 'hidden_dim': 'hidden_dim'},
        'gru': {'input_dim': 'input_dim', 'hidden_dim': 'hidden_dim'},
        'lstm': {'input_dim': 'input_dim', 'hidden_dim': 'hidden_dim'},
    }
    
    @classmethod
    def create_model(cls, model_name: str, input_dim: int, hidden_dim: int, **kwargs) -> nn.Module:
        """
        Create a model instance based on model name
        
        Args:
            model_name: Name of the model architecture ('amgru', 'amlstm', 'bilstm', 'gru', 'lstm')
            input_dim: Input dimension
            hidden_dim: Hidden layer dimension
            **kwargs: Additional model-specific parameters
            
        Returns:
            Model instance
            
        Raises:
            ValueError: If model_name is not registered
        """
        model_name = model_name.lower()
        
        if model_name not in cls._models:
            available_models = ', '.join(cls._models.keys())
            raise ValueError(
                f"Model '{model_name}' not found. Available models: {available_models}"
            )
        
        model_class = cls._models[model_name]
        param_map = cls._param_mapping.get(model_name, {})
        
        # Map parameters according to model-specific naming
        params = {
            param_map.get('input_dim', 'input_dim'): input_dim,
            param_map.get('hidden_dim', 'hidden_dim'): hidden_dim
        }
        params.update(kwargs)
        
        return model_class(**params)
    
    @classmethod
    def register_model(cls, name: str, model_class: Type[nn.Module]) -> None:
        """
        Register a new model architecture
        
        Args:
            name: Name to register the model under
            model_class: Model class to register
        """
        cls._models[name.lower()] = model_class
    
    @classmethod
    def list_models(cls) -> list:
        """
        Get list of all registered model names
        
        Returns:
            List of model names
        """
        return list(cls._models.keys())
    
    @classmethod
    def get_model_info(cls) -> Dict[str, str]:
        """
        Get information about all registered models
        
        Returns:
            Dictionary mapping model names to their descriptions
        """
        return {
            'amgru': 'Attention Mechanism with GRU',
            'amlstm': 'Attention Mechanism with LSTM',
            'bilstm': 'Bidirectional LSTM',
            'gru': 'Gated Recurrent Unit',
            'lstm': 'Long Short-Term Memory',
        }


class LlamaFactory:
    """
    Main factory class providing unified interface for model training
    Inspired by LlamaFactory's design pattern
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize LlamaFactory with configuration
        
        Args:
            config: Configuration dictionary containing model and training parameters
        """
        self.config = config or {}
        self.model_factory = ModelFactory()
    
    def create_model(self, model_name: str = None, **kwargs) -> nn.Module:
        """
        Create a model using factory pattern
        
        Args:
            model_name: Name of model architecture (can also be in config)
            **kwargs: Override parameters
            
        Returns:
            Model instance
        """
        # Get model name from args or config
        model_name = model_name or self.config.get('model_name', 'amgru')
        
        # Get model parameters from config and kwargs
        model_params = {
            'input_dim': self.config.get('input_dim', 10),
            'hidden_dim': self.config.get('hidden_dim', 32),
        }
        model_params.update(kwargs)
        
        return self.model_factory.create_model(model_name, **model_params)
    
    def list_available_models(self) -> list:
        """Get list of available models"""
        return self.model_factory.list_models()
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about available models"""
        return self.model_factory.get_model_info()


def create_model_from_config(config: Dict[str, Any]) -> nn.Module:
    """
    Convenience function to create a model from configuration
    
    Args:
        config: Configuration dictionary with 'model_name', 'input_dim', 'hidden_dim'
        
    Returns:
        Model instance
        
    Example:
        >>> config = {'model_name': 'amgru', 'input_dim': 10, 'hidden_dim': 32}
        >>> model = create_model_from_config(config)
    """
    factory = LlamaFactory(config)
    return factory.create_model()


if __name__ == "__main__":
    # Example usage
    print("=== LlamaFactory Demo ===\n")
    
    # Create factory instance
    factory = LlamaFactory()
    
    # List available models
    print("Available models:")
    for model_name in factory.list_available_models():
        print(f"  - {model_name}")
    
    print("\nModel information:")
    for name, description in factory.get_model_info().items():
        print(f"  - {name}: {description}")
    
    # Create different models
    print("\n=== Creating Models ===")
    
    config = {'model_name': 'amgru', 'input_dim': 10, 'hidden_dim': 32}
    model1 = create_model_from_config(config)
    print(f"Created AMGRU model: {model1.__class__.__name__}")
    
    model2 = ModelFactory.create_model('lstm', input_dim=10, hidden_dim=32)
    print(f"Created LSTM model: {model2.__class__.__name__}")
    
    model3 = factory.create_model('bilstm', input_dim=10, hidden_dim=32)
    print(f"Created BiLSTM model: {model3.__class__.__name__}")
    
    print("\n=== Model Architecture ===")
    print(f"\nAMGRU architecture:\n{model1}")
