"""
Training script using LlamaFactory
Provides a unified interface for training different model architectures
"""
import argparse
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from llamafactory import LlamaFactory, ModelFactory
from typing import Dict, Any
import os


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)


def train_model_from_config(config_path: str = None, config_dict: Dict[str, Any] = None):
    """
    Train a model using configuration
    
    Args:
        config_path: Path to configuration JSON file
        config_dict: Configuration dictionary (alternative to config_path)
    """
    # Load configuration
    if config_path:
        config = load_config(config_path)
    elif config_dict:
        config = config_dict
    else:
        raise ValueError("Either config_path or config_dict must be provided")
    
    # Extract configurations
    model_config = config.get('model_config', {})
    training_config = config.get('training_config', {})
    data_config = config.get('data_config', {})
    logging_config = config.get('logging_config', {})
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model using LlamaFactory
    factory = LlamaFactory(config)
    model_name = config.get('model_name', 'amgru')
    
    print(f"\n=== Creating model: {model_name} ===")
    print(f"Model configuration: {model_config}")
    
    model = factory.create_model(
        model_name=model_name,
        **model_config
    )
    model.to(device)
    
    print(f"Model created successfully: {model.__class__.__name__}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Display model architecture
    print(f"\nModel architecture:\n{model}")
    
    return model, config


def train_with_llamafactory(
    model_name: str,
    input_dim: int,
    hidden_dim: int,
    epochs: int = 500,
    batch_size: int = 32,
    learning_rate: float = 0.0005,
    **kwargs
):
    """
    Convenience function to train a model with LlamaFactory
    
    Args:
        model_name: Name of model architecture
        input_dim: Input dimension
        hidden_dim: Hidden layer dimension
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        **kwargs: Additional parameters
    """
    config = {
        'model_name': model_name,
        'model_config': {
            'input_dim': input_dim,
            'hidden_dim': hidden_dim
        },
        'training_config': {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate
        }
    }
    config.update(kwargs)
    
    return train_model_from_config(config_dict=config)


def main():
    parser = argparse.ArgumentParser(description='Train models using LlamaFactory')
    parser.add_argument('--config', type=str, help='Path to configuration JSON file')
    parser.add_argument('--model', type=str, default='amgru', 
                       choices=ModelFactory.list_models(),
                       help='Model architecture to use')
    parser.add_argument('--input_dim', type=int, default=10, help='Input dimension')
    parser.add_argument('--hidden_dim', type=int, default=32, help='Hidden dimension')
    parser.add_argument('--list_models', action='store_true', help='List available models')
    
    args = parser.parse_args()
    
    if args.list_models:
        print("=== Available Models ===")
        factory = LlamaFactory()
        for name, description in factory.get_model_info().items():
            print(f"  {name:10s} - {description}")
        return
    
    if args.config:
        # Train from config file
        model, config = train_model_from_config(config_path=args.config)
    else:
        # Train with command line arguments
        model, config = train_with_llamafactory(
            model_name=args.model,
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim
        )
    
    print("\n=== Training would start here ===")
    print("To integrate with existing training pipeline, import Train_OPT from run.py")


if __name__ == "__main__":
    main()
