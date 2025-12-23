# LlamaFactory for Namdm

## Overview

LlamaFactory provides a unified, flexible interface for creating and training different neural network architectures in the Namdm project. Inspired by the factory design pattern, it simplifies model instantiation and training workflows.

## Architecture

### Components

1. **ModelFactory**: Core factory class for model instantiation
2. **LlamaFactory**: High-level interface for configuration-based model creation
3. **Integration Scripts**: Bridge between LlamaFactory and existing training pipeline

### Supported Models

| Model Name | Description | Use Case |
|------------|-------------|----------|
| `amgru` | Attention Mechanism with GRU | Time-series with attention |
| `amlstm` | Attention Mechanism with LSTM | Long-term dependencies with attention |
| `bilstm` | Bidirectional LSTM | Bidirectional context |
| `gru` | Gated Recurrent Unit | Faster training than LSTM |
| `lstm` | Long Short-Term Memory | Long-term dependencies |

## Usage

### Basic Model Creation

```python
from llamafactory import ModelFactory

# Create a model directly
model = ModelFactory.create_model(
    model_name='amgru',
    input_dim=10,
    hidden_dim=32
)
```

### Using LlamaFactory Class

```python
from llamafactory import LlamaFactory

# Initialize with configuration
config = {
    'model_name': 'lstm',
    'model_config': {
        'input_dim': 10,
        'hidden_dim': 16
    }
}

factory = LlamaFactory(config)
model = factory.create_model()
```

### Configuration-Based Training

Create a JSON configuration file:

```json
{
  "model_name": "amgru",
  "model_config": {
    "input_dim": 10,
    "hidden_dim": 32
  },
  "training_config": {
    "epochs": 500,
    "batch_size": 32,
    "learning_rate": 0.0005,
    "optimizer": "adam",
    "loss_function": "deltamix",
    "delta": 0.5
  },
  "data_config": {
    "file_path": "yixi.csv",
    "timesteps": 10,
    "train_split": 0.8,
    "shuffle_data": true
  }
}
```

Then train:

```bash
python train_llamafactory.py --config my_config.json
```

### Command-Line Training

```bash
# Train with specific model
python train_llamafactory.py --model lstm --input_dim 10 --hidden_dim 16

# List available models
python train_llamafactory.py --list_models
```

### Integration with Bayesian Optimization

```python
from run_llamafactory import Train_OPT_LlamaFactory

# Train with Bayesian optimization
trainer = Train_OPT_LlamaFactory(
    file_path='yixi.csv',
    init_points=5,
    n_iter=25,
    model_name='amgru'
)
trainer.BOA()
```

### Model Comparison

```python
from run_llamafactory import compare_models

# Compare all available models
results = compare_models(
    file_path='yixi.csv',
    init_points=3,
    n_iter=5
)
```

Or via command line:

```bash
python run_llamafactory.py --compare --init_points 3 --n_iter 5
```

## Advanced Features

### Registering Custom Models

```python
import torch.nn as nn
from llamafactory import ModelFactory

class CustomModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Register the custom model
ModelFactory.register_model('custom', CustomModel)

# Now you can use it
model = ModelFactory.create_model('custom', input_dim=10, hidden_dim=32)
```

### Getting Model Information

```python
from llamafactory import LlamaFactory

factory = LlamaFactory()

# List all models
models = factory.list_available_models()
print(models)  # ['amgru', 'amlstm', 'bilstm', 'gru', 'lstm']

# Get detailed information
info = factory.get_model_info()
for name, description in info.items():
    print(f"{name}: {description}")
```

## Configuration Schema

### Complete Configuration Example

```json
{
  "model_name": "amgru",
  "model_config": {
    "input_dim": 10,
    "hidden_dim": 32
  },
  "training_config": {
    "epochs": 500,
    "batch_size": 32,
    "learning_rate": 0.0005,
    "optimizer": "adam",
    "loss_function": "deltamix",
    "delta": 0.5
  },
  "data_config": {
    "file_path": "yixi.csv",
    "timesteps": 10,
    "train_split": 0.8,
    "shuffle_data": true,
    "normalize": true
  },
  "optimization_config": {
    "use_bayesian_optimization": true,
    "init_points": 5,
    "n_iter": 25,
    "pbounds": {
      "delta": [0, 1],
      "hidden": [3, 32],
      "epoch": [300, 600],
      "batch": [4, 64]
    }
  },
  "logging_config": {
    "log_dir": "logs",
    "log_interval": 50,
    "save_model": true,
    "model_save_path": "checkpoints"
  }
}
```

## Examples

### Example 1: Train LSTM Model

```bash
python train_llamafactory.py \
  --model lstm \
  --input_dim 10 \
  --hidden_dim 16
```

### Example 2: Compare Models

```bash
python run_llamafactory.py \
  --compare \
  --file_path yixi.csv \
  --init_points 3 \
  --n_iter 5
```

### Example 3: Train with Custom Configuration

```bash
python train_llamafactory.py --config config_example.json
```

## Integration with Existing Code

LlamaFactory is designed to work seamlessly with the existing Namdm codebase:

```python
# Import existing components
from run import DataProcessor, ModelTrainer, Logger

# Import LlamaFactory
from llamafactory import LlamaFactory

# Create model using LlamaFactory
factory = LlamaFactory()
model = factory.create_model('amgru', input_dim=10, hidden_dim=32)

# Use with existing training pipeline
data_processor = DataProcessor('yixi.csv', timesteps=10)
trainer = ModelTrainer(device, logger, delta=0.5)
# ... continue with existing workflow
```

## Benefits

1. **Consistency**: Unified interface for all model architectures
2. **Flexibility**: Easy to switch between models
3. **Extensibility**: Simple to add new models
4. **Maintainability**: Centralized model creation logic
5. **Configuration-Driven**: Reproducible experiments
6. **Comparison**: Easy model architecture comparison

## Best Practices

1. **Use Configuration Files**: For reproducible experiments, always use configuration files
2. **Model Comparison**: Use the comparison feature to find the best architecture
3. **Custom Models**: Register custom models for project-specific architectures
4. **Logging**: Always enable logging to track experiment results
5. **Parameter Search**: Use Bayesian optimization for hyperparameter tuning

## Troubleshooting

### Model Creation Errors

```python
# If you get a "Model not found" error, check available models:
from llamafactory import LlamaFactory
factory = LlamaFactory()
print(factory.list_available_models())
```

### Import Errors

Make sure all dependencies are installed:

```bash
pip install -r requirements.txt
```

### Configuration Errors

Validate your configuration file:

```python
import json
with open('config.json', 'r') as f:
    config = json.load(f)
    print("Configuration is valid")
```

## API Reference

### ModelFactory

```python
class ModelFactory:
    @classmethod
    def create_model(cls, model_name: str, input_dim: int, 
                    hidden_dim: int, **kwargs) -> nn.Module
    
    @classmethod
    def register_model(cls, name: str, model_class: Type[nn.Module]) -> None
    
    @classmethod
    def list_models(cls) -> list
    
    @classmethod
    def get_model_info(cls) -> Dict[str, str]
```

### LlamaFactory

```python
class LlamaFactory:
    def __init__(self, config: Optional[Dict[str, Any]] = None)
    
    def create_model(self, model_name: str = None, **kwargs) -> nn.Module
    
    def list_available_models(self) -> list
    
    def get_model_info(self) -> Dict[str, str]
```

## Future Enhancements

- [ ] Support for ensemble models
- [ ] Automatic model selection based on data characteristics
- [ ] Integration with MLflow for experiment tracking
- [ ] Pre-trained model loading
- [ ] Model export for production deployment

## Contributing

To add a new model to LlamaFactory:

1. Create your model class in the `models/` directory
2. Add it to `models/__init__.py`
3. Register it in `llamafactory.py`
4. Add documentation and tests

## License

This project follows the same license as Namdm.
