#!/usr/bin/env python
"""
Quick demonstration of LlamaFactory functionality
"""
from llamafactory import LlamaFactory, ModelFactory
import torch

print("="*70)
print("LlamaFactory Demonstration for Namdm")
print("="*70)

# 1. List available models
print("\n1. Available Models:")
print("-" * 70)
factory = LlamaFactory()
for name, desc in factory.get_model_info().items():
    print(f"   • {name:10s} - {desc}")

# 2. Create models
print("\n2. Creating Models:")
print("-" * 70)
models = {}
for model_name in ['amgru', 'lstm', 'gru']:
    model = ModelFactory.create_model(model_name, input_dim=10, hidden_dim=16)
    models[model_name] = model
    params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ {model_name:10s}: {model.__class__.__name__:15s} ({params:,} parameters)")

# 3. Test forward pass
print("\n3. Testing Forward Pass:")
print("-" * 70)
batch_size, seq_len, input_dim = 4, 10, 10
test_input = torch.randn(batch_size, seq_len, input_dim)
print(f"   Input shape: {tuple(test_input.shape)}")

for name, model in models.items():
    output = model(test_input)
    print(f"   • {name:10s} output shape: {tuple(output.shape)}")

# 4. Configuration-based creation
print("\n4. Configuration-Based Model Creation:")
print("-" * 70)
config = {
    'model_name': 'bilstm',
    'model_config': {
        'input_dim': 10,
        'hidden_dim': 24
    }
}
factory_with_config = LlamaFactory(config)
model = factory_with_config.create_model()
params = sum(p.numel() for p in model.parameters())
print(f"   ✓ Created {model.__class__.__name__} from config ({params:,} parameters)")

print("\n" + "="*70)
print("✓ All demonstrations completed successfully!")
print("="*70)
print("\nNext steps:")
print("  • Train models: python train_llamafactory.py --model lstm")
print("  • Use config:   python train_llamafactory.py --config config_example.json")
print("  • Compare all:  python run_llamafactory.py --compare")
print("="*70)
