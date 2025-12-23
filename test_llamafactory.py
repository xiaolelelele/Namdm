"""
Tests for LlamaFactory
"""
import torch
from llamafactory import ModelFactory, LlamaFactory, create_model_from_config


def test_list_models():
    """Test listing available models"""
    models = ModelFactory.list_models()
    assert isinstance(models, list)
    assert len(models) > 0
    assert 'amgru' in models
    assert 'lstm' in models


def test_create_amgru():
    """Test creating AMGRU model"""
    model = ModelFactory.create_model('amgru', input_dim=10, hidden_dim=32)
    assert model is not None
    assert model.__class__.__name__ == 'AMGRU'
    
    # Test forward pass
    x = torch.randn(4, 10, 10)  # batch_size=4, seq_len=10, input_dim=10
    output = model(x)
    assert output.shape == (4, 1)


def test_create_lstm():
    """Test creating LSTM model"""
    model = ModelFactory.create_model('lstm', input_dim=10, hidden_dim=16)
    assert model is not None
    assert model.__class__.__name__ == 'LSTMModel'
    
    # Test forward pass
    x = torch.randn(4, 10, 10)
    output = model(x)
    assert output.shape == (4, 1)


def test_create_gru():
    """Test creating GRU model"""
    model = ModelFactory.create_model('gru', input_dim=10, hidden_dim=16)
    assert model is not None
    assert model.__class__.__name__ == 'GRUModel'


def test_create_bilstm():
    """Test creating BiLSTM model"""
    model = ModelFactory.create_model('bilstm', input_dim=10, hidden_dim=16)
    assert model is not None
    assert model.__class__.__name__ == 'BILSTMModel'


def test_create_amlstm():
    """Test creating AMLSTM model"""
    model = ModelFactory.create_model('amlstm', input_dim=10, hidden_dim=32)
    assert model is not None
    assert model.__class__.__name__ == 'AMLSTM'


def test_invalid_model():
    """Test error handling for invalid model name"""
    try:
        ModelFactory.create_model('invalid_model', input_dim=10, hidden_dim=32)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert 'not found' in str(e)


def test_llamafactory_init():
    """Test LlamaFactory initialization"""
    factory = LlamaFactory()
    assert factory is not None


def test_llamafactory_with_config():
    """Test LlamaFactory with configuration"""
    config = {
        'model_name': 'lstm',
        'input_dim': 10,
        'hidden_dim': 16
    }
    factory = LlamaFactory(config)
    model = factory.create_model()
    assert model is not None
    assert model.__class__.__name__ == 'LSTMModel'


def test_create_model_from_config():
    """Test convenience function"""
    config = {
        'model_name': 'gru',
        'input_dim': 10,
        'hidden_dim': 16
    }
    model = create_model_from_config(config)
    assert model is not None
    assert model.__class__.__name__ == 'GRUModel'


def test_get_model_info():
    """Test getting model information"""
    factory = LlamaFactory()
    info = factory.get_model_info()
    assert isinstance(info, dict)
    assert 'amgru' in info
    assert 'lstm' in info


if __name__ == "__main__":
    # Run tests manually if pytest is not available
    print("Running LlamaFactory tests...\n")
    
    tests = [
        test_list_models,
        test_create_amgru,
        test_create_lstm,
        test_create_gru,
        test_create_bilstm,
        test_create_amlstm,
        test_invalid_model,
        test_llamafactory_init,
        test_llamafactory_with_config,
        test_create_model_from_config,
        test_get_model_info,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            print(f"✓ {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: {e}")
            failed += 1
    
    print(f"\n{passed} passed, {failed} failed")
