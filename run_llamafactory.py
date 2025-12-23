"""
Integration example: Using LlamaFactory with existing Namdm training pipeline
Demonstrates how to use the factory pattern with the existing Train_OPT class
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from llamafactory import LlamaFactory, ModelFactory
from run import DataProcessor, ModelTrainer, Logger, DeltaMixLoss
from bayes_opt import BayesianOptimization
from datetime import datetime
from typing import Dict, Any
import os


class Train_OPT_LlamaFactory:
    """
    Enhanced training class using LlamaFactory for model creation
    Compatible with existing Train_OPT interface
    """
    def __init__(self, file_path: str, init_points: int, n_iter: int, 
                 timesteps: int = 10, shuffle_data: bool = True,
                 model_name: str = 'amgru'):
        self.data_processor = DataProcessor(file_path, timesteps, shuffle_data)
        self.init_points = init_points
        self.n_iter = n_iter
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = Logger()
        self.model_name = model_name
        self.factory = LlamaFactory()
        
        print(f"\n=== Using LlamaFactory ===")
        print(f"Selected model: {model_name}")
        print(f"Available models: {', '.join(ModelFactory.list_models())}")
        
    def train_test(self, delta: float, hidden: float, epoch: float, batch: float) -> float:
        """Train and test model using LlamaFactory"""
        # Record training parameters
        params = {
            'model_name': self.model_name,
            'delta': delta,
            'hidden_dim': int(hidden),
            'epochs': int(epoch),
            'batch_size': int(batch),
            'device': str(self.device),
            'shuffle_data': self.data_processor.shuffle_data,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.logger.log_parameters(params)
        
        # Data preparation
        train_data, test_data = self.data_processor.split_data()
        train_x, train_y = self.data_processor.process_train_data(train_data)
        test_x, test_y = self.data_processor.process_test_data(test_data)

        # Create data loader
        train_dataset = TensorDataset(train_x, train_y)
        train_loader = DataLoader(train_dataset, batch_size=int(batch), shuffle=True)

        # Initialize model using LlamaFactory
        try:
            model = self.factory.create_model(
                model_name=self.model_name,
                input_dim=train_x.shape[2],
                hidden_dim=int(hidden)
            )
            print(f"Created {self.model_name} model using LlamaFactory")
        except ValueError as e:
            print(f"Error creating model: {e}")
            raise
            
        trainer = ModelTrainer(self.device, self.logger, delta=delta)
        optimizer = optim.Adam(model.parameters(), lr=0.0005)

        # Train and evaluate
        trainer.train(model, train_loader, optimizer, int(epoch))
        return trainer.evaluate(model, test_x, test_y, self.data_processor.scaler_y)

    def BOA(self) -> None:
        """Execute Bayesian Optimization"""
        pbounds = {
            'delta': (0, 1),
            'hidden': (3, 32),
            'epoch': (300, 600),
            'batch': (4, 64),
        }

        optimizer = BayesianOptimization(
            f=self.train_test,
            pbounds=pbounds,
            verbose=2,
            random_state=1,
        )
        optimizer.maximize(
            init_points=self.init_points,
            n_iter=self.n_iter,
        )
        
        # Record best results
        best_metrics = {
            'model_name': self.model_name,
            'best_score': float(optimizer.max['target']),
            'best_params_delta': float(optimizer.max['params']['delta']),
            'best_params_hidden': float(optimizer.max['params']['hidden']),
            'best_params_epoch': float(optimizer.max['params']['epoch']),
            'best_params_batch': float(optimizer.max['params']['batch'])
        }
        self.logger.log_metrics(best_metrics)
        self.logger.save_final_results()
        print(optimizer.max)


def compare_models(file_path: str, init_points: int = 3, n_iter: int = 5):
    """
    Compare different model architectures using LlamaFactory
    
    Args:
        file_path: Path to data file
        init_points: Initial points for Bayesian optimization
        n_iter: Number of iterations for Bayesian optimization
    """
    factory = LlamaFactory()
    models_to_compare = factory.list_available_models()
    
    print("\n" + "="*60)
    print("Model Comparison using LlamaFactory")
    print("="*60)
    
    results = {}
    
    for model_name in models_to_compare:
        print(f"\n{'='*60}")
        print(f"Training model: {model_name}")
        print(f"{'='*60}")
        
        trainer = Train_OPT_LlamaFactory(
            file_path=file_path,
            init_points=init_points,
            n_iter=n_iter,
            timesteps=10,
            shuffle_data=True,
            model_name=model_name
        )
        
        trainer.BOA()
        results[model_name] = trainer.logger.results
    
    print("\n" + "="*60)
    print("Comparison Complete!")
    print("="*60)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train models using LlamaFactory')
    parser.add_argument('--file_path', type=str, default='yixi.csv',
                       help='Path to data file')
    parser.add_argument('--model', type=str, default='amgru',
                       choices=ModelFactory.list_models(),
                       help='Model architecture to use')
    parser.add_argument('--init_points', type=int, default=5,
                       help='Initial points for Bayesian optimization')
    parser.add_argument('--n_iter', type=int, default=25,
                       help='Number of iterations for Bayesian optimization')
    parser.add_argument('--compare', action='store_true',
                       help='Compare all available models')
    
    args = parser.parse_args()
    
    if args.compare:
        # Compare all models
        compare_models(
            file_path=args.file_path,
            init_points=args.init_points,
            n_iter=args.n_iter
        )
    else:
        # Train single model
        trainer = Train_OPT_LlamaFactory(
            file_path=args.file_path,
            init_points=args.init_points,
            n_iter=args.n_iter,
            timesteps=10,
            shuffle_data=True,
            model_name=args.model
        )
        trainer.BOA()
