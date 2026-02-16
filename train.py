"""
Training Module
Train all four models and save results.
"""
import os
import json
import tensorflow as tf
from data_loader import DataLoader
from models import MODEL_REGISTRY

# Set seeds for reproducibility
tf.random.set_seed(42)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

def train_all_models(data_dir, epochs=10, save_dir='trained_models'):
    """
    Train all four models on the dataset.
    
    Args:
        data_dir: Path to dataset directory
        epochs: Number of training epochs
        save_dir: Directory to save trained models
        
    Returns:
        dict: Training results for each model
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Load data
    print("Loading dataset...")
    loader = DataLoader(data_dir)
    train_gen, val_gen = loader.create_generators()
    
    results = {}
    
    # Train each model
    for model_name, create_fn in MODEL_REGISTRY.items():
        print(f"\n{'='*60}")
        print(f"Training {model_name.upper()}")
        print(f"{'='*60}")
        
        try:
            # Create model
            model = create_fn()
            
            # Train
            history = model.fit(
                train_gen,
                epochs=epochs,
                validation_data=val_gen,
                verbose=1
            )
            
            # Save model
            model_path = os.path.join(save_dir, f'{model_name}.h5')
            model.save(model_path)
            print(f"Model saved to {model_path}")
            
            # Store results
            results[model_name] = {
                'train_accuracy': float(history.history['accuracy'][-1]),
                'val_accuracy': float(history.history['val_accuracy'][-1]),
                'train_loss': float(history.history['loss'][-1]),
                'val_loss': float(history.history['val_loss'][-1])
            }
            
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            results[model_name] = {'error': str(e)}
    
    # Save results to JSON
    results_path = os.path.join(save_dir, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(json.dumps(results, indent=2))
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train all models')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--save_dir', type=str, default='trained_models', help='Directory to save models')
    
    args = parser.parse_args()
    
    train_all_models(args.data_dir, args.epochs, args.save_dir)
