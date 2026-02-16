"""
Evaluation Module
Evaluate all trained models and report accuracy metrics.
"""
import os
import json
import tensorflow as tf
from data_loader import DataLoader

def evaluate_all_models(data_dir, model_dir='trained_models'):
    """
    Evaluate all trained models on validation set.
    
    Args:
        data_dir: Path to dataset directory
        model_dir: Directory containing trained models
        
    Returns:
        dict: Evaluation results for each model
    """
    # Load validation data
    print("Loading validation dataset...")
    loader = DataLoader(data_dir)
    _, val_gen = loader.create_generators()
    
    results = {}
    
    # Evaluate each model
    for filename in os.listdir(model_dir):
        if filename.endswith('.h5'):
            model_name = filename.replace('.h5', '')
            model_path = os.path.join(model_dir, filename)
            
            print(f"\nEvaluating {model_name}...")
            
            try:
                # Load model
                model = tf.keras.models.load_model(model_path)
                
                # Evaluate
                loss, accuracy = model.evaluate(val_gen, verbose=0)
                
                results[model_name] = {
                    'accuracy': float(accuracy),
                    'loss': float(loss)
                }
                
                print(f"  Accuracy: {accuracy*100:.2f}%")
                print(f"  Loss: {loss:.4f}")
                
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
                results[model_name] = {'error': str(e)}
    
    # Find best model
    valid_results = {k: v for k, v in results.items() if 'accuracy' in v}
    if valid_results:
        best_model = max(valid_results, key=lambda x: valid_results[x]['accuracy'])
        best_accuracy = valid_results[best_model]['accuracy']
        
        print(f"\n{'='*60}")
        print(f"BEST MODEL: {best_model}")
        print(f"Accuracy: {best_accuracy*100:.2f}%")
        print(f"{'='*60}")
    
    # Save evaluation results
    eval_path = os.path.join(model_dir, 'evaluation_results.json')
    with open(eval_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to {eval_path}")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate all models')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--model_dir', type=str, default='trained_models', help='Directory with trained models')
    
    args = parser.parse_args()
    
    evaluate_all_models(args.data_dir, args.model_dir)
