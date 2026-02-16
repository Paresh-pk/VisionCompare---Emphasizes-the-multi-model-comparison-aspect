"""
End-to-End Pipeline
CLI script to run the complete workflow: train → evaluate → inference
"""
import argparse
from train import train_all_models
from evaluate import evaluate_all_models
from inference import predict_single_image

def run_full_pipeline(data_dir, test_image=None, epochs=10):
    """
    Execute the complete pipeline.
    
    Args:
        data_dir: Path to dataset directory
        test_image: Optional path to test image for inference
        epochs: Number of training epochs
    """
    print("\n" + "="*60)
    print("STEP 1: TRAINING ALL MODELS")
    print("="*60)
    train_results = train_all_models(data_dir, epochs=epochs)
    
    print("\n" + "="*60)
    print("STEP 2: EVALUATING ALL MODELS")
    print("="*60)
    eval_results = evaluate_all_models(data_dir)
    
    if test_image:
        print("\n" + "="*60)
        print("STEP 3: RUNNING INFERENCE ON TEST IMAGE")
        print("="*60)
        inference_result = predict_single_image(test_image, data_dir)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run complete ML pipeline')
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='Path to dataset directory (must contain train/ and validation/ subdirs)')
    parser.add_argument('--epochs', type=int, default=10, 
                        help='Number of training epochs (default: 10)')
    parser.add_argument('--test_image', type=str, default=None,
                        help='Optional: path to test image for inference')
    
    args = parser.parse_args()
    
    run_full_pipeline(args.data_dir, args.test_image, args.epochs)
