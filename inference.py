"""
Inference Module
Run single-image prediction through all models and compare results.
"""
import os
import json
import tensorflow as tf
from data_loader import DataLoader

def predict_single_image(img_path, data_dir, model_dir='trained_models', confidence_threshold=0.6):
    """
    Run inference on a single image using all trained models.
    
    Args:
        img_path: Path to input image
        data_dir: Path to dataset directory (for data loader initialization)
        model_dir: Directory containing trained models
        confidence_threshold: Minimum confidence to accept prediction
        
    Returns:
        dict: Predictions from all models with best result highlighted
    """
    # Preprocess image
    loader = DataLoader(data_dir)
    img_array = loader.preprocess_single_image(img_path)
    
    predictions = {}
    best_confidence = 0.0
    best_model = None
    best_label = "Unknown"
    
    print(f"\nAnalyzing image: {img_path}")
    print(f"{'='*60}")
    
    # Run inference with each model
    for filename in os.listdir(model_dir):
        if filename.endswith('.h5'):
            model_name = filename.replace('.h5', '')
            model_path = os.path.join(model_dir, filename)
            
            try:
                # Load model
                model = tf.keras.models.load_model(model_path)
                
                # Predict
                raw_prediction = model.predict(img_array, verbose=0)[0][0]
                
                # Binary classification: >0.5 = Dog, <0.5 = Cat
                if raw_prediction > 0.5:
                    label = "Dog"
                    confidence = float(raw_prediction)
                else:
                    label = "Cat"
                    confidence = float(1.0 - raw_prediction)
                
                predictions[model_name] = {
                    'label': label,
                    'confidence': confidence,
                    'raw_score': float(raw_prediction)
                }
                
                print(f"{model_name:15s} -> {label:3s} ({confidence*100:5.2f}% confidence)")
                
                # Track best prediction
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_model = model_name
                    best_label = label
                    
            except Exception as e:
                print(f"{model_name:15s} -> Error: {e}")
                predictions[model_name] = {'error': str(e)}
    
    # Apply confidence threshold
    if best_confidence < confidence_threshold:
        final_label = "Not Found (Low Confidence)"
    else:
        final_label = best_label
    
    print(f"{'='*60}")
    print(f"BEST PREDICTION: {final_label}")
    print(f"Model: {best_model}")
    print(f"Confidence: {best_confidence*100:.2f}%")
    print(f"{'='*60}")
    
    # Compile results
    result = {
        'image': img_path,
        'predictions': predictions,
        'best_model': best_model,
        'best_label': final_label,
        'best_confidence': best_confidence
    }
    
    return result

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run inference on a single image')
    parser.add_argument('image_path', type=str, help='Path to input image')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--model_dir', type=str, default='trained_models', help='Directory with trained models')
    parser.add_argument('--threshold', type=float, default=0.6, help='Confidence threshold')
    
    args = parser.parse_args()
    
    result = predict_single_image(args.image_path, args.data_dir, args.model_dir, args.threshold)
    
    # Save result
    output_path = 'inference_result.json'
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=4)
    print(f"\nDetailed results saved to {output_path}")
