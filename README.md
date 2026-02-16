# Cat vs Dog Image Classifier

A production-ready multi-model image classification system that trains and compares 4 different CNN architectures to identify cats and dogs in images.

## 🎯 Project Overview

This project implements a complete machine learning pipeline that:
1. **Trains 4 different models** on cat/dog images
2. **Compares their performance** automatically
3. **Selects the best model** based on validation accuracy
4. **Runs inference** on new images using all models and picks the most confident prediction

## 🏆 Best Model: MobileNetV2

**Winner:** MobileNetV2 achieved **95% validation accuracy** (3 epochs, 100 images per class)

### Why MobileNetV2 Won:

| Model | Val Accuracy | Strengths | Weaknesses |
|-------|--------------|-----------|------------|
| **MobileNetV2** 🥇 | **95.00%** | Pre-trained on ImageNet, efficient architecture, excellent for small datasets | None observed |
| VGG16 🥈 | 80.00% | Good transfer learning, stable | Slight overfitting (84% train vs 80% val) |
| ResNet50 🥉 | 52.50% | Deep architecture | Needs more training data |
| Simple CNN | 47.50% | Fast training | Too simple, random guessing level |

**Key Insight:** Transfer learning models (pre-trained on ImageNet) significantly outperform custom architectures on small datasets.

## 📊 How the System Works

### 1. Data Loading (`data_loader.py`)
- Loads images from `train/` and `validation/` directories
- Resizes all images to 150x150 pixels
- Normalizes pixel values to [0, 1] range
- Creates batches for efficient training

### 2. Model Definitions (`models.py`)
Four distinct architectures:

**Simple CNN:**
```
Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Conv2D(128) → MaxPool → Dense(128) → Output
```

**VGG16 Transfer Learning:**
```
VGG16(frozen) → Flatten → Dense(256) → Dropout(0.5) → Output
```

**ResNet50 Transfer Learning:**
```
ResNet50(frozen) → GlobalAvgPool → Dense(256) → Dropout(0.5) → Output
```

**MobileNetV2 Transfer Learning:**
```
MobileNetV2(frozen) → GlobalAvgPool → Dense(128) → Dropout(0.5) → Output
```

### 3. Training Process (`train.py`)
```python
For each model:
    1. Create model architecture
    2. Compile with Adam optimizer + binary cross-entropy loss
    3. Train on training data
    4. Validate on validation data
    5. Save model weights (.h5 file)
    6. Record accuracy metrics
```

**Output:** `training_results.json` with accuracy/loss for each model

### 4. Model Comparison Logic

**During Training:**
```python
# Automatic best model selection
best_model = max(results, key=lambda x: results[x]['val_accuracy'])
```

**During Inference:**
```python
# Run ALL models on input image
for model in [simple_cnn, vgg16, resnet50, mobilenet]:
    prediction = model.predict(image)
    confidence = calculate_confidence(prediction)
    
# Pick the model with HIGHEST confidence
best_prediction = max(predictions, key=lambda x: x['confidence'])

# Apply 60% confidence threshold
if best_prediction.confidence < 0.60:
    return "Not Found"
else:
    return best_prediction.label
```

### 5. Evaluation (`evaluate.py`)
- Loads all trained models
- Runs evaluation on validation set
- Reports accuracy and loss for each model
- Identifies the best performer

### 6. Inference (`inference.py`)
- Accepts a single image path
- Runs prediction through **all 4 models**
- Displays each model's prediction and confidence
- **Selects the most confident prediction** as final result
- Applies 60% confidence threshold for "Not Found" cases

## 🚀 Installation & Setup

### Prerequisites
```bash
Python 3.8+
```

### Install Dependencies
```bash
cd refactored
pip install -r requirements.txt
```

### Dataset Structure
```
your_dataset/
├── train/
│   ├── cat/
│   │   ├── cat.1.jpg
│   │   ├── cat.2.jpg
│   │   └── ...
│   └── dog/
│       ├── dog.1.jpg
│       ├── dog.2.jpg
│       └── ...
└── validation/
    ├── cat/
    └── dog/
```

## 📖 Usage Guide

### Option 1: Complete Pipeline (Recommended)
```bash
python run_pipeline.py --data_dir path/to/dataset --epochs 3 --test_image path/to/test.jpg
```

### Option 2: Step-by-Step Execution

**Step 1: Prepare Dataset Subset (Optional)**
```bash
python prepare_subset.py --source "C:\path\to\full\dataset" --target data_subset --num_images 100
```

**Step 2: Train All Models**
```bash
python train.py --data_dir data_subset --epochs 3 --save_dir trained_models
```

**Step 3: Evaluate Models**
```bash
python evaluate.py --data_dir data_subset --model_dir trained_models
```

**Step 4: Run Inference on New Image**
```bash
python inference.py path/to/image.jpg --data_dir data_subset --model_dir trained_models
```

## 📈 Example Results

### Training Output
```
============================================================
TRAINING COMPLETE
============================================================
{
  "simple_cnn": {
    "val_accuracy": 0.475
  },
  "vgg16": {
    "val_accuracy": 0.800
  },
  "resnet50": {
    "val_accuracy": 0.525
  },
  "mobilenet": {
    "val_accuracy": 0.950  ← BEST MODEL
  }
}
```

### Inference Output
```
Analyzing image: dog.1753.jpg
============================================================
mobilenet       -> Dog (99.98% confidence)  ← WINNER
vgg16           -> Dog (85.58% confidence)
simple_cnn      -> Dog (52.41% confidence)
resnet50        -> Cat (53.18% confidence)
============================================================
BEST PREDICTION: Dog
Model: mobilenet
Confidence: 99.98%
============================================================
```

## 🔬 Technical Details

### Reproducibility
- **Random Seed:** 42 (set in all modules)
- **Deterministic Operations:** Enabled via `TF_DETERMINISTIC_OPS=1`
- **Fixed Data Splits:** Same train/validation split every run

### Model Selection Criteria
1. **Validation Accuracy** (primary metric)
2. **Generalization** (train vs val accuracy gap)
3. **Inference Confidence** (per-image confidence scores)

### Confidence Threshold
- Predictions below **60% confidence** are classified as **"Not Found"**
- This prevents false positives on ambiguous or non-cat/dog images

## 📁 Project Structure

```
refactored/
├── data_loader.py          # Data preprocessing
├── models.py               # 4 model architectures
├── train.py                # Training script
├── evaluate.py             # Evaluation script
├── inference.py            # Single-image prediction
├── run_pipeline.py         # End-to-end CLI
├── prepare_subset.py       # Dataset sampling utility
├── requirements.txt        # Dependencies
├── README.md              # This file
├── data_subset/           # Training data (created by prepare_subset.py)
└── trained_models/        # Saved model weights
    ├── simple_cnn.h5
    ├── vgg16.h5
    ├── resnet50.h5
    ├── mobilenet.h5
    ├── training_results.json
    └── evaluation_results.json
```

## 🎓 Key Learnings

1. **Transfer Learning is Powerful:** Pre-trained models (MobileNetV2, VGG16) vastly outperform custom CNNs on small datasets
2. **Model Ensemble Approach:** Running multiple models and selecting the most confident prediction improves reliability
3. **Lightweight Models Win:** MobileNetV2 achieved best accuracy despite being the most efficient architecture
4. **Data Quality > Quantity:** 100 well-chosen images per class can train effective models with transfer learning

## 🔧 Customization

### Change Number of Epochs
```bash
python train.py --data_dir data_subset --epochs 10
```

### Adjust Confidence Threshold
```bash
python inference.py image.jpg --data_dir data_subset --threshold 0.7
```

### Use Different Dataset Size
```bash
python prepare_subset.py --source original_dataset --num_images 200
```

## 📊 Performance Metrics

| Metric | MobileNetV2 | VGG16 | ResNet50 | Simple CNN |
|--------|-------------|-------|----------|------------|
| Val Accuracy | **95.00%** | 80.00% | 52.50% | 47.50% |
| Val Loss | **0.099** | 0.464 | 0.689 | 0.691 |
| Inference Speed | Fast | Medium | Slow | Very Fast |
| Model Size | Small (9MB) | Large (76MB) | Very Large (101MB) | Medium (28MB) |

## 🚀 Next Steps

1. **Increase Training Data:** Use full dataset (4000 images) for production
2. **More Epochs:** Train for 10-15 epochs for better convergence
3. **Fine-tuning:** Unfreeze top layers of MobileNetV2 for marginal improvements
4. **Data Augmentation:** Enable rotation, flipping, zoom in `data_loader.py`
5. **Deploy:** Integrate best model into web app or mobile application

## 📝 License

This project is for educational purposes.

## 👨‍💻 Author

Data Science Engineer - Image Classification Project

---

**Note:** This implementation prioritizes clean code, reproducibility, and educational value. All components are well-documented and modular for easy understanding and extension.
