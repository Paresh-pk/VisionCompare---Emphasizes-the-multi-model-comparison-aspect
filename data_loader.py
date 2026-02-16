"""
Data Loading Module
Handles image dataset loading, preprocessing, and data generator creation.
"""
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set random seed for reproducibility
np.random.seed(42)

class DataLoader:
    """Simple data loader for cat/dog image classification."""
    
    def __init__(self, data_dir, img_size=(150, 150), batch_size=32):
        """
        Initialize data loader.
        
        Args:
            data_dir: Path to dataset directory (should contain train/ and validation/ subdirs)
            img_size: Target image size (height, width)
            batch_size: Batch size for training
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        
        self.train_dir = os.path.join(data_dir, 'train')
        self.val_dir = os.path.join(data_dir, 'validation')
        
    def create_generators(self):
        """
        Create training and validation data generators.
        
        Returns:
            tuple: (train_generator, validation_generator)
        """
        # Data augmentation for training (optional, can be disabled for reproducibility)
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            # rotation_range=20,
            # width_shift_range=0.2,
            # height_shift_range=0.2,
            # horizontal_flip=True
        )
        
        # Only rescaling for validation
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=True,
            seed=42
        )
        
        val_generator = val_datagen.flow_from_directory(
            self.val_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=False,
            seed=42
        )
        
        return train_generator, val_generator
    
    def preprocess_single_image(self, img_path):
        """
        Preprocess a single image for inference.
        
        Args:
            img_path: Path to image file
            
        Returns:
            numpy array: Preprocessed image ready for model input
        """
        from tensorflow.keras.preprocessing import image
        
        img = image.load_img(img_path, target_size=self.img_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        
        return img_array
