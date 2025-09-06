import torch
import pandas as pd
import numpy as np
import os
import json
import os
from PIL import Image
import albumentations as A

from configs.checkpoint import read_config

"""
SkinCancerDataset module for HAM10000 skin lesion classification.

This module defines the SkinCancerDataset class for loading and preprocessing dermoscopic images and 
labels from the HAM10000 dataset. It also provides transformation pipelines for basic and augmented 
data augmentation using Albumentations.

Key features:
    - Loads image and label data from CSV and image folder
    - Maps lesion classes to integer indices
    - Uses Albumentations for preprocessing and augmentation  
    - Transformation parameters (x_dim, y_dim, channel_size, probability) are loaded from config.json 
    if not explicitly provided, allowing easy adjustment of preprocessing settings without code changes

Usage:
    - Edit config.json to change image size, channels, or augmentation probability
    - Use basic_transform or augmented_transform for different training scenarios
    - Instantiate SkinCancerDataset for PyTorch DataLoader
"""

class SkinCancerDataset(torch.utils.data.Dataset):
    def __init__(self, root, csv_path, output_classes, transform=None):

        self.root = root
        self.transform = transform
        if isinstance(csv_path, str):
            df = pd.read_csv(csv_path)
        else:
            df = csv_path
        df = df.reindex()
        self.image_files = [os.path.splitext(f)[0] for f in os.listdir(root) if f.endswith('.jpg')]

        assert np.sum(df['image_id'].isin(self.image_files)) == len(df)
        self.image_files = df['image_id'].values.tolist()
        
        classes = df['dx'].unique() # Map class names to integer indices
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        self.labels = [self.class_to_idx[cls] for cls in df['dx'].values]
        self.output_classes = output_classes

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img_path = os.path.join(self.root, f'{self.image_files[index]}.jpg')
        image = np.asarray(Image.open(img_path).convert("RGB"))
        label = self.labels[index]
        if self.transform:
            image = self.transform(image = image)
        if isinstance(image, dict):
            image = image['image']
        return image.to(torch.float32), label


# read config.json and try to get transform parameters
# TODO: BUG: make sure the user knows where to put config.json
config = read_config(os.getcwd())
x_dim = config['x_dim']
y_dim = config['y_dim']
channel_size = config['channel_size']
probability = config['probability'] if 'probability' in config else 0.0


basic_transform = A.Compose([
    A.Resize(y_dim, x_dim),
    *([A.ToGray(channel_size, p=1),] if channel_size == 1 else []),
    A.Normalize(),
    A.ToTensorV2(),
])


augmented_transform = A.Compose([
    A.Resize(y_dim, x_dim),
    A.RandomResizedCrop(size=(x_dim, y_dim), scale=(0.08, 1.0), p=probability),
    A.HorizontalFlip(p=probability),
    A.VerticalFlip(p=probability),
    A.RGBShift(p=probability),
    A.RandomSunFlare(p=probability),
    A.RandomBrightnessContrast(p=probability),
    A.HueSaturationValue(p=probability),
    A.ColorJitter(p=probability),
    A.RandomRotate90(p=probability),
    A.Perspective(p=probability),
    A.MotionBlur(p=probability),
    A.ChannelShuffle(p=probability),
    A.ChannelDropout(p=probability),
    *([A.ToGray(channel_size, p=1),] if channel_size == 1 else []),
    A.Normalize(),
    A.ToTensorV2(),
])
