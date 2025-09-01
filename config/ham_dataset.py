import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import albumentations as A

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
        # print(len(self.image_files))
        # print(np.sum(df['image_id'].isin(self.image_files)))
        # print(len(df))
        assert np.sum(df['image_id'].isin(self.image_files)) == len(df)
        self.image_files = df['image_id'].values.tolist()

        # Create a mapping from image_id to class label
        # self.imageid_to_label = dict(zip(self.df['image_id'], self.df['dx']))

        # Map class names to integer indices
        classes = df['dx'].unique()
        # classes = sorted(df['dx'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        self.labels = [self.class_to_idx[cls] for cls in df['dx'].values]
        self.output_classes = output_classes

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # filename = self.image_files[index]
        # image_id = os.path.splitext(filename)[0]
        # img_path = os.path.join(self.root, filename)
        # image = np.asarray(Image.open(img_path).convert("RGB"))
        # label_name = self.imageid_to_label[image_id]
        # label = self.class_to_idx[label_name]
        # if self.transform:
        #     # If the transform is SelectiveAugment, pass label_name as well 
        #     if hasattr(self.transform, '__class__') and self.transform.__class__.__name__ == "SelectiveAugment":
        #         image = self.transform(image, label_name)
        #     else:
        #         image = self.transform(image = image)
        # if isinstance(image, dict):
        #     image = image['image']
        # return image.to(torch.float32), label
        img_path = os.path.join(self.root, f'{self.image_files[index]}.jpg')
        image = np.asarray(Image.open(img_path).convert("RGB"))
        label = self.labels[index]
        if self.transform:
            image = self.transform(image = image)
        if isinstance(image, dict):
            image = image['image']
        # print('Index :', index, 'Size :', image.size())
        return image.to(torch.float32), label
    

# # NOTE: This was for the selective augmentation (only augments on classes with less image counts than the median)
# # Add augmentations to data that needs it
# # Find classes with fewer image counts (e.g., less than median count)
# class_counts = df['dx'].value_counts()
# median_count = class_counts.median()
# minority_classes = class_counts[class_counts < median_count].index.tolist()
# print(f"Median class count: {median_count}")
# print(f"Minority classes (less than median count): {minority_classes}")
# minority_class_names = [diagnosis_map[c] for c in minority_classes]
# print(f"Minority class diagnosis names: {minority_class_names}")

# Define transforms
get_basic_transform =lambda x_dim, y_dim, channel_size : A.Compose([
    A.Resize(y_dim,x_dim),
    *([A.ToGray(channel_size,p=1),] if  channel_size == 1 else []),
    # A.Normalize(normalization='min_max'),
    # A.Normalize(mean=0.5, std=0.5),
    A.Normalize(),
    A.ToTensorV2(),
])

# Define the augmentation pipeline
get_augmented_transform = lambda x_dim, y_dim, channel_size, probability=0.25 :  A.Compose([
    A.Resize(y_dim,x_dim),
    A.RandomResizedCrop(size=(x_dim,y_dim), scale=(0.08, 1.0), p=probability),
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
    # A.Mosaic(p=probability),
    *([A.ToGray(channel_size,p=1),] if  channel_size == 1 else []),
    # A.Normalize(normalization='min_max'),
    # A.Normalize(mean=0.5, std=0.5),
    A.Normalize(),
    A.ToTensorV2(),
])
