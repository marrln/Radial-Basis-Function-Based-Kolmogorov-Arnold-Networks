import os, sys
import torch
from PIL import Image
import numpy as np
import pandas as pd 
from torch.utils.data import random_split
import albumentations as A
from tqdm import tqdm

os.chdir(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Local
from quantization.parameter_transform import save_tensor_to_bin, cumulate_files

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

dataset_path = r"../Dataset"
csv_path = os.path.join(dataset_path, "HAM10000_metadata.csv") # NOTE: Need to change manually the extension of the file to csv after downloading
csv_test_path = os.path.join(dataset_path, "ISIC2018_Task3_Test_GroundTruth.csv") # NOTE: Need to change manually the extension of the file to csv after downloading

df = pd.read_csv(csv_path)

# check how many images we have
image_dir = os.path.join(dataset_path, "HAM10000_images")
image_bin_dir = os.path.join(dataset_path, "HAM10000_images_bin")
cumulative_fname = image_dir + '_cumulate.bin'
rslt_fname = image_dir + '_cumulate_labels.csv'

target_dtype = 'int16'
fractal_bits = 11

dtype = getattr(torch, target_dtype)

os.makedirs(image_bin_dir, exist_ok=True)

image_count = len(os.listdir(image_dir))
print(f"Total number of images: {image_count}")

# NOTE: image dimensions are 450, 600, 3

# Root Directory to save the training checkpoints
root_dir = os.path.join(dataset_path, "Training Checkpoints with Augmentations")
os.makedirs(root_dir, exist_ok=True)

unique_diagnoses = df['dx'].unique()
num_classes = len(unique_diagnoses)

# Input Dimensions for the model:
# x_dim, y_dim = 256, 256
# x_dim, y_dim = 32, 32
# x_dim, y_dim = 128,128
x_dim, y_dim = 64,64
# x_dim, y_dim = 224, 224
# x_dim, y_dim = 28, 28

# For RGB:
channel_size = 3
# channel_size = 1

seed = 45482
# seed = 4548
# seed = 1
torch.manual_seed(seed=seed)
np.random.seed(seed)

class SkinCancerDatasetBin(torch.utils.data.Dataset):
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
        img_path = os.path.join(self.root, f'{self.image_files[index]}.jpg')
        image = np.asarray(Image.open(img_path).convert("RGB"))
        label = self.labels[index]
        if self.transform:
            image = self.transform(image = image)
        else :
            image = torch.tensor(image)
        if isinstance(image, dict):
            image = image['image']
        # print('Index :', index, 'Size :', image.size())
        return image.to(torch.float32), label, f'{self.image_files[index]}.jpg'
    

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
basic_transform = A.Compose([
    A.Resize(y_dim,x_dim),
    *([A.ToGray(channel_size,p=1),] if  channel_size == 1 else []),
    # A.Normalize(normalization='min_max'),
    # A.Normalize(mean=0.5, std=0.5),
    A.Normalize(),
    A.ToTensorV2(),
])

splits = [0.75, 0.09, 0.16]

# First, create the full dataset without transform (we'll set transforms per split)
# full_dataset = SkinCancerDataset(root=image_dir, csv_path=csv_path, transform=None)
full_dataset = SkinCancerDatasetBin(root=image_dir, csv_path=df, output_classes=num_classes,transform=basic_transform)
# train_dataset, val_dataset = random_split(full_dataset, [0.85, 0.15])
_, _, test_dataset = random_split(full_dataset, splits)

pbar = tqdm(iter(test_dataset))

# Save images to binary
check=False
label_dict = {'Image' : [], 'Label' : []}
for img, label, img_name in pbar:
    
    img2 = (img * 2**fractal_bits).cpu().to(dtype)
    fname = os.path.join(image_bin_dir, img_name)
    if not check:
        assert (np.frombuffer(np.asarray(img2).tobytes(), dtype=target_dtype) == np.frombuffer(np.asarray(torch.nn.Flatten(0)(img2)).tobytes(), dtype=target_dtype)).all()
        check = True
    
    # save_tensor_to_bin(img2, os.path.splitext(fname)[0] + '.bin', verbose = False)
    
    label_dict['Image'].append(os.path.splitext(img_name)[0])
    label_dict['Label'].append(label)
    
rslt_df = pd.DataFrame(label_dict)
    
img_list = os.listdir(image_bin_dir)
rslt_df = rslt_df.set_index('Image')
# cumulate_files(img_list, image_bin_dir, cumulative_fname, 'ISIC')

offset_df = pd.DataFrame({
    'Image' : [os.path.splitext(img_name)[0] for img_name in os.listdir(image_bin_dir)],
    'Offset' : range(len(img_list))
}).set_index('Image')

rslt_df = pd.concat([rslt_df, offset_df], axis=1)
print(rslt_df)
rslt_df = rslt_df.reset_index().set_index('Offset').sort_index()
print(rslt_df)
rslt_df.to_csv(rslt_fname)