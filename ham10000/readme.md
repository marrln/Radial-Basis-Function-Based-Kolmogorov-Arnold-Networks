

# HAM10000 Dataset & KANs

## The HAM10000 Dataset

This research utilizes the HAM10000 dataset ([Tschandl et al., 2018](https://www.isic-archive.com/#!/topWithHeader/onlyHeaderTop/gallery)), an open-source collection focused on melanin-pigmented lesions. The dataset contains 10,015 dermoscopic images, depicting seven categories of skin lesions with the highest frequency in clinical practice. 

The images were collected over more than 20 years from two main sources:
- The Medical University of Vienna, Austria, specializing in high-risk melanoma patients (multiple nevi, sun-protected sites)
- A primary care skin cancer practice center in Queensland, Australia, treating patients with chronic sun damage (solar lentigines, sun-related lesions)

Each image is labeled based on histopathology, follow-up examination (>1.5 years), expert consensus, or in-vivo confocal microscopy. Some images show the same lesion from different perspectives, camera configurations, or equipment, replicating various real-world conditions.

**Class Imbalance:**
The dataset is highly imbalanced. The most populous class, Melanocytic Nevi (MNV), accounts for about 67% of all samples. See the original publication or dataset documentation for class distribution details.

**Preprocessing:**
In this study, images are normalized using ImageNet statistics and resized from their original $600 \times 450$ pixels to $64 \times 64$ pixels before being flattened to vectors for model input.

For more details, see the official [HAM10000 dataset page on the ISIC Archive](https://www.isic-archive.com/#!/topWithHeader/onlyHeaderTop/gallery).

## Using KANs in this Folder
In this folder, we apply Kolmogorov-Arnold Networks (KANs) to the HAM10000 dataset for skin lesion classification. KANs are a novel neural network architecture designed to efficiently approximate complex functions, making them suitable for challenging image classification problems in dermatology.


You will find scripts and notebooks here for:
- Loading and preprocessing the HAM10000 dataset
- Training and evaluating KANs on the dataset
- Analyzing results and visualizing model performance
- Quantizing the KANs for deployment on resource-constrained devices

### Adjusting Model Hyperparameters
You can easily adjust the model and training hyperparameters by editing the `hyperparams.json` file in this folder. This allows you to experiment with different architectures, learning rates, optimizers, and other settings without modifying the code.
