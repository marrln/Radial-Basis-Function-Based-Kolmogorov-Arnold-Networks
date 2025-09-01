
''' Version 3.4 - int16 version (Val. Acc. ~97.49%)

Αλλαγές :
++ Βασίστηκε στο Version 3.2, και είναι φτιαγμένο για quantization :
    - dtype=int16 με fract_bits=8 για τις εκπαιδευμένες παραμέτρους
    - dtype=int32 με frac_bits=16 για τις πράξεις [diff - αφαίρεση], [scaled - πολλαπλασιασμός], για αποφυγή overflow

TODO: 
1. Για να είναι πιο general: 
Σύνδεση του output scale (dtype + frac_bits) με το input scale του επόμενου layer? 
+ Έλεγχος να είναι το grid το ίδιο scale με αυτό του νέου input

'''

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, random_split

from ham_dataset import *
from custom_quant_fasterkan import *
from checkpoint_config import *
from training_and_val import *

# torch.distributed.init_process_group()

# Set the device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
device = "cpu" 
torch.set_num_threads(torch.get_num_threads())

dataset_path = 'Dataset'
csv_path = os.path.join(dataset_path, "HAM10000_metadata.csv")
image_dir = os.path.join(dataset_path, "HAM10000_images")
image_test_dir = image_dir
df = pd.read_csv(csv_path)

unique_diagnoses = df['dx'].unique()
num_classes = len(unique_diagnoses)

nv_df = df[df['dx'] == 'nv'].reset_index()
df = df[~(df['dx'] == 'nv')].reset_index()

num_bits = 16

quant_pth = f'Custom-Quantizer/mixed_{num_bits}_bits/'
# quant_pth = f'Custom-Quantizer/{num_bits}_bits/'

# local_dtype = {
#     'grid'   : (num_bits, False),
#     'scale'  : (num_bits, False),
#     'weight' : (4, False),
#     'sdff'   : (num_bits, False),
#     'actf'   : (11,True),
#     'result' : (num_bits, False),
# }
# local_frac_bits = {
#     'grid'   : num_bits - 5,
#     'scale'  : num_bits - 2,
#     'weight' : 6,
#     'sdff'   : num_bits - 4,
#     'actf'   : 11,
#     'result' : num_bits - 5,
# }
# local_dtype = {       # fixed data widtch
#     'grid'   : (num_bits, False),
#     'scale'  : (num_bits, False),
#     'weight' : (num_bits, False),
#     'sdff'   : (num_bits, False),
#     'actf'   : (num_bits-1,True),
#     'result' : (num_bits, False),
# }
local_dtype = {       # mixed data widtch
    'grid'   : (num_bits, False),
    'scale'  : (num_bits, False),
    'weight' : (num_bits // 2, False),
    'sdff'   : (num_bits, False),
    'actf'   : ((num_bits-1) // 2,True),
    'result' : (num_bits, False),
}
local_frac_bits = default_frac_bits
local_frac_bits = {
    'grid'   : 3,
    'scale'  : 6,
    'weight' : 8,
    'sdff'   : 2,
    'actf'   : 7,
    'result' : 3,
}
hardtanh = False
fit_model = True

# Root Directory to save the training checkpoints
root_dir = rf"{dataset_path}/{quant_pth}"
os.makedirs(root_dir, exist_ok=True)
pretrained_top_dir = root_dir.replace(quant_pth, 'Pretrained')

model_pths= []
for root, dirs, files in os.walk(pretrained_top_dir):
    for file in files:
        if 'epoch_best' in root and os.path.splitext(file)[1] == '.pth':
            model_pths.append(os.path.join(root, file))

x_dim, y_dim, channel_size = 64, 64, 3 
output_classes = 7
batch_size = 32
probability=0.25
pretrained = True

# Input Dimensions for the model:
seed = 45482
torch.manual_seed(seed=seed)
np.random.seed(seed)

# Make dataset
splits = [0.75, 0.09, 0.16]
full_dataset = SkinCancerDataset(root=image_dir, csv_path=df, output_classes=num_classes,transform=None)
train_dataset, val_dataset, test_dataset = random_split(full_dataset, splits)

if pretrained:
    df = pd.concat([df, nv_df]).reset_index()
    full_dataset = SkinCancerDataset(root=image_dir, csv_path=df, output_classes=num_classes,transform=None)
    
    train_dataset.dataset = full_dataset
    val_dataset.dataset = full_dataset
    test_dataset.dataset = full_dataset
    
    nv_ind = df[df['dx'] == 'nv'].index.to_list()
    tr_nv_ind, val_nv_ind, test_nv_ind = random_split(nv_ind, splits)

    train_dataset.indices += tr_nv_ind.indices
    val_dataset.indices += val_nv_ind.indices
    test_dataset.indices += test_nv_ind.indices
    
# Define the transforms for all splits
train_dataset.dataset.transform = get_augmented_transform(x_dim=x_dim,y_dim=y_dim,channel_size=channel_size, probability=probability)
val_dataset.dataset.transform = get_basic_transform(x_dim=x_dim,y_dim=y_dim,channel_size=channel_size)
test_dataset.dataset.transform = get_basic_transform(x_dim=x_dim,y_dim=y_dim,channel_size=channel_size)

train_loader = DataLoader(train_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

calibration_data, _ = next(iter(train_loader))
calibration_data = calibration_data.to(device)

# ---------------------- Load Pre-Trained Floating Point Model ----------------------
for model_pth in model_pths:
    seed, criter, optim, sched, lr, dim_list, grid_size, grid_min, grid_max, inv_denominator = deconstruct_dir(os.path.dirname(os.path.dirname(model_pth)))

    model_tmp, _ = initialize_model(
        root_dir=None,
        dimension=dim_list,
        grid_size=grid_size,
        lr=lr,
        sched=sched,
        optim=optim,
        criterion=criter,
        grid_min=grid_min,
        grid_max=grid_max,
        inv_denominator=inv_denominator,
        x_dim=x_dim,
        y_dim=y_dim,
        channel_size=channel_size,
        seed=seed,
        model_type=FasterKAN
    )
    model_tmp, *_ = load_checkpoint(model_tmp, optimizer_name=None, checkpoint_path=model_pth, device=device)
    model_tmp.eval()
    model = FixedPointFasterKAN(model_tmp, dtype_dict=local_dtype, frac_bits_dict=local_frac_bits, hardtanh=hardtanh).to(device)
    checkpoint_dir = os.path.dirname(model_pth).replace(pretrained_top_dir,root_dir)
    
    if fit_model:
        model.fit_quantize(calibration_data, model_tmp)
        
    save_checkpoint(checkpoint_dir, model, None, quant_pth, None, None)
    
    print(model.state_dict())
    # model.load_state_dict(model.state_dict())
    # print(model.state_dict())
    
# --------------------------------------------------------------------------------------------
for model_pth in model_pths:
    seed, criter, optim, sched, lr, dim_list, grid_size, grid_min, grid_max, inv_denominator = deconstruct_dir(os.path.dirname(os.path.dirname(model_pth)))

    model, _ = initialize_model(
        root_dir=None,
        dimension=dim_list,
        grid_size=grid_size,
        lr=lr,
        sched=sched,
        optim=optim,
        criterion=criter,
        grid_min=grid_min,
        grid_max=grid_max,
        inv_denominator=inv_denominator,
        x_dim=x_dim,
        y_dim=y_dim,
        channel_size=channel_size,
        seed=seed,
        model_type=FixedPointFasterKAN
    )
    model.update_hardtanh(hardtanh)
    validate_model(FloatWrapperModule(model), test_loader, criter, model_pth.replace(pretrained_top_dir,root_dir), device=device, metrics_flag=True)
    for _iter, layer in enumerate(model.layers):
        print(f'Layer {_iter}:', layer.frac_bits_dict)
        print(f'        :', layer.torch_dtype)
    
    