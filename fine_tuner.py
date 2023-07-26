from AB_diffusion import ABTrainer,ABGaussianDiffusion,RandomHintGenerator,ABUnet
import matplotlib.pyplot as plt
import torch
from multiprocessing import cpu_count
import os
from datasets import load_dataset
from datasets import Dataset as HFDataset
from torchvision.datasets import ImageFolder
from torchviz import make_dot
from torchsummary import summary
from AB_diffusion.ab_fine_tuner import copy_layers,freeze_layers,create_adam_optimizer,reinit_layers

device = torch.device(7 if torch.cuda.is_available() else "cpu")
print("Selected GPU:", torch.cuda.get_device_name(device))
#check wicj gpu is selected
torch.cuda.set_device(device)
print("Number of available GPUs:", torch.cuda.device_count())
print("device:", device)
print("Selected GPU:", torch.cuda.current_device())
model_folder = "./models"
model_name = "v_pred_256_1000_cosine_13k.pt"

def load(file_name):
    print(str(model_folder  + f'/{file_name}'))
    data = torch.load(str(model_folder  + f'/{file_name}'), map_location=device)
    return data

#load sem images
image_dir_val = "./data/sem images/val"
image_dir_train = "./data/sem images/train"

dataset_train = load_dataset("imagefolder", data_dir=image_dir_train)["train"]
dataset_val = load_dataset("imagefolder", data_dir=image_dir_val)["train"]


hint_generator = RandomHintGenerator(
    input_size = 256,
    hint_size = 4,
    
    
)

unnet = ABUnet(
    dim = 64,
    out_dim = 2,
    channels=5,    

).to(device)

diffusion = ABGaussianDiffusion(
    unnet,
        image_size = 256,
        timesteps = 1000,
        objective = 'pred_v',
        beta_schedule = 'cosine',
        auto_normalize=False
).to(device)


loaded_data = load(model_name)
diffusion.load_state_dict(loaded_data['model'])

params_to_reinit = ['model.final_res_block.mlp.1.weight', 'model.final_res_block.mlp.1.bias', 
                      'model.final_res_block.block1.proj.weight', 'model.final_res_block.block1.proj.bias', 
                      'model.final_res_block.block1.norm.weight', 'model.final_res_block.block1.norm.bias', 
                      'model.final_res_block.block2.proj.weight', 'model.final_res_block.block2.proj.bias', 
                      'model.final_res_block.block2.norm.weight', 'model.final_res_block.block2.norm.bias', 
                      'model.final_res_block.res_conv.weight', 'model.final_res_block.res_conv.bias', 
                      'model.final_conv.weight', 'model.final_conv.bias']

reinit_layers(diffusion,params_to_reinit)


optimizer = create_adam_optimizer(diffusion, params_to_reinit, 1e-7,1e-4)


fine_tuner = ABTrainer(
    optimizer=optimizer,
    diffusion_model=diffusion,
    ds_val=dataset_train,
    ds_val=dataset_val,
    train_batch_size=30,
    gradient_accumulate_every=1,
    augment_data=True,
    train_num_steps=50000,
    ema_update_every=10,
    ema_decay=0.995,
    adam_betas=(0.9, 0.99),
    val_batch_size=16,
    results_folder='./results',
    amp=False,
    mixed_precision_type='fp16',
    split_batches=True,
    calculate_fid=False,
    inception_block_idx=2048,
    max_grad_norm=1.,
    num_fid_samples=50000,
    hint_generator=hint_generator,
    sample_image_freq=5000,
    save_every=9999
)

fine_tuner.train()