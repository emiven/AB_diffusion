from AB_diffusion import ABTrainer,ABGaussianDiffusion,RandomHintGenerator,ABUnet,ABVParamContinuousTimeGaussianDiffusion
import matplotlib.pyplot as plt
import torch
from multiprocessing import cpu_count
import os
from datasets import load_dataset
from datasets import Dataset as HFDataset
from torchvision.datasets import ImageFolder
from torchviz import make_dot
from torchsummary import summary
from AB_diffusion.ab_fine_tuner import create_adam_optimizer,reinit_params

device = torch.device(7 if torch.cuda.is_available() else "cpu")
print("Selected GPU:", torch.cuda.get_device_name(device))
#check wicj gpu is selected
torch.cuda.set_device(device)
print("Number of available GPUs:", torch.cuda.device_count())
print("device:", device)
print("Selected GPU:", torch.cuda.current_device())

#load sem images
image_dir_val = "./data/sem images/val"
image_dir_train = "./data/sem images/train"

dataset_train = load_dataset("imagenet-1k",split="train")
dataset_test = load_dataset("imagenet-1k",split="validation")


hint_generator = RandomHintGenerator(
    input_size = 256,
    hint_size = 4,
    
)

unnet = ABUnet(
    dim = 64,
    out_dim = 2,
    channels=5,
    learned_sinusoidal_cond=True  
).to(device)

diffusion = ABVParamContinuousTimeGaussianDiffusion(
    unnet,
        image_size = 256,
        num_timesteps = 2000,
        clip_sample_denoised = True
).to(device)




fine_tuner = ABTrainer(
    diffusion_model=diffusion,
    ds_train = dataset_train,
    ds_val = dataset_test,
    train_batch_size = 32,
    val_batch_size = 25,
    gradient_accumulate_every = 2,
    augment_data = True,
    train_lr = 1e-4,
    train_num_steps = 11,
    val_num_steps = 10,
    ema_update_every = 10,
    ema_decay = 0.995,
    adam_betas = (0.9, 0.99),
    
    results_folder = './results',
    results_prefix = 'v_param_contin',

    split_batches = True,
    max_grad_norm = 1.,
    #ab diffusion stuff
    hint_generator = hint_generator,
    hint_color_avg = True,
    
    #logging saving, samplig details
    sample_image_every = 10,
    save_every = 100,
    validation_every = 10,
    num_image_samples = 25,
    use_tensorboard=False,
    eval_automatic_generation = True

)

fine_tuner.train()