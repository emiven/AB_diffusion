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

#loa dimagenet
dataset_train = load_dataset("imagenet-1k",split="train")
dataset_test = load_dataset("imagenet-1k",split="validation")

train_folder = "data/tinyimgnet/train"
test_folder = "data/tinyimgnet/val"
device = torch.device(7 if torch.cuda.is_available() else "cpu")
print("Selected GPU:", torch.cuda.get_device_name(device))
#check wicj gpu is selected
torch.cuda.set_device(device)

print("Number of available GPUs:", torch.cuda.device_count())
print("device:", device)
print("Selected GPU:", torch.cuda.current_device())

model = ABUnet(
    dim = 64,
    out_dim = 2,
    channels=5,
    full_attn = (False, False, False, True)    

).to(device)
diffusion = ABGaussianDiffusion(
    model,
        image_size = 256,
        timesteps = 1000,
        objective = 'pred_v',
        beta_schedule = 'sigmoid',
        auto_normalize=False
).to(device)

hint_generator = RandomHintGenerator(
    input_size = 256,
    hint_size = 4,
    num_hint_range = [0, 10]
    
)

trainer = ABTrainer(
        diffusion_model = diffusion,
        ds_train = dataset_train,
        ds_val = dataset_test,

        train_batch_size = 32,
        val_batch_size = 25,
        gradient_accumulate_every = 2,
        augment_data = False,
        train_lr = 7e-5,
        train_num_steps = 100000,
        val_num_steps = 300,

        ema_update_every = 2,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        
        amp = False,
        mixed_precision_type = 'fp16',
        #ab diffusion stuff
        hint_generator = hint_generator,
        
        #logging saving, samplig details
        sample_image_every= 1000,
        save_every = 10000,
        validation_every = 1000,
        num_image_samples = 25,
        use_tensorboard=True,

        )
trainer.train()