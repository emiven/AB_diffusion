from ab_trainer import *
#from ab_classifier_free_guidance import *
from ab_denoising_diffusion_pytorch import *
from user_hints import *
import matplotlib.pyplot as plt
import torch
from multiprocessing import cpu_count
import os
from datasets import load_dataset, load_dataset_builder, get_dataset_split_names,concatenate_datasets,DatasetDict,load_from_disk
from datasets import Dataset as HFDataset
from torchvision.datasets import ImageFolder

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
        beta_schedule = 'cosine',
        auto_normalize=False
).to(device)

hint_generator = RandomHintGenerator(
    input_size = 256,
    hint_size = 2,
    num_hint_range = [0, 10]
    
)

trainer = ABTrainer(
        diffusion_model = diffusion,
        folder_train = train_folder,
        folder_sample= test_folder,
        train_ds = dataset_train,
        sample_ds= dataset_test,
        train_batch_size = 30,
        gradient_accumulate_every = 2,
        augment_horizontal_flip = False,
        train_lr = 4e-5,
        train_num_steps = 130000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        num_samples = 16,
        results_folder = './results',
        amp = True,
        mixed_precision_type = 'fp16',
        split_batches = True,
        calculate_fid = False,
        inception_block_idx = 2048,
        max_grad_norm = 1.,
        num_fid_samples = 50000,
        save_best_and_latest_only = False,
        #ab diffusion stuff
        hint_generator = hint_generator,
        hint_color_avg = True,
        log_freq = 9999,
        save_every = 9999,

        )
trainer.load(8)
trainer.train()