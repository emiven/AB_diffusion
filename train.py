from AB_diffusion import ABTrainer,ABGaussianDiffusion,RandomHintGenerator,ABUnet
import torch
from datasets import load_dataset

import os
#os.environ['HF_DATASETS_CACHE'] = "/Home/siv32/eve036/.cache/huggingface/datasets"
##loa dimagenet
#dataset_train = load_dataset("imagenet-1k",split="train")
#dataset_test = load_dataset("imagenet-1k",split="validation")
image_dir_val = "./data/SEM images/val"
image_dir_train = "./data/SEM images/train"

dataset_train = load_dataset("imagefolder", data_dir=image_dir_train)["train"]
dataset_val = load_dataset("imagefolder", data_dir=image_dir_val)["train"]


device = torch.device(6 if torch.cuda.is_available() else "cpu")
print("Selected GPU:", torch.cuda.get_device_name(device))
#check wicj gpu is selected
torch.cuda.set_device(device)

print("Number of available GPUs:", torch.cuda.device_count())
print("device:", device)
print("Selected GPU:", torch.cuda.current_device())
model_folder = "./fs_imgnet-pred_v-1000-sigmoid-256"
model_name = "model-pred_v-1000-sigmoid-256-10x10000steps.pt"


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
        min_snr_loss_weight = True, # https://arxiv.org/abs/2303.09556
        min_snr_gamma = 5,
        offset_noise_strength=0.1
).to(device)

hint_generator = RandomHintGenerator()

trainer = ABTrainer(
        diffusion_model = diffusion,
        ds_train = dataset_train,
        ds_val = dataset_val,
        augment_data = True,
        train_batch_size = 32,
        val_batch_size = 32,
        train_lr = 7e-5,
        train_num_steps = 130000,
        val_num_steps = 10,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        gradient_accumulate_every=2,
        results_folder = './results',
        results_prefix = 'no_reinit',
        save_images_to_disk = True,

        #ab diffusion stuff
        hint_generator = hint_generator,
        
        #logging saving, samplig details
        sample_image_every= 1000,
        save_every = 1000,
        validation_every = 1000,
        num_image_samples = 25,
        use_tensorboard=True,
        eval_automatic_generation = True

        )
trainer.load("./results/fs_imgnet-pred_v-1000-sigmoid-256/model-pred_v-1000-sigmoid-256-10x10000steps.pt")
trainer.train()