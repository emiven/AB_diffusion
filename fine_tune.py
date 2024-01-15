from AB_diffusion import ABTrainer,ABGaussianDiffusion,RandomHintGenerator,ABUnet
import torch
import os
from datasets import load_dataset

from AB_diffusion.ab_fine_tuner import create_adam_optimizer,reinit_params

device = torch.device(7 if torch.cuda.is_available() else "cpu")
print("Selected GPU:", torch.cuda.get_device_name(device))
#check wicj gpu is selected
torch.cuda.set_device(device)
print("Number of available GPUs:", torch.cuda.device_count())
print("device:", device)
print("Selected GPU:", torch.cuda.current_device())
model_folder = "./results/fs_imgnet-pred_v-1000-sigmoid-256"
model_name = "model-pred_v-1000-sigmoid-256-10x10000steps.pt"

def load(file_name):
    print(str(model_folder  + f'/{file_name}'))
    data = torch.load(str(model_folder  + f'/{file_name}'), map_location=device)
    return data

#load sem images
image_dir_val = "./data/SEM images/val"
image_dir_train = "./data/SEM images/train"

dataset_train = load_dataset("imagefolder", data_dir=image_dir_train)["train"]
dataset_val = load_dataset("imagefolder", data_dir=image_dir_val)["train"]


hint_generator = RandomHintGenerator()


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



loaded_data = load(model_name)
diffusion.load_state_dict(loaded_data['model'])

params_to_reinit = ['model.final_res_block.mlp.1.weight', 'model.final_res_block.mlp.1.bias', 
                     'model.final_res_block.block1.proj.weight', 'model.final_res_block.block1.proj.bias', 
                     'model.final_res_block.block1.norm.weight', 'model.final_res_block.block1.norm.bias', 
                     'model.final_res_block.block2.proj.weight', 'model.final_res_block.block2.proj.bias', 
                     'model.final_res_block.block2.norm.weight', 'model.final_res_block.block2.norm.bias', 
                     'model.final_res_block.res_conv.weight', 'model.final_res_block.res_conv.bias', 
                     'model.final_conv.weight', 'model.final_conv.bias']

reinit_params(diffusion,params_to_reinit)


optimizer = create_adam_optimizer(diffusion, params_to_reinit, 1e-7,1e-4)


hint_generator = RandomHintGenerator()

fine_tuner = ABTrainer(
        diffusion_model = diffusion,
        ds_train = dataset_train,
        ds_val = dataset_val,
        optimizer=optimizer,
        gradient_accumulate_every = 2,
        train_batch_size = 32,
        val_batch_size = 32,
        train_lr = 1e-4,
        train_num_steps = 15000,
        val_num_steps = 10,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        results_folder = './results',
        results_prefix = 'fine_tune_reinit',
        save_images_to_disk = True,
        augment_data = True,
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
fine_tuner.train()