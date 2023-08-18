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
from AB_diffusion.ab_fine_tuner import create_adam_optimizer,reinit_params

device = torch.device(7 if torch.cuda.is_available() else "cpu")
print("Selected GPU:", torch.cuda.get_device_name(device))
#check wicj gpu is selected
torch.cuda.set_device(device)
print("Number of available GPUs:", torch.cuda.device_count())
print("device:", device)
print("Selected GPU:", torch.cuda.current_device())
model_folder = "./models"
model_name = "model-pred_v-1000-sigmoid-256-10x10000steps.pt"

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
        beta_schedule = 'sigmoid'
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


optimizer = create_adam_optimizer(diffusion, params_to_reinit, 1e-6,1e-4)


fine_tuner = ABTrainer(
    optimizer=optimizer,
    diffusion_model=diffusion,
    ds_train=dataset_train,
    ds_val=dataset_val,
    train_batch_size = 32,
    val_batch_size = 25,
    gradient_accumulate_every = 1,
    augment_data = True,
    train_lr = 1e-4,
    train_num_steps = 10000,
    val_num_steps = 100,
    ema_update_every = 10,
    ema_decay = 0.995,
    adam_betas = (0.9, 0.99),
    
    results_folder = './results',
    results_prefix = 'fine_tune',
    amp = False,
    mixed_precision_type = 'fp16',
    split_batches = True,
    max_grad_norm = 1.,
    #ab diffusion stuff
    hint_generator = hint_generator,
    hint_color_avg = True,
    
    #logging saving, samplig details
    sample_image_every= 2000,
    save_every = 3000,
    validation_every = 1000,
    num_image_samples = 25,
    use_tensorboard=True

)

fine_tuner.train()