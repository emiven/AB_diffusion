from AB_diffusion import ABTrainer,ABGaussianDiffusion,RandomHintGenerator,ABUnet
import torch
from datasets import load_dataset


#loa dimagenet
dataset_train = load_dataset("zh-plus/tiny-imagenet",split="train")#load_dataset("imagenet-1k",split="train")
dataset_test = load_dataset("zh-plus/tiny-imagenet",split="valid")#load_dataset("imagenet-1k",split="validation")

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
        image_size = 64,
        timesteps = 1000,
        objective = 'pred_noise',
        beta_schedule = 'cosine',
).to(device)

hint_generator = RandomHintGenerator(
    input_size = 64,
    hint_size = 2    
)

trainer = ABTrainer(
        diffusion_model = diffusion,
        ds_train = dataset_train,
        ds_val = dataset_test,

        train_batch_size = 100,
        val_batch_size = 100,
        train_lr = 1e-4,
        train_num_steps = 100000,
        val_num_steps = 100,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        
        results_folder = './results',
        results_prefix = 'test',
        save_images_to_disk = True,

        #ab diffusion stuff
        hint_generator = hint_generator,
        
        #logging saving, samplig details
        sample_image_every= 5000,
        save_every = 20000,
        validation_every = 2000,
        num_image_samples = 49,
        use_tensorboard=False,
        eval_automatic_generation = True

        )
trainer.train()