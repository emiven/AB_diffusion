from datetime import datetime
from multiprocessing import cpu_count
from pathlib import Path
from random import random

from accelerate import Accelerator
from ema_pytorch import EMA
from kornia.color import rgb_to_lab
from skimage import io
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T, utils
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from torch import nn
import math

from AB_diffusion.color_handling import LAB2RGB
from AB_diffusion.user_hints import get_color_hints

# This file is based on the code from the Denoising Diffusion Probabilistic Model Pytorch implementation by Phil Wang
# Specifically the trainer class and relevant helpers
# And aproptiated for the diffusion colorizer model by me.
# Original Source: https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
# 
# Modifications includes:
# - Added support for tensorboard logging, modifications to the saving and loading functions
# - Modified train() to support the AB colorizer model
# - Refactored train(), image sampling is done in a separate function, added validation function
# - New dataset class for the AB colorizer
# - Removed some of the code and functionality that was not needed for my project such as the FID calculation


# Helper functions
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cast_tuple(t, length = 1):
    if isinstance(t, tuple):
        return t
    return ((t,) * length)

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data
def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


class ABDataset(Dataset):
    """
    Dataset class for the AB colorizer model
    
    Convert images to LAB color space, normalize to [-1,1] as per https://arxiv.org/abs/2006.11239
    
    Support for data augmentation, would not recommend using it for natural image datasets due to color jittering
    """
    def __init__(
        self,
        dataset,
        image_size,
        augment_data=False,
    ):
        super().__init__()
        self.dataset = dataset
        self.image_size = image_size

        
        self.transform = T.Compose([
            T.Lambda(self.set_rgb_format),
            T.Resize(image_size),
            T.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.0, hue=0.4) if augment_data else nn.Identity(),
            T.RandomHorizontalFlip() if augment_data else nn.Identity(),
            T.RandomVerticalFlip() if augment_data else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Lambda(rgb_to_lab)
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img = self.dataset[index]["image"]
        imgLAB = self.transform(img)
        imgL, imgAB = imgLAB[:1, :, :], imgLAB[1:, :, :]
        imgL = imgL / 50.0 - 1.0
        imgAB = (imgAB / 128.0)
        return imgL, imgAB

    def set_rgb_format(self, img):
        if  img.mode != "RGB":
            img = img.convert("RGB")
            
        return img

# trainer class

class ABTrainer(object):
    """
    Trainer class for the AB colorizer model

    Original source: https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py

    Modified to support the AB colorizer model

    """

    def __init__(
        self,
        diffusion_model,
    
        optimizer = None,
        ds_train = None,
        ds_val = None,
        *,
        train_batch_size = 16,
        val_batch_size = 16,
        gradient_accumulate_every = 1,
        augment_data = True,
        train_lr = 1e-4,
        train_num_steps = 100000,
        val_num_steps = 1000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        
        results_folder = './results',
        results_prefix = '',
        amp = False,
        mixed_precision_type = 'fp16',
        split_batches = True,
        max_grad_norm = 1.,
        #ab diffusion stuff
        hint_generator = None,
        hint_color_avg = True,
        
        #logging saving, samplig details
        sample_image_every= 10000,
        save_every = 10000,
        validation_every = 10000,
        num_image_samples = 25,
        use_tensorboard=False,




    ):
        super().__init__()


        #timings
        self.sample_image_every = sample_image_every
        self.save_every = save_every
        self.validation_every = validation_every
        num_image_samples = num_image_samples
        

        #ab diffusion stuff
        self.hint_generator = hint_generator
        self.hint_color_avg = hint_color_avg
        
        
        
 
        
        self.train_lr = train_lr
        self.ema_decay = ema_decay
        # accelerator

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no'
        )

        # model

        self.model = diffusion_model
        self.channels = diffusion_model.channels

        # sampling and training hyperparameters

        assert has_int_squareroot(num_image_samples), 'number of samples must have an integer square root'
        self.num_samples = num_image_samples

        
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.val_num_steps = val_num_steps

        self.image_size = diffusion_model.image_size

        self.max_grad_norm = max_grad_norm

        # dataset and dataloader
        print("loading datasets:")     
        self.ds_train = ABDataset(ds_train,self.image_size, augment_data = augment_data)
        self.ds_val = ABDataset(ds_val, self.image_size, augment_data = augment_data)

        
        dl_train = DataLoader(self.ds_train, batch_size = self.train_batch_size, shuffle = True, pin_memory = True,num_workers=cpu_count()//2)
        dl_train = self.accelerator.prepare(dl_train)
        

        dl_val = DataLoader(self.ds_val, batch_size = self.val_batch_size, shuffle = True, pin_memory = True,num_workers=cpu_count()//2)
        dl_val = self.accelerator.prepare(dl_val)
        
        self.dl_train = cycle(dl_train)
        self.dl_val = cycle(dl_val)

        # optimizer
        if optimizer is not None:
            self.opt = optimizer
        else:
            self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)


        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder + f'/{results_prefix}-{self.model.objective}-{self.model.sampling_timesteps}-{self.model.beta_schedule}-{self.model.image_size}')
        self.results_folder.mkdir(exist_ok = True)



        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        # step counter state

        self.step = 1

        #logging
        if use_tensorboard:
            current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            self.writer = SummaryWriter(log_dir=results_folder + f'/tensorboard_logs/{results_prefix}-{self.model.objective}-{self.model.sampling_timesteps}-{self.model.beta_schedule}-{self.model.image_size}-{current_time}')
            hyperparameters = {
                'image_size': self.model.image_size,
                'timesteps': self.model.num_timesteps,
                'sampling_timesteps': self.model.sampling_timesteps,
                'objective': self.model.objective,
                'beta_schedule': self.model.beta_schedule,
                'offset_noise_strength': self.model.offset_noise_strength,
                "learning_rate": self.train_lr,
                "batch_size": self.train_batch_size,
                "gradient_accumulate_every": self.gradient_accumulate_every,
                "ema_decay": self.ema_decay,
                "mixed_precision_type": mixed_precision_type,

            }
            self.writer.add_hparams(hyperparameters, {})
        else:
            self.writer = None

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
                            }

        filename = f"model-{self.model.objective}-{self.model.sampling_timesteps}-{self.model.beta_schedule}-{self.model.image_size}-{milestone}x{self.save_every}steps.pt"
        torch.save(data, str(self.results_folder / filename))

    def load(self, file_name):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'{file_name}'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        print("model loaded")
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])


        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):

        accelerator = self.accelerator
        device = accelerator.device

        total_loss = 0
        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:
            
            while self.step <= self.train_num_steps:
                
                step_loss = 0.
                
                for _ in range(self.gradient_accumulate_every):

                    imgL,imgAB = next(self.dl_train)
                    imgAB = imgAB.to(device)

                    with self.accelerator.autocast():

                        hint_masks = self.hint_generator(imgAB.shape[0])
                        hints = get_color_hints(imgAB, hint_masks, avg_color = self.hint_color_avg, device = device).to(device)
                        conditioning = torch.cat([imgL.to(device), hints], dim = 1)

                            
                        loss = self.model(imgAB,conditioning = conditioning)
                        loss = loss / self.gradient_accumulate_every
                        step_loss += loss.item()
                    
                    
                    self.accelerator.backward(loss)
                    
                    
                total_loss += step_loss
                average_loss = total_loss / self.step   
                

                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                pbar.set_description(f"Training: Step {self.step}/{self.train_num_steps}, loss={step_loss:.4f}, avg_loss={average_loss:.4f}")
                
                accelerator.wait_for_everyone()
                self.opt.step()
                self.opt.zero_grad()
                accelerator.wait_for_everyone()
                

                if accelerator.is_main_process:
                    self.ema.update()
                    if self.step % self.save_every == 0:                        
                        self.save(milestone = self.step // self.save_every)
                    if self.step % self.sample_image_every == 0:
                        self.sample_images()
                    if self.step % self.validation_every == 0:
                        self.validate()
                
                if self.writer is not None:
                    self.writer.add_scalar("Training Loss", step_loss, self.step)
                    self.writer.add_scalar("Training Loss avg", average_loss, self.step)            
                
                pbar.update(1)
                self.step += 1
        accelerator.print('training complete')
        
        if self.writer is not None:
            self.writer.close()

        
    
    def validate(self):
        
        accelerator = self.accelerator
        device = accelerator.device
        accelerator.print('validation starting...')
        total_loss = 0
        step = 1
        with tqdm(initial = step, total = self.val_num_steps, disable = not accelerator.is_main_process) as pbar:
            with torch.inference_mode():            
                self.ema.ema_model.eval()

                while step <= self.val_num_steps:
                    step_loss = 0.


                    imgL,imgAB = next(self.dl_val)
                    conditioning = imgL.to(device)
                    imgAB = imgAB.to(device)
                    with self.accelerator.autocast():
                        hint_masks = self.hint_generator(imgAB.shape[0])
                        hints = get_color_hints(imgAB, hint_masks, avg_color = self.hint_color_avg, device = device).to(device)
                        conditioning = torch.cat([conditioning, hints], dim = 1)
                        loss = self.ema.ema_model(imgAB,conditioning = conditioning)
                        
                        step_loss += loss.item()

                    total_loss += step_loss
                    average_loss = total_loss / step 

                    pbar.set_description(f"Validation: Step {step}/{self.val_num_steps}, val_loss={step_loss:.4f}, avg_val_loss={average_loss:.4f}")

                    accelerator.wait_for_everyone()
                    accelerator.wait_for_everyone()
                    pbar.update(1)
                    step += 1

        accelerator.print('validation complete')
        if self.writer is not None:
            self.writer.add_scalar("Validation Loss", total_loss/step, self.step)

    
    def sample_images(self):

        accelerator = self.accelerator
        device = accelerator.device

        images_pred_list = []
        images_original_list = []
        images_hint_list = []

        colorization_loss = 0
        with torch.inference_mode():
                           
            self.ema.ema_model.eval()
            batches = num_to_groups(self.num_samples, self.val_batch_size)

            for i, b in enumerate(batches):
                imgL,imgAB = next(self.dl_val)
                masks= self.hint_generator(imgL.shape[0])
                hints = get_color_hints(imgAB,masks,self.hint_color_avg,device)
                
                hints_batch = hints[:b].to(device)
                imgL_batch = imgL[:b].to(device)
                imgAB_batch = imgAB[:b].to(device)
                conditioning_batch = torch.cat([imgL_batch, hints_batch], dim=1)

                pred_ab = self.ema.ema_model.sample(conditioning_batch)
                colorization_loss += F.mse_loss(pred_ab, imgAB_batch).item()

                images_pred_list.append(torch.cat([imgL_batch.to(device), pred_ab], dim=1))                 
                images_hint_list.append(torch.cat([torch.ones_like(imgL_batch), hints_batch], dim=1))
                images_original_list.append(torch.cat([imgL_batch, imgAB_batch], dim= 1))


        images_hint_rgb= torch.cat([torch.from_numpy(LAB2RGB(im.cpu())).permute(0,3,1,2).to(self.device) for im in images_hint_list], dim = 0)
        images_pred_rgb= torch.cat([torch.from_numpy(LAB2RGB(im.cpu())).permute(0,3,1,2).to(self.device) for im in images_pred_list],dim = 0)
        images_original_rgb= torch.cat([torch.from_numpy(LAB2RGB(im.cpu())).permute(0,3,1,2).to(self.device) for im in images_original_list],dim = 0)


        hint_grid = utils.make_grid(images_hint_rgb, nrow=int(math.sqrt(self.val_batch_size)))
        pred_grid = utils.make_grid(images_pred_rgb, nrow=int(math.sqrt(self.val_batch_size)))
        original_grid = utils.make_grid(images_original_rgb, nrow=int(math.sqrt(self.val_batch_size)))
        
        #convert to numpy arrays    
        
        if self.writer is not None:
            self.writer.add_scalar("Colorization Loss", colorization_loss, self.step)
            self.writer.add_image('Colorization', pred_grid.cpu().detach(), self.step)
            self.writer.add_image('Ground Truth', original_grid.cpu().detach(), self.step)
            self.writer.add_image('User Hints', hint_grid.cpu().detach(), self.step)
        
        

        #save the images to results folder
        io.imsave(self.results_folder / f'images_pred_grid-{self.step // self.sample_image_every}x{self.sample_image_every}.png', (pred_grid.permute(1,2,0).cpu().detach().numpy()*255).astype('uint8'))
        io.imsave(self.results_folder / f'images_original_grid-{self.step // self.sample_image_every}x{self.sample_image_every}.png', (original_grid.permute(1,2,0).cpu().detach().numpy()*255).astype('uint8'))
        io.imsave(self.results_folder / f'images_hint_grid-{self.step // self.sample_image_every}x{self.sample_image_every}.png', (hint_grid.permute(1,2,0).cpu().detach().numpy()*255).astype('uint8'))


