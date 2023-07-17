#put in all necessary imports for Dataset, DataLoader, and Trainer
import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
from torch import nn, einsum
from torch.cuda.amp import autocast
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam

from torchvision import transforms as T, utils

import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator

from denoising_diffusion_pytorch.fid_evaluation import FIDEvaluation

from denoising_diffusion_pytorch.version import __version__

#ab diffusion related imports
from torch.utils.tensorboard import SummaryWriter

from user_hints import *
from color_handling import *
from IPython.utils import io as iol
from skimage import io
from kornia.color import rgb_to_lab, lab_to_rgb
from datasets import load_from_disk

import time



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

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

#ab diffusion related helpers



class ABDataset(Dataset):
    def __init__(
        self,
        dataset,
        image_size,
        augment_horizontal_flip=False,
    ):
        super().__init__()
        self.dataset = dataset
        self.image_size = image_size

        
        self.transform = T.Compose([
            T.Lambda(self.maybe_convert_image),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
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

    def maybe_convert_image(self, img):
        if  img.mode != "RGB":
            img = img.convert("RGB")
            
        return img

# trainer class

class ABTrainer(object):
    def __init__(
        self,
        diffusion_model,
        folder_train,
        folder_sample,
        train_ds = None,
        sample_ds = None,

        *,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        augment_horizontal_flip = True,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        num_samples = 25,
        results_folder = './results',
        amp = False,
        mixed_precision_type = 'fp16',
        split_batches = True,
        calculate_fid = True,
        inception_block_idx = 2048,
        max_grad_norm = 1.,
        num_fid_samples = 50000,
        save_best_and_latest_only = False,
        #ab diffusion stuff
        hint_generator = None,
        hint_color_avg = True,
        log_freq = 1000,
        save_every = 10000,




    ):
        super().__init__()


        #ab diffusion stuff
        self.hint_generator = hint_generator
        self.hint_color_avg = hint_color_avg
        self.log_freq = log_freq
        self.save_every = save_every

       

        
        
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

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        if save_best_and_latest_only:
            assert calculate_fid, "`calculate_fid` must be True to provide a means for model evaluation for `save_best_and_latest_only`."
            self.best_fid = 1e10 # infinite
        self.save_best_and_latest_only = save_best_and_latest_only

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        self.max_grad_norm = max_grad_norm

        # dataset and dataloader
        print(self.batch_size)
        print(train_batch_size)

        print("loading datasets:")
        #time to load datasets
        current_time = time.time()
        if train_ds is None:
            self.ds_train = ABDataset(load_from_disk(dataset_path=folder_train), self.image_size, augment_horizontal_flip = augment_horizontal_flip) 
        else:
            self.ds_train = ABDataset(train_ds,self.image_size, augment_horizontal_flip = augment_horizontal_flip)
        print("time to load train dataset: ", time.time() - current_time)
        print("train dataset length: ", len(self.ds_train))
        current_time = time.time()
        if sample_ds is None:
            self.ds_sample = ABDataset(load_from_disk(dataset_path=folder_sample), self.image_size, augment_horizontal_flip = augment_horizontal_flip)
        else:
            self.ds_sample = ABDataset(sample_ds, self.image_size, augment_horizontal_flip = augment_horizontal_flip)
        print("time to load sample dataset: ", time.time() - current_time)
        print("sample dataset length: ", len(self.ds_sample))
        
        dl_train = DataLoader(self.ds_train, batch_size = self.batch_size, shuffle = True, pin_memory = True,num_workers=cpu_count()//2)
        dl_train = self.accelerator.prepare(dl_train)
        self.dl_train = cycle(dl_train)

        dl_sample = DataLoader(self.ds_sample, batch_size = self.batch_size, shuffle = True, pin_memory = True,num_workers=cpu_count()//2)
        dl_sample = self.accelerator.prepare(dl_sample)
        self.dl_sample = cycle(dl_sample)

        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        # FID-score computation

        self.calculate_fid = calculate_fid and self.accelerator.is_main_process

        if self.calculate_fid:
            if not self.model.is_ddim_sampling:
                self.accelerator.print(
                    "WARNING: Robust FID computation requires a lot of generated samples and can therefore be very time consuming."\
                    "Consider using DDIM sampling to save time."
                )
            self.fid_scorer = FIDEvaluation(
                batch_size=self.batch_size,
                dl=self.dl_sample,
                sampler=self.ema.ema_model,
                channels=self.channels,
                accelerator=self.accelerator,
                stats_dir=results_folder,
                device=self.device,
                num_fid_samples=num_fid_samples,
                inception_block_idx=inception_block_idx
            )

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
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        print("model loaded")
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):

         #logging
        #add leanring rate, objective and image size to tensorboard summary comment
        #self.writer = SummaryWriter(comment=f"lr{self.train_lr}-{self.model.objective}-{self.model.image_size}")
        #make a hyperparameter dictionary and add it to tensorboard
        #hyperparams = {
        #    "train_batch_size": self.batch_size,
        #    "gradient_accumulate_every": self.gradient_accumulate_every,
        #    "train_lr": self.train_lr,
        #    "train_num_steps": self.train_num_steps,
        #    "ema_decay": self.ema_decay,
        #    "calculate_fid": self.calculate_fid,
        #    "timesteps":self.model.num_timesteps,
        #    "beta_schedule":self.model.beta_schedule,
        #    "objective":self.model.objective,
        #    "offset_noise_strength": self.model.offset_noise_strength,
        #    "image_size": self.model.image_size,
        #}
        #self.writer.add_hparams(hyperparams, {})
        ##add model graph to tensorboard
        #self.writer.add_graph(self.model.model, 
        #                      (torch.zeros((self.batch_size, 2, self.image_size, self.image_size)).to(self.device),
        #                       torch.randint(0, self.model.num_timesteps, (self.batch_size,), device=self.device).long(),
        #                       torch.zeros((self.batch_size, 3, self.image_size, self.image_size)).to(self.device)
        #                       )
        #                       )

        accelerator = self.accelerator
        device = accelerator.device

        avg_loss_over_n_steps = 0
        avg_loss = 0
        log_step = 0 # initialize log step counter
        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:
            
            while self.step < self.train_num_steps:
                
                
                #print("new_step")
                step_loss = 0.
                
                for _ in range(self.gradient_accumulate_every):

                    imgL,imgAB = next(self.dl_train)
                    conditioning = imgL.to(device)
                    imgAB = imgAB.to(device)
                    #print("imgAB")
                    #plotMinMax(imgAB)
                    #print("imgL")
                    #plotMinMax(imgL)

                    with self.accelerator.autocast():
                        if self.hint_generator is not None:
                            hint_masks = self.hint_generator(imgAB.shape[0])

                            hints = get_color_hints(imgAB, hint_masks, avg_color = self.hint_color_avg, device = device).to(device)
                            conditioning = torch.cat([conditioning, hints], dim = 1)

                            
                        loss = self.model(imgAB,conditioning = conditioning)
                        loss = loss / self.gradient_accumulate_every
                        step_loss += loss.item()
                        avg_loss_over_n_steps+=loss.item()
                        avg_loss += loss.item()

                    self.accelerator.backward(loss)
                
                #logging
                self.step += 1
                #self.writer.add_scalar('loss/train', step_loss, self.step)
                #self.writer.add_scalar('avg_loss/train', avg_loss/self.step, self.step)

                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                pbar.set_description(f"Step {self.step}/{self.train_num_steps}, loss={step_loss:.4f}, avg_loss={avg_loss/self.step:.4f}")
                accelerator.wait_for_everyone()
                self.opt.step()
                self.opt.zero_grad()
                accelerator.wait_for_everyone()
                
                if accelerator.is_main_process:
                    self.ema.update()
                    milestone = log_step // self.log_freq
                    
                    if log_step != 0 and log_step % self.save_every == 0:                        
                        self.save(milestone = self.step // self.save_every)
                    
                    if log_step != 0 and log_step % self.log_freq == 0:
                        images_pred_list = []
                        images_original_list = []
                        images_hint_list = []
                        fid_score = 0

                        self.ema.ema_model.eval()


                        with torch.inference_mode():
                           

                            batches = num_to_groups(self.num_samples, self.batch_size)
                            for i, b in enumerate(batches):
                                imgL,imgAB = next(self.dl_sample)
                                imgL_batch = imgL[:b].to(device)
                                imgAB = imgAB.to(device)
                                if self.hint_generator is not None:
                                    hints_sample = self.hint_generator(imgL.shape[0])
                                    hints_AB_sample = get_color_hints(imgAB,hints_sample,self.hint_color_avg,device)[:b].to(device)
                                    images_hint = torch.cat([torch.ones_like(imgL_batch), hints_AB_sample[:b]], dim=1)
                                    imgL_batch = torch.cat([imgL_batch, hints_AB_sample[:b]], dim=1)
                                images_pred_list.append(torch.cat([imgL[:b].to(device), self.ema.ema_model.sample(imgL_batch)], dim=1))
                                images_hint_list.append(images_hint)
                                
                                images_original_list.append(torch.cat([imgL[:b], imgAB[:b]], dim=1))

                        if self.hint_generator is not None:
                            images_hint_list_rgb = [torch.from_numpy(LAB2RGB(im.cpu())).permute(0,3,1,2).to(self.device) for im in images_hint_list]
                            images_hint_rgb= torch.cat(images_hint_list_rgb, dim = 0)
                            images_hint_grid = utils.make_grid(images_hint_rgb, nrow=int(math.sqrt(self.num_samples))).permute(1,2,0)


                        
                        images_pred_list_rgb = [torch.from_numpy(LAB2RGB(im.cpu())).permute(0,3,1,2).to(self.device) for im in images_pred_list]
                        images_pred_rgb= torch.cat(images_pred_list_rgb, dim = 0)

                        images_original_list_rgb = [torch.from_numpy(LAB2RGB(im.cpu())).permute(0,3,1,2).to(self.device) for im in images_original_list]
                        images_original_rgb= torch.cat(images_original_list_rgb, dim = 0)
                        
                        images_original_grid = utils.make_grid(images_original_rgb, nrow=int(math.sqrt(self.num_samples))).permute(1,2,0)
                        images_pred_grid = utils.make_grid(images_pred_rgb, nrow=int(math.sqrt(self.num_samples))).permute(1,2,0)
                        



                        #convert to numpy arrays    
                        images_pred_grid = images_pred_grid.cpu().detach().numpy()
                        images_original_grid = images_original_grid.cpu().detach().numpy()
                        images_hint_grid = images_hint_grid.cpu().detach().numpy()

                        #display the grid images
                        #log the grid images to tensorboard
                        #self.writer.add_image('images_pred_grid', images_pred_grid, self.step)
                        #self.writer.add_image('images_original_grid', images_original_grid, self.step)
                        #self.writer.add_image('images_hint_grid', images_hint_grid, self.step)

                        #save the images to results folder
                        io.imsave(self.results_folder / f'images_pred_grid-{milestone}.png', images_pred_grid)
                        io.imsave(self.results_folder / f'images_original_grid-{milestone}.png', images_original_grid)
                        io.imsave(self.results_folder / f'images_hint_grid-{milestone}.png', images_hint_grid)
                        
                        #plt.figure(figsize=(10,10))
                        #plt.imshow(images_pred_grid)
                        #plt.show()
                        #plt.figure(figsize=(10,10))
                        #plt.imshow(images_original_grid)
                        #plt.show()
                        #plt.figure(figsize=(10,10))
                        #plt.imshow(images_hint_grid)
                        #plt.show()
                        #save the grid images




                        

                        # whether to calculate fid

                        #if self.calculate_fid:
                        #    fid_score = self.fid_scorer.fid_score()
                        #    accelerator.print(f'fid_score: {fid_score}')
                        #if self.save_best_and_latest_only:
                        #    if self.best_fid > fid_score:
                        #        self.best_fid = fid_score
                        #        self.save("best")
                        #    self.save("latest")
                        #else:
                        #    self.save(milestone)
                        #avg_loss_over_n_steps = 0

                pbar.update(1)
                log_step += 1

        accelerator.print('training complete')