
from AB_diffusion.ab_trainer import ABDataset
from torch.utils.data import DataLoader
import torch
from AB_diffusion.user_hints import  get_color_hints 
from AB_diffusion.color_handling import LAB2RGB
from tqdm.auto import tqdm
import torch.nn.functional as F
from ignite.metrics import PSNR, SSIM, InceptionScore, FID, MeanSquaredError

import csv
import os
from pathlib import Path




def cycle(dl):
    while True:
        for data in dl:
            yield data


class Evaluator:
    def __init__(self,
                  model,
                  model_name = "model", 
                  dataset = None, 
                  num_steps = 1000,
                  compute_every = 100,
                  batch_size  = 32, 
                  device = "cpu",
                  image_size = 64,
                    augment_data = False,
                    hint_generator = None,

                    folder = None
                  ):
        
        
        self.metrics_storage = {
            "Step": [],
            "MSE": [],
            "PSNR": [],
            "SSIM": [],
            "Inception Score": [],
            "FID": [],
            "Automatic MSE": [],
            "Automatic PSNR": [],
            "Automatic SSIM": [],
            "Automatic Inception Score": [],
            "Automatic FID": []
            
        }


        self.folder = Path(folder + f'/experiments')
        self.folder.mkdir(exist_ok = True)

        self.num_steps = num_steps
        self.device = device
        self.model = model
        self.model_name = model_name
        self.image_size = image_size
        self.batch_size = batch_size 
        self.hint_generator = hint_generator
        self.compute_every = compute_every
        print("loading datasets:")     
        self.dataset = ABDataset(dataset,self.image_size, augment_data = augment_data)
        self.dataloader = cycle(DataLoader(self.dataset, batch_size = self.batch_size, shuffle = True, pin_memory = True))

        # Initialize metrics from Ignite
        self.mse_metric = MeanSquaredError()
        self.psnr_metric = PSNR(data_range=1.0)
        self.ssim_metric = SSIM(data_range=1.0)
        self.inception_score_metric = InceptionScore(device=self.device)
        self.fid_metric = FID(device=self.device)
        
        self.a_mse_metric = MeanSquaredError()
        self.a_psnr_metric = PSNR(data_range=1.0)
        self.a_ssim_metric = SSIM(data_range=1.0)
        self.a_inception_score_metric = InceptionScore(device=self.device)
        self.a_fid_metric = FID(device=self.device)

        
        self.batch_count = 0
        self.step = 1 


    def sample_batch(self, imgL, imgAB):
        self.model.eval()

        # Sampling with hints
        masks = self.hint_generator(imgL.shape[0])
        hints = get_color_hints(imgAB, masks, True, self.device)

        conditioning_batch = torch.cat([imgL, hints], dim=1)
        with torch.no_grad():
            pred_ab_hints = self.model.sample(conditioning_batch)

        # Automatic generation
        conditioning_batch_no_hints = torch.cat([imgL, torch.zeros_like(hints)], dim=1)
        with torch.no_grad():
            pred_ab_automatic = self.model.sample(conditioning_batch_no_hints)

        # Convert LAB to RGB
        images_pred_rgb = torch.from_numpy(LAB2RGB(torch.cat([imgL, pred_ab_hints], dim=1).cpu())).permute(0, 3, 1, 2).to(self.device)
        images_original_rgb = torch.from_numpy(LAB2RGB(torch.cat([imgL, imgAB], dim=1).cpu())).permute(0, 3, 1, 2).to(self.device)
        images_pred_auto_rgb = torch.from_numpy(LAB2RGB(torch.cat([imgL, pred_ab_automatic], dim=1).cpu())).permute(0, 3, 1, 2).to(self.device)
        # Resize for Inception model
        images_pred_rgb = F.interpolate(images_pred_rgb, size=(299, 299), mode='bilinear', align_corners=False)
        images_original_rgb = F.interpolate(images_original_rgb, size=(299, 299), mode='bilinear', align_corners=False)
        images_pred_auto_rgb = F.interpolate(images_pred_auto_rgb, size=(299, 299), mode='bilinear', align_corners=False)

        return {
            "images_pred_rgb": images_pred_rgb,
            "images_original_rgb": images_original_rgb,
            "images_pred_auto_rgb": images_pred_auto_rgb
        }

    def update_metrics(self, images_pred_rgb, images_original_rgb, images_pred_auto_rgb):


        # Metrics for colorized images with hints
        self.mse_metric.update((images_pred_rgb, images_original_rgb))
        self.psnr_metric.update((images_pred_rgb, images_original_rgb))
        self.ssim_metric.update((images_pred_rgb, images_original_rgb))
        self.inception_score_metric.update(images_pred_rgb)
        self.fid_metric.update((images_pred_rgb, images_original_rgb))


        # Metrics for automatically colorized images
        self.a_mse_metric.update((images_pred_auto_rgb, images_original_rgb))
        self.a_psnr_metric.update((images_pred_auto_rgb, images_original_rgb))
        self.a_ssim_metric.update((images_pred_auto_rgb, images_original_rgb))
        self.a_inception_score_metric.update(images_pred_auto_rgb)
        self.a_fid_metric.update((images_pred_auto_rgb, images_original_rgb))
    def compute_metrics(self):
        # Metrics for colorized images with hints
        self.metrics_storage["Step"].append(self.step)
        self.metrics_storage["MSE"].append(self.mse_metric.compute())
        self.metrics_storage["PSNR"].append(self.psnr_metric.compute())
        self.metrics_storage["SSIM"].append(self.ssim_metric.compute())
        inception_score = self.inception_score_metric.compute()
        self.metrics_storage["Inception Score"].append(inception_score)
        self.metrics_storage["FID"].append(self.fid_metric.compute())
        
        # Metrics for automatically colorized images
        self.metrics_storage["Automatic MSE"].append(self.a_mse_metric.compute())
        self.metrics_storage["Automatic PSNR"].append(self.a_psnr_metric.compute())
        self.metrics_storage["Automatic SSIM"].append(self.a_ssim_metric.compute())
        inception_score_auto = self.a_inception_score_metric.compute()
        self.metrics_storage["Automatic Inception Score"].append(inception_score_auto)
        self.metrics_storage["Automatic FID"].append(self.a_fid_metric.compute())

        # Reset metrics
        self.mse_metric.reset()
        self.psnr_metric.reset()
        self.ssim_metric.reset()
        self.inception_score_metric.reset()
        self.fid_metric.reset()
        self.a_mse_metric.reset()
        self.a_psnr_metric.reset()
        self.a_ssim_metric.reset()
        self.a_inception_score_metric.reset()
        self.a_fid_metric.reset()

        
    
    def save_metrics_to_csv(self):
        csv_file = str(self.folder / f"{self.model_name}.csv")

        # Check if file exists, if not, write headers
        if not os.path.exists(csv_file):
            with open(csv_file, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.metrics_storage.keys())
                writer.writeheader()

        # Append the latest metrics to the csv
        with open(csv_file, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.metrics_storage.keys())
            last_metrics = {key: self.metrics_storage[key][-1] for key in self.metrics_storage}
            writer.writerow(last_metrics)
    
    def average_metrics(self):
        return {key: sum(self.metrics_storage[key]) / len(self.metrics_storage[key]) for key in self.metrics_storage}


    def evaluate_model(self):
        
        with tqdm(initial = self.step, total = self.num_steps) as pbar: 
            while self.step <= self.num_steps:
                pbar.set_description(f"Step {self.step}/{self.num_steps}")
                imgL,imgAB = next(self.dataloader)
                sampled_images = self.sample_batch(imgL.to(self.device), imgAB.to(self.device))
                self.update_metrics(sampled_images["images_pred_rgb"], sampled_images["images_original_rgb"], sampled_images["images_pred_auto_rgb"])
                if self.step % self.compute_every == 0:
                    self.compute_metrics()
                    self.save_metrics_to_csv()
                    
                    pbar.set_postfix(self.average_metrics())
                self.step += 1

            



