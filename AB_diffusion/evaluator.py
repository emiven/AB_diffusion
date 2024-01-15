import torch
import torch.nn.functional as F
from skimage import io as skio

import torchmetrics
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from AB_diffusion.color_handling import normalize_lab, de_normalize_lab, LAB2RGB,plotMinMax
from AB_diffusion.ab_trainer import ABDataset
from torchvision import transforms as T, utils
import os
from pathlib import Path
import csv
import math
from torchmetrics.functional.image.lpips import learned_perceptual_image_patch_similarity
from torchmetrics.functional.image import structural_similarity_index_measure
from torchmetrics.functional.image import peak_signal_noise_ratio




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
class Evaluator:
    def __init__(self, model, model_name="model", dataset=None, num_steps=1000, display_every=100,compute_every = 10, batch_size=32, 
                 device="cpu", image_size=64, augment_data=False, hint_generator=None, display_images=False, folder=None,suffix = ""):
        self.metrics_storage = {}

        self.compute_every = compute_every
        self.hint_levels = hint_generator.uniform_hint_range if hint_generator else [0]
        for hint in self.hint_levels:
            self.metrics_storage[f"PSNR@{hint}"] = []
            self.metrics_storage[f"SSIM@{hint}"] = []
            self.metrics_storage[f"LPIPS@{hint}"] = []
            self.metrics_storage[f"MSE@{hint}"] = []


        self.model_name = model_name
        self.display_images = display_images
        model_folder_name = self.model_name.replace('.pt', '')
        self.folder = Path(folder + f'/experiments/{suffix + model_folder_name}')
        self.folder.mkdir(parents=True, exist_ok=True)
        (self.folder / 'images').mkdir(exist_ok=True)

        self.num_steps = num_steps
        self.device = device
        self.model = model
        self.image_size = image_size
        self.batch_size = batch_size 
        self.hint_generator = hint_generator
        self.display_every = display_every

        self.dataset = ABDataset(dataset, self.image_size, augment_data=augment_data)
        self.dataloader = cycle(DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True))
        

        
        self.batch_count = 0
        self.step = 1 


    def sample_batch(self, imgL, imgAB,hints):
        self.model.ema_model.eval()
        conditioning_batch = torch.cat([imgL, hints], dim=1)
        
        with torch.no_grad():
            pred_ab_hints = self.model.ema_model.sample(conditioning_batch)

        images_pred_rgb = torch.from_numpy(LAB2RGB(torch.cat([imgL, pred_ab_hints], dim=1).cpu())).permute(0, 3, 1, 2).to(self.device)
        images_original_rgb = torch.from_numpy(LAB2RGB(torch.cat([imgL, imgAB], dim=1).cpu())).permute(0, 3, 1, 2).to(self.device)
        images_hint_rgb = torch.from_numpy(LAB2RGB(torch.cat([imgL, hints], dim=1).cpu())).permute(0, 3, 1, 2).to(self.device)
        

        if self.display_images and self.step % self.display_every == 0:
            hint_grid = utils.make_grid(images_hint_rgb, nrow=int(math.sqrt(self.batch_size)), padding=0)
            pred_grid = utils.make_grid(images_pred_rgb, nrow=int(math.sqrt(self.batch_size)), padding=0)
            original_grid = utils.make_grid(images_original_rgb, nrow=int(math.sqrt(self.batch_size)), padding=0)
            
            skio.imsave(self.folder / f'images/image_step{self.step}_pred.png', (pred_grid.permute(1,2,0).cpu().detach().numpy()*255).astype('uint8'))
            skio.imsave(self.folder / f'images/image_step{self.step}_original.png', (original_grid.permute(1,2,0).cpu().detach().numpy()*255).astype('uint8'))
            skio.imsave(self.folder / f'images/image_step{self.step}_hint.png', (hint_grid.permute(1,2,0).cpu().detach().numpy()*255).astype('uint8'))

        return {
            "images_pred_rgb": images_pred_rgb,
            "images_original_rgb": images_original_rgb,
        }

    def update_metrics(self, images_pred_rgb, images_original_rgb, num_hints_list=None):
        B, C, H, W = images_pred_rgb.shape


        for idx in range(B):  # B is the batch size

            # throw error if num_hints_list[idx] is not in self.hint_generator.uniform_hint_range
            if num_hints_list[idx] not in self.hint_levels:
                raise ValueError(f"num_hints_list[idx] is not in self.hint_generator.uniform_hint_range: {num_hints_list[idx]}")

            image_pred = images_pred_rgb[idx].unsqueeze(0)
            image_orig = images_original_rgb[idx].unsqueeze(0)
            num_hints = num_hints_list[idx]

            # Compute and store MSE, PSNR, SSIM, LPIPS for the specific hint level
                    # Metrics
            
            mse_value = F.mse_loss(image_pred, image_orig).item()
            psnr_value = peak_signal_noise_ratio(image_pred, image_orig,data_range =  (0,1))
            ssim_value = structural_similarity_index_measure(image_pred, image_orig,data_range =  (0,1))
            lpips_value = learned_perceptual_image_patch_similarity(image_pred.to("cpu"), image_orig.to("cpu"),net_type='alex',normalize = "True").to(self.device)
        # ... (rest of your code for calculating metrics for individual image)

            num_hints = num_hints_list[idx]
            self.metrics_storage[f"MSE@{num_hints}"].append( mse_value)
            self.metrics_storage[f"PSNR@{num_hints}"].append( psnr_value.item())
            self.metrics_storage[f"SSIM@{num_hints}"].append( ssim_value.item())
            self.metrics_storage[f"LPIPS@{num_hints}"].append( lpips_value.item())

    def save_mean_to_csv(self):
        # Create the CSV file if it doesn't exist
        csv_file = self.folder / "metrics.csv"
        if not csv_file.exists():
            with open(csv_file, "w", newline='') as f:
                writer = csv.writer(f)
                # Write the header row
                header = ["Step"]
                for hint in self.hint_levels:
                    header.extend([f"PSNR@{hint}", f"SSIM@{hint}", f"LPIPS@{hint}", f"MSE@{hint}"])
                writer.writerow(header)

        # Compute the averages and write to CSV
        with open(csv_file, "a", newline='') as f:
            writer = csv.writer(f)
            row = [self.step]
            for hint in self.hint_levels:
                for metric in ["PSNR", "SSIM", "LPIPS", "MSE"]:
                    key = f"{metric}@{hint}"
                    values = self.metrics_storage[key]
                    avg = sum(values) / len(values) if values else float('nan')
                    row.append(avg)
                    # Reset the metrics for this hint level
                    self.metrics_storage[key] = []
            writer.writerow(row)

    def evaluate_model(self):
        with tqdm(initial=self.step, total=self.num_steps) as pbar: 
            while self.step <= self.num_steps:
            # Initialize metrics for the step with NaN
            
                imgL, imgAB = next(self.dataloader)
                imgL = imgL.to(self.device)
                imgAB = imgAB.to(self.device)
                if self.hint_generator:
                    hints, num_hints_list = self.hint_generator.generate_hints(imgAB)
                    hints = hints.to(self.device)
                else:
                    num_hints_list = [0] * imgL.shape[0]  # Default to 0 hints
                
                sampled_images = self.sample_batch(imgL.to(self.device), imgAB.to(self.device),hints)
                self.update_metrics(sampled_images["images_pred_rgb"], sampled_images["images_original_rgb"], num_hints_list)
            
            # Save mean metrics every compute_every steps
                if self.step % self.compute_every == 0:
                    self.save_mean_to_csv()

                pbar.update(1)
                self.step += 1
        
        # Insert NaNs for any remaining metrics after the loop ends
        self.save_mean_to_csv()
