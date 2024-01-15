import torch
import math
import numpy as np
class RandomHintGenerator:
    def __init__(self, hint_size_range=[1, 9], p=1/8, reveal_all_rate=0.01, mu_factor=2, sigma_factor=4,return_n_hints = False, fixed_hint_size = None,uniform_hint_range = None,uniform_hint_loc = False):
        self.hint_size_range = hint_size_range
        self.p = p
        self.reveal_all_rate = reveal_all_rate
        self.mu_factor = mu_factor
        self.sigma_factor = sigma_factor
        self.return_n_hints = return_n_hints
        self.fixed_hint_size = fixed_hint_size
        self.uniform_hint_range = uniform_hint_range
        self.uniform_hint_loc = uniform_hint_loc
    def generate_hints(self, color_images):
        B, C, H, W = color_images.shape
        B_ab = torch.zeros((B, H, W, 2), device=color_images.device, dtype=color_images.dtype)
        num_hint_list = []
        for i in range(B):
            #if random 0-1 is bigger than reveal rate, B_ab[i] = color_images[idx].clone().permute(1, 2, 0)
            if np.random.uniform() < self.reveal_all_rate:
                B_ab[i] = color_images[i].clone().permute(1, 2, 0)
                num_hint_list.append(0)
                continue
            num_hints = self._sample_num_hints()
            num_hint_list.append(num_hints)
            for _ in range(num_hints):
                self._place_hint(B_ab[i], color_images[i])
        
        if self.return_n_hints:
            return B_ab.permute(0, 3, 1, 2), num_hint_list
        return B_ab.permute(0, 3, 1, 2)
    
    def _place_hint(self, B_ab, color_image):
        H, W = B_ab.shape[:2]
        hint_size = self._sample_hint_size()
        hint_location = self._sample_hint_location(H, W)
        
        y, x = hint_location
        h, w = hint_size
    
        # Ensure the hint patch is completely within the image boundaries
        y = min(max(0, y), H - h)
        x = min(max(0, x), W - w)
        
        # Extract the patch from the ground truth AB channels
        # Ensure we are extracting a 2D region from both the A and B channels
        patch = color_image[:, y:y+h, x:x+w]
        
        # Debug: Print the shape of the patch
        #print("Patch Shape:", patch.shape)
        
        # Compute the mean across the spatial dimensions (1 and 2) 
        # to get an average color with two components (A and B)
        patch_mean = torch.mean(patch, dim=(1, 2))
        
        # Debug: Print the shape of patch_mean
        #print("Patch Mean Shape:", patch_mean.shape)
        
        # Assign this average color to the hint patch in B_ab
        B_ab[y:y+h, x:x+w, 0] = patch_mean[0]
        B_ab[y:y+h, x:x+w, 1] = patch_mean[1]

    def _sample_hint_location(self, H, W):

        if self.uniform_hint_loc is not None:
            return (np.random.randint(0, H-2), np.random.randint(0, W-2))
        else:
            mu = torch.tensor([H / self.mu_factor, W / self.mu_factor])
            sigma = torch.tensor([(H / self.sigma_factor) ** 2, (W / self.sigma_factor) ** 2])
            return torch.normal(mean=mu, std=sigma.sqrt()).to(dtype=torch.int)

    def _sample_hint_size(self):
        if self.fixed_hint_size is not None:
            return (self.fixed_hint_size, self.fixed_hint_size)
        else:
            size = np.random.randint(self.hint_size_range[0], self.hint_size_range[1] + 1)
            return (size, size)

    def _sample_num_hints(self):
        if self.uniform_hint_range is not None:
            return np.random.choice(self.uniform_hint_range) 
        else:
            return np.random.geometric(p=self.p) - 1
