# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'

import numpy as np
import torch
import math
import torch.nn.functional as F
from torchvision import transforms as T

class RandomHintGenerator:
    '''
    Use RandomHintGenerator in BEiT as random hint generator
    '''

    def __init__(self, input_size, hint_size=2, num_hint_range=[0, 10]):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size
        self.num_hint_location = self.height * self.width // (hint_size * hint_size)
        self.num_hint_range = num_hint_range

    def __repr__(self):
        repr_str = (f'Hint: total hint locations {self.num_hint_location},'
                    f'number of hints range {self.num_hint_range}')
        return repr_str

    def __call__(self, batch_size):
        return self.uniform_gen(batch_size)

    def uniform_gen(self, batch_size = 1):
    

        # Generate random number of zeroes for each row
        num_zeroes = torch.randint(self.num_hint_range[0], self.num_hint_range[1]+1, (batch_size,))
        # Create a tensor of ones
        hints = torch.ones((batch_size, self.num_hint_location))
        # Generate mask for indices to be set to zero
        mask = torch.zeros((batch_size, self.num_hint_location))
        for i in range(batch_size):
            zero_indices = torch.randperm(self.num_hint_location)[:num_zeroes[i]]
            mask[i, zero_indices] = 1
        # Set selected indices to zero
        hints[mask.bool()] = 0
        return hints
    
def get_color_hints(imgAB = None, hints = None, avg_color = True, device = "cpu"):
    B, _, H, W = imgAB.shape
    _, L = hints.shape
    # assume square inputs
    hint_size = int(math.sqrt(H * W // L))
    mask = torch.reshape(hints, (B, H // hint_size, W // hint_size)).to(device)
    _mask = mask.unsqueeze(1).type(f'torch.FloatTensor')
    _full_mask = F.interpolate(_mask, scale_factor=hint_size)  # Needs to be Float


    full_mask = _full_mask.type(f'torch.BoolTensor')

    # mask ab channels
    if avg_color and hint_size >= 2:
        _avg_AB = F.interpolate(imgAB, size=(H // hint_size, W // hint_size), mode='bilinear')

        _avg_AB.masked_fill_(mask.unsqueeze(1).type(f'torch.BoolTensor').to(device), 0)
        return F.interpolate(_avg_AB, scale_factor=hint_size, mode='nearest')
    else:
        return imgAB[:, 0:, :, :].masked_fill(full_mask.squeeze(1), 0)

