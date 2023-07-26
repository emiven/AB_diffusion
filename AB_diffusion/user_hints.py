import torch
import math
import torch.nn.functional as F

class RandomHintGenerator:
    """Generates random hints for training.
    note: this is a heavily modified version of the original code from https://github.com/pengzhiliang/MAE-pytorch/blob/main/masking_generator.py

    """

    def __init__(self, input_size, hint_size=2, num_hint_range=[0, 10], uniform = False):
        #num_hint_range only wokrs for uniform gen
        #the hint size must be a factor of the input size, throw an error if not
        assert input_size % hint_size == 0, f'input size {input_size} must be a factor of hint size {hint_size}'

        self.uniform = uniform
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
        if self.uniform:
            return self.uniform_gen(batch_size)
        else:
            return self.geometric_gen(batch_size)
    

    def geometric_gen(self, batch_size = 1):
        """User hint generation using a geometric distribution per https://arxiv.org/abs/1705.02999

        """

        # Define the probability for the geometric distribution
        p = 1/8

        # Draw number of hints from a geometric distribution
        num_hints = torch.distributions.geometric.Geometric(p).sample((batch_size,))
        num_hints.clamp_(self.num_hint_range[0], self.num_hint_location)
        num_hints = num_hints.int()

        mask = torch.zeros((batch_size, self.num_hint_location))
        hints = torch.ones((batch_size, self.num_hint_location))
        # Define standard deviations for the Gaussian distribution
        std_h = (self.height / 4) * 0.8  # Note: 0.8 is an arbitrary factor, higher values lead hint location being closer to the edges for some reason
        std_w = (self.width / 4) * 0.8   
        hint_size = int(math.sqrt(self.height * self.width // self.num_hint_location))

        for i in range(batch_size):
            # Draw hint locations from a 2D Gaussian distribution
            loc_h = torch.clamp(torch.normal(0.5 * self.height, std_h, size=(num_hints[i],)), 0, self.height - 1).long()
            loc_w = torch.clamp(torch.normal(0.5 * self.width, std_w, size=(num_hints[i],)), 0, self.width - 1).long()
            loc_h = loc_h // hint_size
            loc_w = loc_w // hint_size
           
            # Compute hint indices
            hint_indices = loc_h * self.width // hint_size + loc_w
            mask[i, hint_indices] = 1

        hints[mask.bool()] = 0 
        return hints

    # Uniform hint generation
    def uniform_gen(self, batch_size = 1):
    

        num_zeroes = torch.randint(self.num_hint_range[0], self.num_hint_range[1], (batch_size,))
        # Create a tensor of ones
        hints = torch.ones((batch_size, self.num_hint_location))
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
    _full_mask = F.interpolate(_mask, scale_factor=hint_size)  

    full_mask = _full_mask.type(f'torch.BoolTensor')

    # mask ab channels
    if avg_color and hint_size >= 2:
        _avg_AB = F.interpolate(imgAB, size=(H // hint_size, W // hint_size), mode='bilinear')

        _avg_AB.masked_fill_(mask.unsqueeze(1).type(f'torch.BoolTensor').to(device), 0)
        return F.interpolate(_avg_AB, scale_factor=hint_size, mode='nearest')
    else:
        return imgAB[:, 0:, :, :].masked_fill(full_mask.squeeze(1), 0)

