import torch
from IPython.utils import io as iol
import skimage.color as skcolor
import numpy as np

# Various color handling functions, normalizing, denormalizing, converisons, etc.


def plotMinMax(image_tensor):
    """
    Prints the minimum and maximum values of each color channel in the image tensor.

    Args:
        image_tensor (torch.Tensor): Tensor representing an image.
    """
    labels = ["C{}".format(i) for i in range(image_tensor.shape[1])]
    print("______________________________")
    for i, lab in enumerate(labels):
        mi = torch.min(image_tensor[:, i, :, :])
        ma = torch.max(image_tensor[:, i, :, :])
        print("{} : MIN={:8.4f}, MAX={:8.4f}".format(lab, mi.item(), ma.item()))


def de_normalize_lab(LAB):
    """
    Denormalizes a LAB color space tensor.

    Args:
        LAB (torch.Tensor): Tensor in LAB color space.

    Returns:
        torch.Tensor: Denormalized LAB tensor.
    """
    if LAB.shape[1] == 3:
        L, A, B = LAB[:, 0, :, :], LAB[:, 1, :, :], LAB[:, 2, :, :] 
        L = (L + 1) * 50.0
        A = A * 128.0
        B = B * 128.0
        return torch.stack([L, A, B], axis=1)
    elif LAB.shape[1] == 2:
        LAB = LAB * 128.0
        return LAB
    elif LAB.shape[1] == 1:
        return (LAB + 1) * 50.0
    else:
        print("shape: " + str(LAB.shape))
        raise ValueError("LAB shape not recognized")


def normalize_lab(LAB):
    """
    Normalizes a LAB color space tensor.

    Args:
        LAB (torch.Tensor): Tensor in LAB color space.

    Returns:
        torch.Tensor: Normalized LAB tensor.
    """
    if LAB.shape[1] == 3:
        L, A, B = LAB[:, 0, :, :], LAB[:, 1, :, :], LAB[:, 2, :, :]
        L = L / 50.0 - 1.0
        A = A / 128.0
        B = B / 128.0
        return torch.stack([L, A, B], axis=1)
    elif LAB.shape[1] == 2:
        LAB = LAB / 128.0
        return LAB
    elif LAB.shape[1] == 1:
        return LAB / 50.0 - 1.0
    else:
        print("shape: " + str(LAB.shape))
        raise ValueError("LAB shape not recognized")


def LAB2RGB(im_lab):
    """
    Converts an image tensor from LAB color space to RGB color space.

    Args:
        im_lab (torch.Tensor): Tensor in LAB color space.

    Returns:
        np.array: Image array in RGB color space.
    """
    lab = de_normalize_lab(im_lab)
    lab = lab.permute(0, 2, 3, 1)
    with iol.capture_output() as captured:
        rgb = skcolor.lab2rgb(lab)

    return rgb


class PointColorConversions:
    """
    A class to convert colors between different formats.
    """
    def __init__(self):
        pass

    def rgb_to_hex(self, color_rgb):
        """
        Converts RGB color to hexadecimal color.

        Args:
            color_rgb (tuple): RGB color.

        Returns:
            str: Hexadecimal color.
        """
        return '#%02x%02x%02x' % color_rgb

    def hex_to_rgb(self, color_hex):
        """
        Converts hexadecimal color to RGB color.

        Args:
            color_hex (str): Hexadecimal color.

        Returns:
            tuple: RGB color.
        """
        color_hex = color_hex.lstrip('#')
        return tuple(int(color_hex[i:i+2], 16) for i in (0, 2, 4))

    def rgb_to_lab(self, color_rgb):
        """
        Converts RGB color to LAB color.

        Args:
            color_rgb (tuple): RGB color.

        Returns:
            tuple: LAB color.
        """
        color_rgb = np.array(color_rgb).reshape(1, 1, 3) / 255.0
        color_lab = skcolor.rgb2lab(color_rgb)
        return tuple(color_lab[0, 0])

    def lab_to_rgb(self, color_lab):
        """
        Converts LAB color to RGB color.

        Args:
            color_lab (tuple): LAB color.

        Returns:
            tuple: RGB color.
        """
        color_lab = np.array(color_lab).reshape(1, 1, 3)
        color_rgb = skcolor.lab2rgb(color_lab)
        return tuple((color_rgb[0, 0] * 255).astype(int))

    def hex_to_lab(self, color_hex):
        """
        Converts hexadecimal color to LAB color.

        Args:
            color_hex (str): Hexadecimal color.

        Returns:
            np.array: LAB color.
        """
        color_rgb = np.array(self.hex_to_rgb(color_hex)) / 255.0
        return skcolor.rgb2lab(color_rgb.reshape(1, 1, 3))[0, 0]

    def lab_to_hex(self, color_lab):
        """
        Converts LAB color to hexadecimal color.

        Args:
            color_lab (tuple): LAB color.

        Returns:
            str: Hexadecimal color.
        """
        color_rgb = self.lab_to_rgb(color_lab)
        return self.rgb_to_hex(color_rgb)
