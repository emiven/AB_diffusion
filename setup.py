from setuptools import setup, find_packages

setup(
  name = 'AB_diffusion',
  packages = find_packages(),
  install_requires=[
    'accelerate',
    'einops',
    'ema-pytorch',
    'numpy',
    'pillow',
    'pytorch-fid',
    'torch',
    'torchvision',
    'tqdm',
    "kornia",
    "scikit-image",

  ])