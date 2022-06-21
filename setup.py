from setuptools import setup, find_packages

setup(
    name='audio-diffusion',
    version='1.0.0',
    url='https://github.com/zqevans/audio-diffusion.git',
    author='Zach Evans',
    packages=find_packages(),    
    install_requires=[
        'auraloss',
        'einops',
        'fairscale',
        'nwt-pytorch',
        'perceiver-pytorch',
        'prefigure',
        'pytorch_lightning', 
        'torch',
        'torchaudio',
        'vector-quantize-pytorch',
        'wandb',
        'jukebox @ git+https://github.com/drscotthawley/jukebox.git'
        #'cached_conv @ git+https://github.com/caillonantoine/cached_conv.git#egg=cached_conv'
        #'udls @ git+https://github.com/caillonantoine/UDLS.git#egg=udls',
    ],
)