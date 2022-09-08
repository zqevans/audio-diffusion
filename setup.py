from setuptools import setup, find_packages

setup(
    name='audio-diffusion',
    version='1.0.0',
    url='https://github.com/zqevans/audio-diffusion.git',
    author='Zach Evans',
    packages=find_packages(),    
    install_requires=[
        'accelerate',
        'auraloss',
        'einops',
        'fairscale',
        'nwt-pytorch',
        'pandas',
        'perceiver-pytorch',
        'prefigure',
        'pytorch_lightning', 
        'scipy',
        'torch',
        'torchaudio',
        'tqdm',
        'transformers',
        'vector-quantize-pytorch',
        'wandb',
 #       'jukebox @ git+https://github.com/drscotthawley/jukebox.git',
        'cached_conv @ git+https://github.com/caillonantoine/cached_conv.git',
        'udls @ git+https://github.com/caillonantoine/UDLS.git',
    ],
)
