from setuptools import setup, find_packages

setup(
    name='audio-diffusion',
    version='1.0.0',
    url='https://github.com/zqevans/audio-diffusion.git',
    author='Zach Evans',
    packages=find_packages(),    
    dependency_links=[
        "https://github.com/caillonantoine/cached_conv.git#egg=cached_conv",
        "https://github.com/caillonantoine/UDLS.git#egg=udls",
    ],
    install_requires=[
        'auraloss',
        'einops',
        'perceiver_pytorch',
        'pytorch_lightning', 
        'torch',
        'torchaudio',
        'vector_quantize_pytorch',
        'wandb'
    ],
)