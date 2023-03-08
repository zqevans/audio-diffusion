import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import argparse
import torchaudio
from autoencoders.models import AudioAutoencoder, LatentAudioDiffusionAutoencoder

def load_audio(audio_dir, autoencoder):
    target_sample_rate = 48000
    target_sequence_length = 262144
    audio_data = []
    for filename in os.listdir(audio_dir):
        print(f'Processing {filename}...')
        if filename.endswith('.wav'):
            filepath = os.path.join(audio_dir, filename)
            audio, sample_rate = torchaudio.load(filepath)

            # Resample audio to target sample rate
            if sample_rate != target_sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
                audio = resampler(audio)

            # Pad or crop audio to target sequence length
            audio_length = audio.shape[1]
            if audio_length > target_sequence_length:
                audio = audio[:, :target_sequence_length]
            elif audio_length < target_sequence_length:
                pad_length = target_sequence_length - audio_length
                audio = torch.nn.functional.pad(audio, (0, pad_length))

            # Ensure audio is stereo
            if audio.shape[0] == 1:
                audio = torch.cat([audio, audio], dim=0)

            audio = audio.unsqueeze(0).float()
            encoded_audio = encode_audio(audio, autoencoder)
            print(encoded_audio.shape)
            audio_data.append(encoded_audio.cpu())

    return torch.cat(audio_data, dim=0)

def encode_audio(audio, autoencoder):
    encoded_audio = autoencoder.encode(audio.to("cuda"))
    return encoded_audio

def perform_pca(data):
    pca = PCA()
    pca.fit(data)
    pca_data = pca.transform(data)
    components = pca.components_
    explained_variance_ratios = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratios)
    informative_dimensions = np.argmax(cumulative_variance_ratio >= 0.95) + 1

    return pca_data, components, explained_variance_ratios, informative_dimensions

def plot_scree(explained_variance_ratios):
    plt.plot(np.arange(1, explained_variance_ratios.size+1), explained_variance_ratios, 'bo-', linewidth=2)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Scree Plot')
    plt.grid()

def save_plot(filename):
    plt.savefig(filename)

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Perform PCA on audio signals')
    parser.add_argument('audio_dir', metavar='audio_dir', type=str,
                        help='path to directory containing audio files')
    parser.add_argument('--pretrained_ckpt_path', type=str,
                        help='path to pretrained checkpoint')
    args = parser.parse_args()

    first_stage_config = {"capacity": 64, "c_mults": [2, 4, 8, 16, 32], "strides": [2, 2, 2, 2, 2], "latent_dim": 32}

    first_stage_autoencoder = AudioAutoencoder( 
        **first_stage_config
    ).eval()

    latent_diffae_config = {
        "second_stage_latent_dim": 32,
        "downsample_factors": [2, 2, 2, 2],
        "encoder_base_channels": 128,
        "encoder_channel_mults": [1, 2, 4, 8, 8],
        "encoder_num_blocks": [8, 8, 8, 8],
        "diffusion_channel_dims": [512] * 10
    }

    latent_diffae = LatentAudioDiffusionAutoencoder(autoencoder=first_stage_autoencoder, **latent_diffae_config).to("cuda").eval().requires_grad_(False)

    print(f'Loading pretrained diffAE checkpoint from {args.pretrained_ckpt_path}...')
    latent_diffae.load_state_dict(torch.load(args.pretrained_ckpt_path, map_location='cpu')['state_dict'])

    print(f'Loading audio files from {args.audio_dir}...')
    # Load the audio files and encode them
    audio_data = load_audio(args.audio_dir, latent_diffae)

    # Reshape the data to [num_samples, num_features]
    num_samples, num_channels, sequence_length = audio_data.shape
    print(f'Number of samples: {num_samples}')
    data = audio_data.permute(0, 2, 1).reshape(-1, num_channels).numpy()

    # Perform PCA
    print('Performing PCA...')
    pca_data, components, explained_variance_ratios, informative_dimensions = perform_pca(data)

    # Plot the scree plot
    print('Plotting scree plot...')
    plot_scree(explained_variance_ratios)

    # Save the plot as a PNG file
    save_plot('scree_plot.png')

    print(f'Number of informative dimensions: {informative_dimensions}')