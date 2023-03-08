import torch
import torch.nn as nn
import numpy as np


class LogGaborFilterBank(nn.Module):
    def __init__(self, num_filters, filter_length, audio_channels, sample_rate):
        super(LogGaborFilterBank, self).__init__()

        self.num_filters = num_filters
        self.filter_length = filter_length
        self.audio_channels = audio_channels
        self.sample_rate = sample_rate

        self.filters = self._create_filters()

    def _create_filters(self):
        filters = []

        # Define parameters for Log Gabor filters
        min_center_frequency = 50
        max_center_frequency = self.sample_rate // 2
        bandwidth = 1

        # Calculate center frequencies in log scale
        center_frequencies = np.geomspace(
            min_center_frequency, max_center_frequency, self.num_filters)

        for channel in range(self.audio_channels):
            channel_filters = []
            for freq in center_frequencies:
                t = torch.linspace(-self.filter_length // 2, self.filter_length //
                                   2, self.filter_length) / self.sample_rate
                omega = 2 * np.pi * freq
                sigma = freq / bandwidth

                # Create the filter in the frequency domain
                log_gabor = torch.exp(-((torch.log(omega * t) -
                                      torch.log(freq)) ** 2) / (2 * (np.log(sigma)) ** 2))
                # Set the DC component to zero
                log_gabor[self.filter_length // 2] = 0

                # Transform the filter to the time domain
                log_gabor_time = torch.ifft(log_gabor, n=self.filter_length)

                channel_filters.append(log_gabor_time.view(1, 1, -1))

            filters.append(torch.cat(channel_filters, dim=0))

        return nn.Parameter(torch.cat(filters, dim=1), requires_grad=False)

    def forward(self, x):
        """
        Apply the filter bank to an input audio signal.

        Args:
            x (torch.Tensor): Input audio signal with shape (batch_size, audio_channels, signal_length).

        Returns:
            torch.Tensor: Encoded audio signal with shape (batch_size, num_filters * audio_channels, output_length).
        """
        batch_size, audio_channels, signal_length = x.size()

        # Apply the filter bank to the input audio signal
        x = x.view(batch_size * audio_channels, 1, signal_length)
        y = torch.conv1d(x, self.filters, padding=self.filter_length // 2)

        # Reshape the output tensor
        output_length = y.size(-1)
        y = y.view(batch_size, self.num_filters *
                   self.audio_channels, output_length)

        return y

    def decode(self, encoded_audio):
        """
        Apply the inverse filter bank to the encoded audio signal.

        Args:
            encoded_audio (torch.Tensor): Encoded audio signal with shape (batch_size, num_filters * audio_channels, signal_length).

        Returns:
            torch.Tensor: Decoded audio signal with shape (batch_size, audio_channels, output_length).
        """
        batch_size, _, signal_length = encoded_audio.size()
        output_length = signal_length + self.filter_length - 1

        # Reshape the encoded audio tensor
        encoded_audio = encoded_audio.view(
            batch_size, self.num_filters, self.audio_channels, -1)

        # Transpose the tensor to have the filter dimension first
        encoded_audio = encoded_audio.permute(0, 1, 3, 2)

        # Apply the inverse filter bank to each frequency channel and channel separately
        decoded_audio = []
        for channel in range(self.audio_channels):
            channel_filters = self.filters[:, channel *
                                        self.num_filters:(channel+1)*self.num_filters, :]
            channel_encoded_audio = encoded_audio[:, :, :, channel]
            channel_decoded_audio = torch.conv1d(
                channel_encoded_audio, channel_filters, padding=self.filter_length - 1)
            decoded_audio.append(channel_decoded_audio)

        # Reshape the output tensor
        decoded_audio = torch.stack(decoded_audio, dim=-1)
        decoded_audio = decoded_audio.view(
            batch_size, self.audio_channels, output_length)

        return decoded_audio
