from diffusion.utils import MidSideEncoding, MidSideDecoding, Stereo
import torch

import unittest

class TestUtils(unittest.TestCase):

    def test_mid_side_encoding_invertible(self):
        signal = torch.randn([2, 131072], device='cuda')
        encoder = MidSideEncoding()
        decoder = MidSideDecoding()

        #Encoding and decoding should be a no-op
        self.assertTrue(torch.equal(signal, decoder(encoder(signal))))

    def test_mid_side_encoding_mono(self):
        mono_signal = torch.randn([1, 131072], device='cuda')
        # Make the signal stereo
        signal = mono_signal.repeat(2, 1)
        encoder = MidSideEncoding()
        encoded_mono = encoder(signal)

        #Mid content should be the same as the mono signal
        self.assertTrue(torch.equal(mono_signal, encoded_mono[0]))

        #Side content should be all zeros
        zeros = torch.zeros_like(encoded_mono[1])
        self.assertTrue(torch.equal(zeros, encoded_mono[1]))

    def test_mono_to_stereo(self):
        mono_signal = torch.randn([1, 131072], device='cuda')
        stereo_signal = Stereo()(mono_signal)
        signal_shape = stereo_signal.shape
        self.assertEqual(len(signal_shape), 2)
        self.assertEqual(signal_shape[0], 2)
        self.assertEqual(signal_shape[1], 131072)

    def test_surround_to_stereo(self):
        mono_signal = torch.randn([6, 131072], device='cuda')
        stereo_signal = Stereo()(mono_signal)
        signal_shape = stereo_signal.shape
        self.assertEqual(len(signal_shape), 2)
        self.assertEqual(signal_shape[0], 2)
        self.assertEqual(signal_shape[1], 131072)

    def test_one_channel_to_stereo(self):
        mono_signal = torch.randn([131072], device='cuda')
        stereo_signal = Stereo()(mono_signal)
        signal_shape = stereo_signal.shape
        self.assertEqual(len(signal_shape), 2)
        self.assertEqual(signal_shape[0], 2)
        self.assertEqual(signal_shape[1], 131072)


if __name__ == '__main__':
    unittest.main()