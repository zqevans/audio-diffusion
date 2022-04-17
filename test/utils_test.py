from diffusion.utils import MidSideEncoding, MidSideDecoding
import torch

import unittest

class TestUtils(unittest.TestCase):

    def test_mid_side_encoding_invertible(self):
        signal = torch.randn([2, 131072], device='cuda')
        encoder = MidSideEncoding()
        decoder = MidSideDecoding()

        #Encoding and decoding should be a no-op
        assert(torch.equal(signal, decoder(encoder(signal))))

    def test_mid_side_encoding_mono(self):
        mono_signal = torch.randn([1, 131072], device='cuda')
        # Make the signal mono
        signal = mono_signal.repeat(2, 1)
        encoder = MidSideEncoding()
        encoded_mono = encoder(signal)

        #Side content should be all zeros
        zeros = torch.zeros_like(encoded_mono[1])
        assert(torch.equal(zeros, encoded_mono[1]))


if __name__ == '__main__':
    unittest.main()