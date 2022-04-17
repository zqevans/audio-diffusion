from diffusion.utils import MidSideEncoding, MidSideDecoding
import torch

import unittest

class TestUtils(unittest.TestCase):

    def test_mid_side_encoding(self):
        noise = torch.randn([1, 2, 131072], device='cuda')
        encoder = MidSideEncoding()
        decoder = MidSideDecoding()

        #Encoding and decoding should be a no-op
        assert(torch.equal(noise, decoder(encoder(noise))))

if __name__ == '__main__':
    unittest.main()