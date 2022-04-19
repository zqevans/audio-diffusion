from diffusion.utils import MidSideEncoding, MidSideDecoding, Stereo
import torch

from RAVE.rave.pqmf import PQMF

import unittest

class TestPqmf(unittest.TestCase):

    def test_pqmf_invertible(self):
        signal = torch.randn([1, 1, 131072])
        pqmf = PQMF(100, 128)
        encoded = pqmf(signal)
        decoded = pqmf.inverse(encoded)

        #Encoding and decoding should be a no-op
        self.assertTrue(torch.equal(signal, decoded))



if __name__ == '__main__':
    unittest.main()