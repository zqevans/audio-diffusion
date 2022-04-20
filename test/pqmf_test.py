import torch

from diffusion.pqmf import CachedPQMF as PQMF

import unittest

class TestPqmf(unittest.TestCase):

    def test_pqmf_shapes_equal(self):
        signal = torch.randn([1, 1, 131072])
        pqmf = PQMF(1, 100, 32)
        encoded = pqmf(signal)
        decoded = pqmf.inverse(encoded)
        
        #the inverse has the same shape as the original
        self.assertEqual(list(signal.shape), list(decoded.shape))

    def test_pqmf_stereo_shapes(self):
        signal = torch.randn([1, 2, 131072])
        pqmf = PQMF(2, 100, 32)
        encoded = pqmf(signal)
        print(encoded.shape)
        decoded = pqmf.inverse(encoded)
        
        #the inverse has the same shape as the original
        self.assertEqual(list(signal.shape), list(decoded.shape))
        

if __name__ == '__main__':
    unittest.main()