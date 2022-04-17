from diffusion.utils import MidSideEncoding, MidSideDecoding
import torch

def testMidSideEncoding():
    noise = torch.randn([1, 2, 131072], device='cuda')
    encoder = MidSideEncoding()
    decoder = MidSideDecoding()

    #Encoding and decoding should be a no-op
    assert(torch.equal(noise, decoder(encoder(noise))))

def runTests():
    testMidSideEncoding()

if __name__ == '__main__':
    runTests()