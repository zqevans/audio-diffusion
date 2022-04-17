from utils import MidSideEncoding, MidSideDecoding
import torch

def testMidSideEncoding():
    #make a noise signal
    noise = torch.randn([4, 2, 131072], device='cuda')
    encoded_noise = MidSideEncoding()(noise)
    decoded_noise = MidSideDecoding()(encoded_noise) 
    assert(torch.equal(noise, decoded_noise))

def runTests():
    testMidSideEncoding()

if __name__ == '__main__':
    runTests()