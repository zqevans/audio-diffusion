import torch 
import torch.nn.functional as F 


class KernelPredictor(torch.nn.Module):
    ''' Kernel predictor for the location-variable convolutions
    '''

    def __init__(self, 
                 cond_channels,
                 conv_in_channels,
                 conv_out_channels,
                 conv_layers,
                 conv_kernel_size=3,
                 kpnet_hidden_channels=64,
                 kpnet_conv_size=1,
                 kpnet_dropout=0.0,
                 kpnet_nonlinear_activation="LeakyReLU",
                 kpnet_nonlinear_activation_params={"negative_slope":0.1}
                 ):
        '''
        Args:
            cond_channels (int): number of channel for the conditioning sequence,
            conv_in_channels (int): number of channel for the input sequence,
            conv_out_channels (int): number of channel for the output sequence,
            conv_layers (int):
            kpnet_
        '''
        super().__init__() 

        self.conv_in_channels = conv_in_channels 
        self.conv_out_channels = conv_out_channels 
        self.conv_kernel_size = conv_kernel_size 
        self.conv_layers = conv_layers

        kpnet_kernel_channels = conv_in_channels * conv_out_channels * conv_kernel_size * conv_layers 
        kpnet_bias_channels = conv_out_channels * conv_layers 

        padding = (kpnet_conv_size - 1)//2 
        self.input_conv = torch.nn.Sequential(
            torch.nn.Conv1d(cond_channels, kpnet_hidden_channels, 5, padding=0, bias=True), 
            getattr(torch.nn, kpnet_nonlinear_activation)(**kpnet_nonlinear_activation_params),
        )

        self.residual_conv = torch.nn.Sequential(
            torch.nn.Dropout(kpnet_dropout), 
            torch.nn.Conv1d(kpnet_hidden_channels, kpnet_hidden_channels, kpnet_conv_size, padding=padding, bias=True), 
            getattr(torch.nn, kpnet_nonlinear_activation)(**kpnet_nonlinear_activation_params),
            torch.nn.Conv1d(kpnet_hidden_channels, kpnet_hidden_channels, kpnet_conv_size, padding=padding, bias=True), 
        )

        self.kernel_conv = torch.nn.Conv1d(kpnet_hidden_channels, kpnet_kernel_channels, kpnet_conv_size, padding=padding, bias=True) 
        self.bias_conv = torch.nn.Conv1d(kpnet_hidden_channels, kpnet_bias_channels, kpnet_conv_size, padding=padding, bias=True) 
        
    def forward(self, c):
        '''
        Args:
            c (Tensor): the conditioning sequence (batch, cond_channels, cond_length) 
        Returns:
        '''
        batch, cond_channels, cond_length = c.shape 

        c = self.input_conv( c ) 
        c = c + self.residual_conv( c ) 
        k = self.kernel_conv( c ) 
        b = self.bias_conv( c ) 
        kernels = k.contiguous().view( batch, 
                                      self.conv_layers,
                                      self.conv_in_channels,
                                      self.conv_out_channels,
                                      self.conv_kernel_size,
                                      cond_length - 4 ) 
        bias = b.contiguous().view( batch, 
                                    self.conv_layers, 
                                    self.conv_out_channels,
                                    cond_length - 4 ) 
        return kernels, bias



class LVCBlock(torch.nn.Module):
    ''' the location-variable convolutions 
    '''

    def __init__(self,
                 in_channels,
                 cond_channels,
                 conv_layers=10,
                 conv_kernel_size=3,
                 cond_hop_length=256,
                 kpnet_hidden_channels=64,
                 kpnet_conv_size=1,
                 kpnet_dropout=0.0
                 ):
        super().__init__() 

        self.cond_hop_length = cond_hop_length 
        self.conv_layers = conv_layers 
        self.conv_kernel_size = conv_kernel_size 

        self.kernel_predictor = KernelPredictor( 
                    cond_channels=cond_channels,
                    conv_in_channels=in_channels,
                    conv_out_channels=2*in_channels, 
                    conv_layers=conv_layers,
                    conv_kernel_size=conv_kernel_size,
                    kpnet_hidden_channels=kpnet_hidden_channels,
                    kpnet_conv_size=kpnet_conv_size,
                    kpnet_dropout=kpnet_dropout 
                    ) 

    def forward(self, x, c):
        ''' forward propagation of the location-variable convolutions.  
        Args: 
            x (Tensor): the input sequence (batch, in_channels, in_length) 
            c (Tensor): the conditioning sequence (batch, cond_channels, cond_length)
        
        Returns:
            Tensor: the output sequence (batch, in_channels, in_length) 
        ''' 
        batch, in_channels, in_length = x.shape 
        batch, cond_channels, cond_length = c.shape 
        assert in_length == ( (cond_length - 4) * self.cond_hop_length ), ( 
            f"the length of input ({in_length}, {cond_length}) is not match in LVCNet" ) 

        kernels, bias = self.kernel_predictor( c ) 

        for i in range(self.conv_layers):
            dilation = 2**i 
            k = kernels[ :, i, :, :, :, : ] 
            b = bias[ :, i, :, : ] 
            x = self.location_variable_convolution( x, k, b, dilation, self.cond_hop_length ) 
            x = torch.sigmoid( x[ :, :in_channels, : ] ) * torch.tanh( x[ :, in_channels:, : ] ) 
        return x 
            
    
    def location_variable_convolution(self, x, kernel, bias, dilation, hop_size):
        ''' perform location-variable convolution operation on the input sequence (x) using the local convolution kernl. 
        Time: 414 μs ± 309 ns per loop (mean ± std. dev. of 7 runs, 1000 loops each), test on NVIDIA V100. 
        Args:
            x (Tensor): the input sequence (batch, in_channels, in_length). 
            kernel (Tensor): the local convolution kernel (batch, in_channel, out_channels, kernel_size, kernel_length) 
            bias (Tensor): the bias for the local convolution (batch, out_channels, kernel_length) 
            dilation (int): the dilation of convolution. 
            hop_size (int): the hop_size of the conditioning sequence. 
        Returns:
            (Tensor): the output sequence after performing local convolution. (batch, out_channels, in_length).
        '''
        batch, in_channels, in_length = x.shape 
        batch, in_channels, out_channels, kernel_size, kernel_length = kernel.shape 

        assert in_length == (kernel_length*hop_size), "length of (x, kernel) is not matched" 

        padding = dilation * int( (kernel_size - 1) / 2 ) 
        x = F.pad( x, (padding, padding), 'constant', 0 )      # (batch, in_channels, in_length + 2*padding)
        x = x.unfold( 2, hop_size + 2 * padding, hop_size )    # (batch, in_channels, kernel_length, hop_size + 2*padding)

        if hop_size < dilation:
            x = F.pad( x, (0, dilation), 'constant', 0 ) 
        x = x.unfold(3, dilation, dilation)     # (batch, in_channels, kernel_length, (hop_size + 2*padding)/dilation, dilation)
        x = x[ :, :, :, :, :hop_size ]          
        x = x.transpose( 3, 4 )                 # (batch, in_channels, kernel_length, dilation, (hop_size + 2*padding)/dilation)  
        x = x.unfold( 4, kernel_size, 1 )       # (batch, in_channels, kernel_length, dilation, _, kernel_size)

        o = torch.einsum( 'bildsk,biokl->bolsd', x, kernel ) 
        o = o + bias.unsqueeze(-1).unsqueeze(-1) 
        o = o.contiguous().view(batch, out_channels, -1) 
        return o 
