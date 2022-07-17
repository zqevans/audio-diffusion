import torch
import auraloss

class PerceptualSumAndDifferenceSTFTLoss(torch.nn.Module):
    """Sum and difference sttereo STFT loss module.
    See [Steinmetz et al., 2020](https://arxiv.org/abs/2010.10291)
    Args:
        fft_sizes (list, optional): List of FFT sizes.
        hop_sizes (list, optional): List of hop sizes.
        win_lengths (list, optional): List of window lengths.
        window (str, optional): Window function type.
        w_sum (float, optional): Weight of the sum loss component. Default: 1.0
        w_diff (float, optional): Weight of the difference loss component. Default: 1.0
        output (str, optional): Format of the loss returned.
            'loss' : Return only the raw, aggregate loss term.
            'full' : Return the raw loss, plus intermediate loss terms.
            Default: 'loss'
    Returns:
        loss:
            Aggreate loss term. Only returned if output='loss'.
        loss, sum_loss, diff_loss:
            Aggregate and intermediate loss terms. Only returned if output='full'.
    """

    def __init__(
        self,
        fft_sizes=[1024, 2048, 512],
        hop_sizes=[120, 240, 50],
        win_lengths=[600, 1200, 240],
        window="hann_window",
        w_sum=1.0,
        w_diff=1.0,
        output="loss",
        sample_rate=48000,
        **stft_args
    ):
        super(PerceptualSumAndDifferenceSTFTLoss, self).__init__()
        self.sd = auraloss.perceptual.SumAndDifference()
        self.w_sum = 1.0
        self.w_diff = 1.0
        self.output = output
        self.mrstft = auraloss.freq.MultiResolutionSTFTLoss(fft_sizes, hop_sizes, win_lengths, window, sample_rate=sample_rate, **stft_args)
        self.aw_fir = auraloss.perceptual.FIRFilter(filter_type="aw", fs=sample_rate)

    def forward(self, input, target):
        # input_sum, input_diff = self.aw_fir(*self.sd(input))
        # target_sum, target_diff = self.aw_fir(*self.sd(target))

        input_sum, input_diff = self.sd(input)
        target_sum, target_diff = self.sd(target)

        sum_loss = self.mrstft(input_sum, target_sum)
        diff_loss = self.mrstft(input_diff, target_diff)
        loss = ((self.w_sum * sum_loss) + (self.w_diff * diff_loss)) / 2

        if self.output == "loss":
            return loss
        elif self.output == "full":
            return loss, sum_loss, diff_loss