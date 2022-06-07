from torch.utils import data
import pytorch_lightning as pl

from dataset.dataset import SampleDataset

class SampleDataModule(pl.LightningDataModule):
    def __init__(self, global_args):
        self.files = []
        self.global_args = global_args

    def setup(self):
        self.train_set = SampleDataset([self.global_args.training_dir], self.global_args)
        
    def train_dataloader(self):
        return data.DataLoader(self.train_set, self.global_args.batch_size, shuffle=True,
                                   num_workers=self.global_args.num_workers, persistent_workers=True, pin_memory=True)