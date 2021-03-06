import numpy as np
import torch


class ConcatDataset(torch.utils.data.Dataset):

    def __init__(self, dataloader_label, dataloader_unlabel):
        super().__init__()
        self.datasets = (dataloader_label, dataloader_unlabel)

    def __getitem__(self, index):
        return tuple(d[index] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)
    
    
class ConcatDataset3(torch.utils.data.Dataset):

    def __init__(self, dataloader_syn, dataloader_ITS, dataloader_real):
        super().__init__()
        self.datasets = (dataloader_syn, dataloader_ITS, dataloader_real)

    def __getitem__(self, index):
        return tuple(d[index] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


class ConcatDataset_return_max(torch.utils.data.Dataset):

    def __init__(self, dataloader_label, dataloader_unlabel, unlabel_index_dir):
        super().__init__()
        self.datasets = (dataloader_label, dataloader_unlabel)

        if unlabel_index_dir != '':
            self.unlabel_length = len(np.load(unlabel_index_dir))
        else:
            self.unlabel_length = len(dataloader_unlabel)

    def __getitem__(self, index):
        return tuple(d[index] for d in self.datasets)

    def __len__(self):
        return max(len(self.datasets[0]), self.unlabel_length)