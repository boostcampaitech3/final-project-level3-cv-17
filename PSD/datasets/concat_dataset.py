import torch


class ConcatDataset(torch.utils.data.Dataset):

    def __init__(self, dataloader_syn, dataloader_real):

        super().__init__()
        self.datasets = (dataloader_syn, dataloader_real)

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