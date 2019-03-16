from typing import Callable, Any, Optional

import torch
from torch.utils.data import Dataset, DataLoader

from pytorch_toolbox.utils import to_device
from pytorch_toolbox.training.defaults import default_collate, defaults


class DeviceDataLoader:

    @classmethod
    def create(cls, dataset: Dataset, batch_size: int = 64, shuffle: bool = False,
               device: torch.device = defaults.device,
               num_workers: int = defaults.cpus,
               collate_fn: Callable = default_collate, **kwargs: Any):
        dl = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, **kwargs)
        return cls(dl, device=device, collate_fn=collate_fn)

    @property
    def batch_size(self):
        return self.dl.batch_size

    @batch_size.setter
    def batch_size(self, batch_size):
        self.dl.batch_size = batch_size

    @property
    def num_workers(self):
        return self.dl.num_workers

    @num_workers.setter
    def num_workers(self, num_workers):
        self.dl.num_workers = num_workers

    def __init__(self, dl: DataLoader, device: torch.device,
                 collate_fn: Callable = default_collate):
        self.dl = dl
        self.device = device
        self.dl.collate_fn = collate_fn

    def proc_batch(self, batch):
        input_ = to_device(batch[0], self.device)
        output = {}
        for k, v in batch[1].items():
            if isinstance(v, torch.Tensor):
                output[k] = to_device(v, self.device)
            else:
                output[k] = v
        return input_, output

    def __iter__(self):
        for b in self.dl:
            yield self.proc_batch(b)


class DataBunch:
    def __init__(self, train_dl: DataLoader, valid_dl: DataLoader, test_dl: Optional[DataLoader] = None,
                 device: torch.device = None, collate_fn: Callable = default_collate):
        self.device = defaults.device if device is None else device
        self.train_dl = DeviceDataLoader(train_dl, self.device, collate_fn)
        self.valid_dl = DeviceDataLoader(valid_dl, self.device, collate_fn)
        self.test_dl = DeviceDataLoader(test_dl, self.device, collate_fn) if test_dl else None

    @classmethod
    def create(cls, train_ds: Dataset, valid_ds: Dataset, test_ds: Dataset = None,
               train_bs: int = 64, val_bs: int = None, test_bs: int = None, sampler=None,
               num_workers: int = defaults.cpus, pin_memory: bool = False,
               device: torch.device = None,
               collate_fn: Callable = default_collate) -> 'DataBunch':

        if val_bs is None:
            val_bs = train_bs * 2
        if test_bs is None:
            test_bs = train_bs * 2
        train_dl = DataLoader(train_ds, train_bs, sampler=sampler, num_workers=num_workers, pin_memory=pin_memory,
                              drop_last=True)
        val_dl = DataLoader(valid_ds, val_bs, shuffle=False, num_workers=num_workers)
        test_dl = DataLoader(test_ds, test_bs, shuffle=False, num_workers=num_workers)
        dls = [train_dl, val_dl, test_dl]
        return cls(*dls, device=device, collate_fn=collate_fn)