import sys

sys.path.append("../fastai")

from fastai import *
import fastai
from fastai import vision


class PytorchToolboxDeviceDataLoader(fastai.DeviceDataLoader):
    """
    This is subclasses because there are situations where the batch being returns may contain more than tensors e.g.
    maybe we also return the UID of the image. Hence the proc_batch function is overridden to provide this functionality
    """

    def proc_batch(self, b):
        input = fastai.to_device(b[0], self.device)
        output = {}
        for k, v in b[1].items():
            if isinstance(v, torch.Tensor):
                output[k] = fastai.to_device(v, None)
            else:
                output[k] = v
        return input, output

    def __iter__(self):
        for b in self.dl:
            yield self.proc_batch(b)

    @classmethod
    def create(cls, dataset: fastai.Dataset, bs: int = 64, shuffle: bool = False,
               device: torch.device = fastai.defaults.device,
               tfms: Collection[Callable] = None, num_workers: int = fastai.defaults.cpus,
               collate_fn: Callable = fastai.data_collate, **kwargs: Any):
        "Create DeviceDataLoader from `dataset` with `batch_size` and `shuffle`: processs using `num_workers`."
        return cls(DataLoader(dataset, batch_size=bs, shuffle=shuffle, num_workers=num_workers, **kwargs),
                   device=device, tfms=tfms, collate_fn=collate_fn)


class DeviceDataLoader:
    EXPOSED_ATTRIBUTES_FOR_DEVICE_DATA_LOADER = []

    @classmethod
    def create(cls, dataset: fastai.Dataset, bs: int = 64, shuffle: bool = False,
               device: torch.device = fastai.defaults.device, num_workers: int = fastai.defaults.cpus,
               collate_fn: Callable = fastai.data_collate, **kwargs: Any):
        device_data_loader = PytorchToolboxDeviceDataLoader(
            DataLoader(dataset, batch_size=bs, shuffle=shuffle, num_worker=num_workers, **kwargs),
            device=device, tfms=None, collate_fn=collate_fn)
        return cls(device_data_loader)

    @classmethod
    def create_with_initialized_dl(cls, dl, device, collate_fn):
        device_data_loader = PytorchToolboxDeviceDataLoader(dl=dl, device=device, tfms=None, collate_fn=collate_fn)
        return cls(device_data_loader)

    # def __getattr__(self, item):
    #     try:
    #         return getattr(self, item)
    #     except AttributeError:
    #         if item in self.EXPOSED_ATTRIBUTES_FOR_DEVICE_DATA_LOADER:
    #             return getattr(self.learner, item)
    #         else:
    #             raise AttributeError

    def __iter__(self):
        return self.device_data_loader.__iter__()

    def __init__(self, device_data_loader):
        self.device_data_loader = device_data_loader


class DataBunch:
    """
    This purpose of this subclass was to provide more customization on how to set the batch sizes for the val and test
    dataset, instead them being set @ twice the size of the train bs. This is because there are situations where we
    would want to train smaller images (maybe it speed up training, or for better generalization as for smaller images
    the higher features would have to be identitfied), but test on the full sized image.
    """
    "Bind `train_dl`,`valid_dl` and`test_dl` to `device`. tfms are DL tfms (normalize). `path` is for models."

    def __init__(self, train_dl: DataLoader, valid_dl: DataLoader, test_dl: Optional[DataLoader] = None,
                 device: torch.device = None, tfms: Optional[Collection[Callable]] = None, path: PathOrStr = '.',
                 collate_fn: Callable = data_collate):
        "Bind `train_dl`,`valid_dl` and`test_dl` to `device`. tfms are DL tfms (normalize). `path` is for models."
        self.tfms = fastai.listify(tfms)
        self.device = fastai.defaults.device if device is None else device
        self.train_dl = DeviceDataLoader.create_with_initialized_dl(train_dl, self.device, collate_fn)
        self.valid_dl = DeviceDataLoader.create_with_initialized_dl(valid_dl, self.device, collate_fn)
        self.test_dl = DeviceDataLoader.create_with_initialized_dl(test_dl, self.device,
                                                                   collate_fn) if test_dl else None
        self.path = Path(path)

    @classmethod
    def create(cls, train_ds: Dataset, valid_ds: Dataset, test_ds: Dataset = None, path: PathOrStr = '.',
               train_bs: int = 64, val_bs: int = None, test_bs: int = None, sampler=None,
               num_workers: int = fastai.defaults.cpus, pin_memory: bool = False,
               tfms: Optional[Collection[Callable]] = None,
               device: torch.device = None,
               collate_fn: Callable = data_collate) -> 'DataBunch':
        "`DataBunch` factory. `bs` batch size, `ds_tfms` for `Dataset`, `tfms` for `DataLoader`."
        if val_bs is None: val_bs = train_bs * 2
        if test_bs is None: test_bs = train_bs * 2

        datasets = [train_ds, valid_ds]
        if test_ds is not None: datasets.append(test_ds)
        if sampler is None:
            train_dl = DataLoader(train_ds, train_bs, shuffle=True, num_workers=num_workers, pin_memory=pin_memory,
                                  drop_last=True)
        else:
            train_dl = DataLoader(train_ds, train_bs, sampler=sampler, num_workers=num_workers, pin_memory=pin_memory,
                                  drop_last=True)
        val_dl = DataLoader(valid_ds, val_bs, shuffle=False, num_workers=num_workers)
        test_dl = DataLoader(test_ds, test_bs, shuffle=False, num_workers=num_workers)
        dls = [train_dl, val_dl, test_dl]
        return cls(*dls, path=path, device=device, tfms=tfms, collate_fn=collate_fn)
