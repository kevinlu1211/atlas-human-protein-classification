import sys
sys.path.append("../fastai")
from functools import partial

import torch
import numpy as np

from fastai import vision

def tensor2img(image_tensor, imtype=np.uint8, denorm_fn=None):
    shape = image_tensor.shape
    image_tensor = image_tensor.cpu()
    if denorm_fn is not None:
        image_tensor = denorm_fn(image_tensor)
    np_img = image_tensor.float().numpy()
    if len(shape) == 4:
        ret = []
        for img in np_img:
            if img.shape[0] == 1:
                img = np.tile(img, (3, 1, 1))
            ret.append(img)
        ret = np.transpose(np.array(ret), (0, 2, 3, 1))
    elif len(shape) == 3:
        if shape[0] == 1:
            np_img = np.tile(np_img, (3, 1, 1))
        ret = np.transpose(np_img, (1, 2, 0))
    elif len(shape) == 2:
        ret = np_img
    else:
        print("Expected a Tensor of C x H x W or B x C x H x W")
        return
    return (ret * 255.0).astype(imtype)


# ref.: https://www.kaggle.com/stainsby/fast-tested-rle
def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.transpose().flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T.astype(np.uint8)  # Needed to align to RLE direction


def rle_decode_with_nans(rle, shape):
    return rle_decode(str(rle[0])) if str(rle[0]) != 'nan' else np.zeros(shape).astype(np.uint8)


def open_mask(mask_path, div=False, with_name=False):
    mask = vision.open_mask(mask_path, div=div)
    if with_name:
        mask.name = mask_path.name
    return mask

def open_image(img_path, with_name=False):
    img = vision.open_image(img_path)
    if with_name:
        img.name = img_path.name
    return img


def normalize(tensor, *, mean, sd):
    mean = torch.Tensor(mean)
    sd = torch.Tensor(sd)
    tensor = ((tensor - mean[..., None, None]) / sd[..., None, None])
    return tensor


def denormalize(tensor, *, mean, sd):
    mean = torch.Tensor(mean)
    sd = torch.Tensor(sd)
    tensor = tensor * sd[..., None, None] + mean[..., None, None]
    return tensor


IMAGE_NET_STATS = {'mean': [0.485, 0.456, 0.406],
                   'sd': [0.229, 0.224, 0.225]}


FOUR_CHANNEL_PNASNET5LARGE_STATS = {
    'mean': [0.5, 0.5, 0.5, 0.5],
    'sd': [0.5, 0.5, 0.5, 0.5]
}

FOUR_CHANNEL_IMAGE_NET_STATS = {
    'mean': [0.485, 0.456, 0.406, 0.485],
    'sd': [0.229, 0.224, 0.224, 0.229]
}
image_net_normalize = partial(normalize, **IMAGE_NET_STATS)
image_net_denormalize = partial(denormalize, **IMAGE_NET_STATS)


four_channel_image_net_normalize = partial(normalize, **FOUR_CHANNEL_IMAGE_NET_STATS)
four_channel_image_net_denormalize = partial(denormalize, **FOUR_CHANNEL_IMAGE_NET_STATS)


four_channel_pnasnet5large_normalize = partial(normalize, **FOUR_CHANNEL_PNASNET5LARGE_STATS)
four_channel_pnasnet5large_denormalize = partial(denormalize, **FOUR_CHANNEL_PNASNET5LARGE_STATS)

normalize_fn_lookup = {
    "image_net_normalize": image_net_normalize,
    "four_channel_image_net_normalize": four_channel_image_net_normalize,
    "four_channel_pnasnet5large_normalize": four_channel_image_net_normalize,
    "identity": lambda x: x
}

denormalize_fn_lookup = {
    "image_net_denormalize": image_net_denormalize,
    "four_channel_image_net_denormalize": four_channel_image_net_denormalize,
    "four_channel_pnasnet5large_denormalize": four_channel_image_net_denormalize,
    "identity": lambda x: x
}