from functools import partial

import torch
import torchvision
from torchvision.transforms import FiveCrop, ToTensor, Lambda
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose, ElasticTransform,
    Resize, Crop, RandomBrightnessContrast, RandomCrop
)


def very_simple_aug(p=0.5, height=None, width=None):
    augs = [
        RandomRotate90(p=p),
        Flip(p=p),
        ShiftScaleRotate(shift_limit=0.05, scale_limit=0.2, rotate_limit=90, p=p),
        RandomBrightnessContrast(brightness_limit=3, contrast_limit=0.5, p=p)
    ]
    if height is None and width is None:
        return Compose(augs, p=1)
    else:
        if height is not None and width is None:
            width = height
    if width is not None and height is None:
        height = width
    return Compose([Resize(height=height, width=width, always_apply=True)] + augs, p=1)


def crop_rotate_flip(crop_height, crop_width, with_image_wrapper=False):
    augs = [
        RandomCrop(crop_height, crop_width),
        Flip(),
        RandomRotate90(),
        RandomBrightnessContrast(brightness_limit=3, contrast_limit=0.5)
    ]
    if with_image_wrapper:
        return partial(albumentations_transform_wrapper, augment_fn=augs)
    else:
        return augs


def very_simple_aug_with_elastic_transform(p=1, height=None, width=None, with_image_wrapper=False):
    augs = [
        Flip(),
        RandomRotate90(),
        RandomBrightnessContrast(brightness_limit=3, contrast_limit=0.5),
        ElasticTransform(sigma=50, alpha_affine=50),

    ]
    if height is None and width is None:
        augs = Compose(augs, p=p)
    else:
        if height is not None and width is None:
            width = height
        if width is not None and height is None:
            height = width
        augs = Compose([Resize(height=height, width=width, always_apply=True)] + augs, p=1)
    if with_image_wrapper:
        return partial(albumentations_transform_wrapper, augment_fn=augs)
    else:
        return augs


def very_simple_aug_with_elastic_transform_and_crop(height=None, width=None,
                                                    crop_height=None, crop_width=None, with_image_wrapper=False):
    augs = [
        RandomCrop(crop_height, crop_width),
        RandomRotate90(),
        Flip(),
        RandomBrightnessContrast(brightness_limit=3, contrast_limit=0.5),
        ElasticTransform(sigma=50, alpha_affine=50, p=0.5),
    ]
    if height is None and width is None:
        augs = Compose(augs, p=1)
    else:
        if height is not None and width is None:
            width = height
        if width is not None and height is None:
            height = width
        augs = Compose([Resize(height=height, width=width, always_apply=True)] + augs, p=1)
    if with_image_wrapper:
        return partial(albumentations_transform_wrapper, augment_fn=augs)
    else:
        return augs


def simple_aug(p=1, height=None, width=None):
    augs = [
        RandomRotate90(),
        Flip(),
        ShiftScaleRotate(shift_limit=0.05, scale_limit=0.2, rotate_limit=90),
        RandomBrightnessContrast(brightness_limit=3, contrast_limit=0.5),
        Blur(blur_limit=5),
        ElasticTransform(sigma=50, alpha_affine=50)
    ]
    if height is None and width is None:
        return Compose(augs, p=p)
    else:
        if height is not None and width is None:
            width = height
        if width is not None and height is None:
            height = width
        return Compose([Resize(height=height, width=width, always_apply=True)] + augs, p=p)


def simple_aug_lower_prob(p=1, height=None, width=None):
    augs = [
        RandomRotate90(),
        Flip(),
        ShiftScaleRotate(shift_limit=0.05, scale_limit=0.2, rotate_limit=90, p=0.25),
        RandomBrightnessContrast(brightness_limit=3, contrast_limit=0.5, p=0.25),
        Blur(blur_limit=5, p=0.25),
        ElasticTransform(sigma=50, alpha_affine=50, p=0.25)
    ]
    if height is None and width is None:
        return Compose(augs, p=p)
    else:
        if height is not None and width is None:
            width = height
        if width is not None and height is None:
            height = width
        return Compose([Resize(height=height, width=width, always_apply=True)] + augs, p=p)


def resize_aug(p=1, height=None, width=None, with_image_wrapper=False):
    augs = Compose([Resize(height=height, width=width)], p=p)
    if with_image_wrapper:
        return partial(albumentations_transform_wrapper, augment_fn=augs)
    else:
        return augs


def identity_aug(with_image_wrapper=False):
    augs = lambda image: {"image": image}
    if with_image_wrapper:
        return partial(albumentations_transform_wrapper, augment_fn=augs)
    else:
        return augs


def albumentations_transform_wrapper(image, augment_fn):
    augmentation = augment_fn(image=image.px)
    return augmentation['image']


def five_crop_tta_transform(img_tensor, crop_height, crop_width):
    assert len(img_tensor.shape) == 3
    width, height = img_tensor.shape[1:]
    top_left = img_tensor[:, 0:crop_height, 0:crop_width]
    top_right = img_tensor[:, 0:crop_height, width - crop_width:width]
    bottom_left = img_tensor[:, height - crop_height: height, 0: crop_width]
    bottom_right = img_tensor[:, height - crop_height: height, width - crop_width: width]
    height_margin = (height - crop_height) // 2
    width_margin = (width - crop_width) // 2
    center = img_tensor[:, height_margin: height - height_margin, width_margin: width - width_margin]
    return torch.stack([top_left, top_right, bottom_left, bottom_right, center])



augment_fn_lookup = {
    "very_simple_aug": very_simple_aug,
    "very_simple_aug_with_elastic_transform": very_simple_aug_with_elastic_transform,
    "very_simple_aug_with_elastic_transform_and_crop": very_simple_aug_with_elastic_transform_and_crop,
    "simple_aug": simple_aug,
    "simple_aug_lower_prob": simple_aug_lower_prob,
    "resize_aug": resize_aug,
    "identity_aug": identity_aug,
    "five_crop_tta_transform": five_crop_tta_transform
}
