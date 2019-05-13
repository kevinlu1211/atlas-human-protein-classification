from functools import partial

import torch
import torchvision
from torchvision.transforms import FiveCrop, ToTensor, Lambda
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose, ElasticTransform,
    Resize, Crop, RandomBrightnessContrast, RandomCrop, Cutout
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


def crop_rotate90_flip(height=None, width=None, crop_height=None, crop_width=None, with_image_wrapper=False):
    augs = [
        RandomCrop(crop_height, crop_width),
        RandomRotate90(),
        Flip(),
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


def crop_rotate90_flip_brightness_elastic(height=None, width=None,
                                          crop_height=None, crop_width=None, brightness_transform_p=0.5, elastic_transform_p=0.5,
                                          with_image_wrapper=False):
    augs = [
        RandomCrop(crop_height, crop_width),
        RandomRotate90(),
        Flip(),
        # RandomBrightnessContrast(brightness_limit=3, contrast_limit=0.5, p=brightness_transform_p),
        RandomBrightnessContrast(brightness_limit=3, contrast_limit=1, p=brightness_transform_p),
        ElasticTransform(sigma=50, alpha_affine=50, p=elastic_transform_p),
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


def random_crop_flip_shift_scale_rotate_brightness_contrast(height=None, width=None,
                                                   crop_height=None, crop_width=None, with_image_wrapper=False):
    augs = [
        RandomCrop(crop_height, crop_width),
        Flip(),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=1),
        RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=1),
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

def random_crop_flip_shift_scale_rotate_brightness_contrast_cutout(crop_height, crop_width, cutout_height, cutout_width,
                                                                   height=None, width=None, with_image_wrapper=False):
    augs = [
        RandomCrop(crop_height, crop_width),
        Flip(),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=1),
        RandomBrightnessContrast(brightness_limit=3, contrast_limit=0.5, p=1),
        Cutout(num_holes=8, max_h_size=cutout_height, max_w_size=cutout_width)
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
    "crop_rotate90_flip": crop_rotate90_flip,
    "very_simple_aug_with_elastic_transform": very_simple_aug_with_elastic_transform,
    "very_simple_aug_with_elastic_transform_and_crop": crop_rotate90_flip_brightness_elastic,
    "random_crop_flip_shift_scale_rotate_brightness_contrast": random_crop_flip_shift_scale_rotate_brightness_contrast,
    "random_crop_flip_shift_scale_rotate_brightness_contrast_cutout": random_crop_flip_shift_scale_rotate_brightness_contrast_cutout,
    "crop_rotate90_flip_brightness_elastic": crop_rotate90_flip_brightness_elastic,
    "simple_aug": simple_aug,
    "simple_aug_lower_prob": simple_aug_lower_prob,
    "resize_aug": resize_aug,
    "identity_aug": identity_aug,
    "five_crop_tta_transform": five_crop_tta_transform
}
