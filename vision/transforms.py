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


def very_simple_aug_with_elastic_transform(p=1, height=None, width=None):
    augs = [
        RandomRotate90(p=0.25),
        Flip(p=0.25),
        RandomBrightnessContrast(brightness_limit=3, contrast_limit=0.5, p=0.25),
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


def very_simple_aug_with_elastic_transform_and_crop(height, width):
    augs = [
        RandomRotate90(),
        Flip(),
        RandomBrightnessContrast(brightness_limit=3, contrast_limit=0.5),
        ElasticTransform(sigma=50, alpha_affine=50, p=0.5),
        RandomCrop(height, width)
    ]
    return Compose(augs, p=1)


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


def resize_aug(p=1, height=None, width=None):
    return Compose([Resize(height=height, width=width)], p=p)


def albumentations_transform_wrapper(image, augment_fn):
    augmentation = augment_fn(image=image.px)
    return augmentation['image']


augment_fn_lookup = {
    "very_simple_aug": very_simple_aug,
    "very_simple_aug_with_elastic_transform": very_simple_aug_with_elastic_transform,
    "very_simple_aug_with_elastic_transform_and_crop": very_simple_aug_with_elastic_transform_and_crop,
    "simple_aug": simple_aug,
    "simple_aug_lower_prob": simple_aug_lower_prob,
    "resize_aug": resize_aug,
}
