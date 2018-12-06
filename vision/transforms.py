from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose, ElasticTransform,
    Resize, Crop, RandomBrightnessContrast
)


def simple_aug(p=1, height=None, width=None):
    augs = [
        RandomRotate90(),
        Flip(),
        ShiftScaleRotate(shift_limit=0.05, scale_limit=0.2, rotate_limit=90),
        RandomBrightnessContrast(brightness_limit=3, contrast_limit=0.5),
        Blur(blur_limit=5, p=0.33),
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

def resize_aug(p=1, height=None, width=None):
    return Compose([Resize(height=height, width=width)], p=p)

