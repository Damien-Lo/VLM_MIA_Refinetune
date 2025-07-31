import torch
import torchvisions.transforms as transfomrs
from functool import partial

def get_augmentations(cfg):
    aug_dict = cfg.data.augmentations
    aug_func_dict = dict()
    for key, values in aug_dict:
        if key == "RandomResize" and values.use == true:
            _aug_f_list = list()
            for _size in values.size:
                aug_f = transforms.RandomResizedCrop(
                    size=values.size,
                )
                _aug_f_list.append(aug_f)
            aug_dict[key] = _aug_f_list
        elif key == "RandomRotation" and values.use == true:
            _aug_f_list= list()
            for _degrees in values.degrees:
                aug_f = transforms.RandomRotation(
                    degrees=values.degrees
                )
                _aug_f_list.append(aug_f)
                aug_dict[key] = _aug_f_list
        elif key == "RandomAffine" and values.use == true:
            _aug_f_list = list()
            for _degree, _translate, _scale in zip(values.degrees, values.translate, values.scale):
                aug_f = transforms.RandomAffine(
                    degrees=_degree,
                    translate=_translate,
                    scale=_scale
                )
                _aug_f_list.append(aug_f)
                aug_dict[key] = aug_f
        elif key == "ColorJitter" and values.use == true:
            _aug_f_list = list()
            for _brightness, _contrast, _saturation, _hue in zip(values.brightness,
                                                                 values.contrast,
                                                                 values.saturation,
                                                                 values.hue):
                aug_f = transforms.ColorJitter(
                    brightness = values.brightness,
                    contrast = values.contrast,
                    saturation = values.saturation,
                    hue = values.hue
                )
                _aug_f_list.append(aug_f)
            aug_dict[key] = _aug_f_list
        elif key == "GaussianNoise" and values.use == true:
            for _mean, _std, _clip in zip(values.mean, values.std, values.clip):
                _aug_f_list = list()
                aug_f = transforms.AddGaussianNoisePIL(
                    mean=values.mean,
                    std=values.std,
                    clip=values.clip
                )
                _aug_f_list.append(aug_f)
            aug_dict[key] = _aug_f_list
            
    return aug_dict