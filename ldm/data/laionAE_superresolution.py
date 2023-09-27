import os
import numpy as np
import cv2
import albumentations
from PIL import Image
from torch.utils.data import Dataset
import json

import torchvision.transforms as tt
import torch
import einops

from annotator.canny import CannyDetector
from annotator.util import HWC3

from skimage import feature



class LaionSuperRes(Dataset):
    def __init__(self, size=None, random_resized_crop=False,# random_crop=False,
                 interpolation="bicubic", scale=[1.0, 1.0], control_size_factor=4,
                 full_set=False, data_csv=None, caption_csv=None, data_root=None,
                 use_pillow=True,
                 ):
        
    
        
        if data_csv is None:
            self.split = self.get_split()
            self.data_csv = {"train": "data/laion2B_en_aesthetic_train_split.txt",
                            "validation": "data/laion2B_en_aesthetic_train_split.txt"}[self.split]
        else: self.data_csv = data_csv
        if caption_csv is None:
            self.split = self.get_split()
            self.caption_csv = {"train": "data/laion2B_en_aesthetic_train_split_captions.json",
                            "validation": "data/laion2B_en_aesthetic_train_split_captions.json"}[self.split]
        else: self.caption_csv = caption_csv

        self.data_root = data_root or "data/laionAE_subset250"
        self.use_pillow = use_pillow

        self.control_size_factor = control_size_factor
        
        self.full_set = full_set
        with open(self.data_csv, "r") as f:
            self.image_paths = f.read().splitlines()
        
        if not full_set:
            if self.split == 'train':
                self.image_paths = self.image_paths[1000:]
            else:
                self.image_paths = self.image_paths[:1000]
        
        with open(self.caption_csv, "r") as f:
            captions = json.load(f)
            self.captions = [captions[img.split('.')[0]] for img in self.image_paths]
        

        self._length = len(self.image_paths)

        self.labels = {
            # "class_label": self.class_labels,
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [os.path.join(self.data_root, l)
                           for l in self.image_paths],
            # "segmentation_path_": [os.path.join(self.segmentation_root, l.replace(".jpg", ".png"))
            #                        for l in self.image_paths],
            "caption": self.captions
        }

        size = None if size is not None and size<=0 else size
        self.size = size
        self.random_resized_crop = random_resized_crop

        
        if use_pillow:
            ########## PILLOW RESIZE VERSION ##########
            if self.size is not None:
                self.interpolation = interpolation
                self.interpolation = {
                    "nearest": tt.InterpolationMode.NEAREST,
                    "bilinear": tt.InterpolationMode.BILINEAR,
                    "bicubic": tt.InterpolationMode.BICUBIC,
                    "lanczos": tt.InterpolationMode.LANCZOS}[self.interpolation]
                self.image_rescaler = tt.Resize(
                    size=self.size,
                    interpolation=self.interpolation,
                    antialias=True
                    )

                self.center_crop = not random_resized_crop
                if self.center_crop:
                    self.cropper = tt.CenterCrop(size=self.size)
                elif random_resized_crop:
                    self.cropper = tt.RandomResizedCrop(
                        size=self.size, scale=scale, ratio=[1.0, 1.0],
                        interpolation=self.interpolation, antialias=True
                        )
                else:
                    self.cropper = tt.RandomCrop(size=self.size)
                self.preprocessor = self.cropper
            
            self.hint_model = tt.Compose([
                tt.Resize(
                    size=self.size // self.control_size_factor,
                    interpolation=self.interpolation,
                    antialias=True
                    ),
                tt.Resize(
                    size=self.size,
                    interpolation=self.interpolation,
                    antialias=True
                    )
            ])
        else:
            ########## OPEN-CV RESIZE VERSION ##########
            if self.size is not None:
                self.interpolation = interpolation
                self.interpolation = {
                    "nearest": cv2.INTER_NEAREST,
                    "bilinear": cv2.INTER_LINEAR,
                    "bicubic": cv2.INTER_CUBIC,
                    "area": cv2.INTER_AREA,
                    "lanczos": cv2.INTER_LANCZOS4}[self.interpolation]
                self.image_rescaler = albumentations.SmallestMaxSize(max_size=self.size,
                                                                    interpolation=self.interpolation)
                # self.segmentation_rescaler = albumentations.SmallestMaxSize(max_size=self.size,
                #                                                             interpolation=cv2.INTER_NEAREST)
                # self.center_crop = not (random_crop or random_resized_crop)
                self.center_crop = not random_resized_crop
                if self.center_crop:
                    self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
                elif random_resized_crop:
                    self.cropper = albumentations.RandomResizedCrop(
                        height=self.size, width=self.size, scale=scale, ratio=[1.0, 1.0], interpolation=self.interpolation, p=1.0)
                else:
                    self.cropper = albumentations.RandomCrop(height=self.size, width=self.size)
                self.preprocessor = self.cropper

                



    def __len__(self):
        return self._length
    
    def get_coord_grids(self, height, width):
        '''
        returns [HxW, HxW] coordinate tensor with coordinates
        '''
        # center initial grid (probably not necessary for the horizontal dimension)
        x_i = np.linspace(-(width // 2 + 1), width // 2 + 1, width // 2 * 2 + 1)[:width]
        y_i = np.linspace(-(height // 2 + 1), height // 2 + 1, height // 2 * 2 + 1)[:height]
        
        grid_y, grid_x = np.meshgrid(y_i, x_i, indexing='ij')
        return grid_y, grid_x

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example["file_path_"])
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)




        # low_th = np.random.randint(50, 100)
        # high_th = np.random.randint(200, 350)
        # detected_map = self.hint_model(image, low_th, high_th) # original sized greyscale edges
        # image_edge = np.concatenate([image, detected_map[..., None]], axis=2)
        image = tt.ToTensor()(image)

        # concat = np.array(einops.rearrange(concat, 'c h w -> h w c'))        
        
        if not self.random_resized_crop:
            if self.size is not None:
                image = self.image_rescaler(image)

        if self.size is not None:
            processed = self.preprocessor(image)
        else:
            processed = image
        
        example['hint'] = (np.array(einops.rearrange(self.hint_model(processed), 'c h w -> h w c')) * 2 - 1.0).astype(np.float32)
        processed = np.array(einops.rearrange(processed, 'c h w -> h w c'))

        example["image"] = (processed*2 - 1.0).astype(np.float32)

        return example


class LaionTrain(LaionSuperRes):
    # default to random_crop=True
    def __init__(self, size=None, random_resized_crop=False,# random_crop=True,
                 interpolation="bicubic", scale=[1.0, 1.0], control_size_factor=4, data_root=None):
        super().__init__(size=size, interpolation=interpolation,
                          scale=scale, random_resized_crop=random_resized_crop,
                          control_size_factor=control_size_factor, data_root=data_root)

    def get_split(self):
        return "train"


class LaionValidation(LaionSuperRes ):
    def __init__(self, size=None,# random_crop=False,
                  interpolation="bicubic", control_size_factor=4, data_root=None):
        super().__init__(size=size, interpolation=interpolation,
                         control_size_factor=control_size_factor, data_root=data_root)

    def get_split(self):
        return "validation"


if __name__ == "__main__":
    dset = LaionTrain(size=256)
    ex = dset[0]
    # dset = LaionValidation(size=256)
    # ex = dset[0]
    for k in ["image", "caption"
              ]:
        print(type(ex[k]))
        try:
            print(ex[k].shape)
        except:
            print(ex[k])
