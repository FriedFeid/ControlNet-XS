import os
import numpy as np
import cv2
import albumentations
from PIL import Image
from torch.utils.data import Dataset
import json
from ldm.data.util import resize_image_pil

import torchvision.transforms as tt
import torch
import einops

from annotator.canny import CannyDetector
from annotator.midas import MidasDetector
from annotator.util import HWC3

from skimage import feature



class LaionBase(Dataset):
    def __init__(self, size=None, random_resized_crop=False,# random_crop=False,
                 interpolation="bicubic", scale=[1.0, 1.0], control_mode='canny',
                 full_set=False, data_csv=None, caption_csv=None, data_root=None,
                 use_pillow=True, np_format=True,
                 original_size_as_tuple=False, crop_coords_top_left=False, target_size_as_tuple=False,
                 ):
        
    
        assert control_mode in ('canny', 'midas', 'image', 'image_norm', 'colour_blocks'), f'[<control_mode> {control_mode} not implemented]'

        self.np_format = np_format
        self.original_size_as_tuple = original_size_as_tuple
        self.crop_coords_top_left = crop_coords_top_left
        self.target_size_as_tuple = target_size_as_tuple

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

        self.control_mode = control_mode
        if control_mode == 'canny':
            self.hint_model = CannyDetector()
        elif control_mode == 'midas':
            self.hint_model = MidasDetector()
        else:# control_mode == 'image':
            self.hint_model = None
            # self.hint_model = feature.canny
        # self.data_root = "tmp"

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
            if type(captions[list(captions.keys())[0]]) == list:
                new_image_paths = []
                self.captions = []
                image_format = self.image_paths[0].split('.')[-1]
                for key in captions.keys():
                    self.captions += captions[key]
                    new_image_paths += len(captions[key]) * [key + '.' + image_format]
                self.image_paths = new_image_paths
            else:
                self.captions = [captions[img.split('.')[0]] for img in self.image_paths]
        

        self._length = len(self.image_paths)
        
        # class labels
        # self.synsets = [p.split("/")[0] for p in self.image_paths]
        # unique_synsets = np.unique(self.synsets)
        # print(f"{self.__class__.__name__}: unique class labels: {unique_synsets}")
        # class_dict = dict((synset, i) for i, synset in enumerate(unique_synsets))
        # self.class_labels = [class_dict[s] for s in self.synsets]
        # self.num_classes = len(unique_synsets)
        # /class labels

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

    def get_colour_hint(self, image, blocks=10):
        if not isinstance(image, torch.Tensor):
            image = tt.ToTensor()(image)
        c, h, w = image.shape
        if self.size is None:
            size = (h, w)
        elif isinstance(self.size, int):
            size = (self.size, self.size)
        else: size = self.size
            
        kernel_size = min(h, w) // blocks
        pool = torch.nn.AvgPool2d(kernel_size, stride=kernel_size)
        image = pool(image)
        
        image = tt.Resize(size, interpolation=tt.InterpolationMode.NEAREST)(image)
        
        return image
    
    def get_coord_grids(self, height, width):
        '''
        returns [HxW, HxW] coordinate tensor with coordinates
        '''
        # center initial grid (probably not necessary for the horizontal dimension)
        x_i = np.linspace(-(width // 2 + 1), width // 2 + 1, width // 2 * 2 + 1)[:width]
        y_i = np.linspace(-(height // 2 + 1), height // 2 + 1, height // 2 * 2 + 1)[:height]
        
        grid_y, grid_x = np.meshgrid(y_i, x_i, indexing='ij')
        return grid_y, grid_x

    def __getitem__(self, i, low_th=None, high_th=None):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example["file_path_"])
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        



        if self.control_mode == 'canny':
            low_th = low_th or np.random.randint(50, 100)
            high_th = high_th or np.random.randint(200, 350)
            detected_map = self.hint_model(image, low_th, high_th) # original sized greyscale edges
        elif self.control_mode == 'midas':
            max_resolution = 1024
            image = resize_image_pil(image, resolution=min(min(*image.shape[:-1]), max_resolution))
            detected_map, _ = self.hint_model(image)
        elif self.control_mode in ('image', 'image_norm', 'colour_blocks'):
            detected_map = image

        image_hint = np.concatenate([
                        image,
                        detected_map if self.control_mode in ('image', 'image_norm', 'colour_blocks') else detected_map[..., None]
                            ], axis=2)
        image_hint = tt.ToTensor()(image_hint)

        # concat = np.array(einops.rearrange(concat, 'c h w -> h w c'))        
        
        if not self.random_resized_crop:
            if self.size is not None:
                image_hint = self.image_rescaler(image_hint)
        orig_size = image_hint.shape[-2:]

        if self.size is not None:
            processed = self.preprocessor(image_hint)
        else:
            processed = image_hint
        target_size = processed.shape[-2:]

        if self.control_mode == 'colour_blocks':
            n_blocks = torch.randint(low=5, high=16, size=[1]).item()
            colour_blocks = self.get_colour_hint(processed[ :3, ...], blocks=n_blocks)*2 - 1.0
        processed = np.array(einops.rearrange(processed, 'c h w -> h w c'))


        example["image"] = (processed[..., :3]*2 - 1.0).astype(np.float32)
        if self.control_mode == 'image':
            example['hint'] = processed[..., 3:] * 255
        elif self.control_mode == 'image_norm':
            example['hint'] = (processed[..., :3]*2 - 1.0).astype(np.float32)
        elif self.control_mode == 'canny':
            example['hint'] = processed[..., 3:].repeat(3, 2)
        elif self.control_mode == 'midas':
            example['hint'] = processed[..., 3:].repeat(3, 2) * 2. - 1.
        elif self.control_mode == 'colour_blocks':
            example['hint'] = np.array(einops.rearrange(colour_blocks, 'c h w -> h w c')).astype(np.float32)
        else: 
            raise NotImplementedError()
        
        if self.crop_coords_top_left:
            example['crop_coords_top_left'] = torch.tensor([0, 0])
        if self.original_size_as_tuple:
            example['original_size_as_tuple'] = torch.tensor(orig_size)
        if self.target_size_as_tuple:
            example['target_size_as_tuple'] = torch.tensor(target_size)


        ############ Canny edge detection with cv
        # low_th = np.random.randint(50, 100)
        # high_th = np.random.randint(200, 350)
        # detected_map_canny = self.hint_model(processed["image"], low_th, high_th)
        # example['hint'] = HWC3(detected_map_canny) / 255.0

        ############ Canny edge detection with skimage
        # high_th = np.random.uniform(1,.5)
        # low_th = np.random.uniform(0.1, min(.25, high_th))

        # img_gray = np.array(tt.Grayscale(1)(torch.tensor(example["image"]).permute(2,0,1))[0])
        # detected_map_canny = self.hint_model(img_gray, sigma=1., low_threshold=low_th, high_threshold=high_th)
        # detected_map_canny = detected_map_canny[:, :, None].repeat(3, 2)

        # example['hint'] = np.array(detected_map_canny).astype(np.float32)

        # example['grid_y'] = processed['grid_y']
        # example['grid_x'] = processed['grid_x']
        # print(f'WITCHER GRID SHAPE \t{example["grid_y"].shape}')
        if not self.np_format:
            example['image'] = einops.rearrange(example['image'], 'h w c -> c h w')
            example['hint'] = einops.rearrange(example['hint'], 'h w c -> c h w')

        return example


class LaionTrain(LaionBase):
    # default to random_crop=True
    def __init__(self, size=None, random_resized_crop=False, np_format=True,# random_crop=True,
                 interpolation="bicubic", scale=[1.0, 1.0], control_mode='canny', data_root=None,
                 original_size_as_tuple=False, crop_coords_top_left=False, target_size_as_tuple=False,
                 ):
        super().__init__(size=size, interpolation=interpolation,
                          scale=scale, random_resized_crop=random_resized_crop,
                          control_mode=control_mode, data_root=data_root, np_format=np_format,
                          original_size_as_tuple=original_size_as_tuple, crop_coords_top_left=crop_coords_top_left,
                          target_size_as_tuple=target_size_as_tuple
                          )

    def get_split(self):
        return "train"


class LaionValidation(LaionBase):
    def __init__(self, size=None, np_format=True,# random_crop=False,
                  interpolation="bicubic", control_mode='canny', data_root=None,
                  original_size_as_tuple=False, crop_coords_top_left=False, target_size_as_tuple=False,
                  ):
        super().__init__(size=size, interpolation=interpolation,
                         control_mode=control_mode, data_root=data_root, np_format=np_format,
                         original_size_as_tuple=original_size_as_tuple, crop_coords_top_left=crop_coords_top_left,
                         target_size_as_tuple=target_size_as_tuple
                         )

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
