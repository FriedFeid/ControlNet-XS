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

from light_util import get_perspective
from ldm.data.util import create_image_grid

from skimage import feature
import yaml



class LaionLightBase(Dataset):
    def __init__(self, size=None, random_resized_crop=False,# random_crop=False,
                 interpolation="bicubic", scale=[1.0, 1.0], control_mode='canny',
                 full_set=False, data_csv=None, data_root=None,
                 use_pillow=True, use_edges=True, use_render=True, use_distance=True,
                 use_depth_maps=False,
                 ):
        
    
        
        self.use_edges = use_edges
        self.use_render = use_render
        self.use_distance = use_distance
        self.use_depth_maps = use_depth_maps
        if data_csv is None:
            self.split = self.get_split()
            self.data_csv = {"train": "data/sun360_light_train_split.txt",
                            "validation": "data/sun360_light_train_split.txt"}[self.split]
        else: 
            self.data_csv = data_csv
            if not full_set:
                self.split = self.get_split()

        self.data_root = data_root or "/export/data/vislearn/rother_subgroup/datasets/SUN360/"
        self.use_pillow = use_pillow

        self.control_mode = control_mode
        if control_mode == 'canny':
            self.hint_model = CannyDetector()
            # self.hint_model = feature.canny
        # self.data_root = "tmp"

        self.full_set = full_set
        with open(self.data_csv, "r") as f:

            metadata = yaml.load(f, Loader=yaml.FullLoader)

            # metadata = json.load(f)
            # meta_keys = metadata.keys()

            idx_start, idx_end = 0, -1
            if not full_set:
                if self.split == 'train':
                    idx_start = 1000
                else:
                    idx_end = 1000
            self.captions = [metadata[key]['caption'] for key in metadata.keys()][idx_start:idx_end]
            self.rabbit_paths = [metadata[key]['rabbit_path'] for key in metadata.keys()][idx_start:idx_end]
            self.env_map_paths = [metadata[key]['environment_map'] for key in metadata.keys()][idx_start:idx_end]
            self.distance_map_paths = [metadata[key]['distance_map_path'] for key in metadata.keys()][idx_start:idx_end]
            self.angles = [metadata[key]['angle'] for key in metadata.keys()][idx_start:idx_end]
            self.fovs = [metadata[key]['fov'] for key in metadata.keys()][idx_start:idx_end]
            # self.light_parameters = [metadata[key]['light_parameters'] for key in metadata.keys()][idx_start:idx_end]
            self.edge_paths = [metadata[key]['edges_path'] for key in metadata.keys()]
            self.perspective_paths = [metadata[key]['perspective_path'] for key in metadata.keys()]
            self.depth_paths = [metadata[key]['depth_path'] for key in metadata.keys()]
            # if 'depth_path' in metadata[list(metadata.keys())[0]]:
            #     self.depth_paths = [metadata[key]['depth_path'] for key in metadata.keys()]
            # else:
            #     self.depth_paths = [None] * len(self.perspective_paths)

        
        

        self._length = len(self.env_map_paths)

        self.labels = {
            'relative_file_path_': [l for l in self.env_map_paths],
            'env_path': [os.path.join(self.data_root, l)
                           for l in self.env_map_paths],
            'rabbit_path': [os.path.join(self.data_root, l)
                            for l in self.rabbit_paths],
            'distance_map_path' : [os.path.join(self.data_root, l)
                             for l in self.distance_map_paths],
            'edge_path' : [os.path.join(self.data_root, l)
                             for l in self.edge_paths],
            'perspective_path' : [os.path.join(self.data_root, l)
                             for l in self.perspective_paths],
            'depth_path' : [os.path.join(self.data_root, l)
                             for l in self.depth_paths],
            'angle' : self.angles,
            # 'light_parameters' : self.light_parameters,
            'caption': self.captions,
            'fov' : self.fovs,
        }

        size = None if size is not None and size<=0 else size
        self.size = size
        self.random_resized_crop = random_resized_crop
        print('[NO RANDOM CROPPING IS PERFORMED FOR LIGHT CONTROL TRAINING]')

        
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
                    self.cropper = tt.CenterCrop(size=(self.size))
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
    
    def get_coord_grids(self, height, width):
        '''
        returns [HxW, HxW] coordinate tensor with coordinates
        '''
        # center initial grid (probably not necessary for the horizontal dimension)
        x_i = np.linspace(-(width // 2 + 1), width // 2 + 1, width // 2 * 2 + 1)[:width]
        y_i = np.linspace(-(height // 2 + 1), height // 2 + 1, height // 2 * 2 + 1)[:height]
        
        grid_y, grid_x = np.meshgrid(y_i, x_i, indexing='ij')
        return grid_y, grid_x
    
    def unwrap_image(self, image, fov, angle, height, width): # TODO
        image = get_perspective(
            image=image,
            FOV=fov,
            THETA=angle,
            PHI=0,
            height=height,
            width=width
        )
        return image
    
    def get_distance_map(self, ): # TODO
        distance_map = ...
        return distance_map
    
    def get_sun_map(self, ): # TODO
        sun_map = ...
        return sun_map

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example['perspective_path'])
        depthmap = Image.open(example['depth_path'])
        rabbit = tt.ToTensor()(Image.open(example['rabbit_path']))

        distance_map = Image.open(example['distance_map_path'])

        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)

        if not depthmap.mode == "RGB":
            depthmap = depthmap.convert("RGB")
        depthmap = np.array(depthmap).astype(np.uint8)

        if not distance_map.mode == "RGB":
            distance_map = distance_map.convert("RGB")
        distance_map = np.array(distance_map).astype(np.uint8)
    
        # align distance_map to the angle
        distance_map = get_perspective(
            image=distance_map,
            FOV=example['fov'],
            THETA=example['angle'],# + 180 % 360,
            PHI=0,
            height=self.size,
            width=self.size
        )


        image = tt.ToTensor()(image)
        depthmap = tt.ToTensor()(depthmap)
        edges = tt.ToTensor()(Image.open(example['edge_path']))
        

        # low_th = np.random.randint(50, 100)
        # high_th = np.random.randint(200, 350)
        # detected_map = self.hint_model(image, low_th, high_th) # original sized greyscale edges
        # image_edge = np.concatenate([image, detected_map[..., None]], axis=2)
        # image_edge = tt.ToTensor()(image_edge)

        # concat = np.array(einops.rearrange(concat, 'c h w -> h w c'))        
        
        if self.size is not None:
            # processed = self.image_rescaler(image_edge)
            image= self.image_rescaler(image)
            edges= self.image_rescaler(edges)
            depthmap = self.image_rescaler(depthmap)
            rabbit= self.image_rescaler(rabbit)
            # depthmap = self.image_rescaler(depthmap)
            
            image_edge = torch.cat([image, edges], dim=0)
            if self.use_depth_maps:
                image_edge = torch.cat([image_edge, depthmap], dim=0)

            processed = self.preprocessor(image_edge)
            rabbit = self.preprocessor(rabbit)
            # depthmap = self.preprocessor(depthmap)
        else:
            image_edge = torch.cat([image, edges], dim=0)
            if self.use_depth_maps:
                image_edge = torch.cat([image_edge, depthmap], dim=0)
            processed = image_edge
        processed = np.array(einops.rearrange(processed, 'c h w -> h w c'))
        # distance_map = np.array(einops.rearrange(distance_map, 'c h w -> h w c'))
        rabbit = np.array(einops.rearrange(rabbit, 'c h w -> h w c'))


        example["image"] = (processed[..., :3]*2 - 1.0).astype(np.float32)
        example['distance_map'] = distance_map
        example['rabbit'] = rabbit
        if self.use_depth_maps:
            example['depth_map'] = (processed[..., 4:]).astype(np.float32)

        # hint : [edges, rabbit, distance_map, distancemap] -> [H x W x 4]
        hints = []
        controls = 0
        grid = [example['image'] / 2. + 0.5]
        if self.use_edges:
            # hints.append(processed[..., 3:])
            hints.append(processed[..., 3:4].repeat(3, 2))
            grid.append(processed[..., 3:4].repeat(3, 2))
            controls += 1
        if self.use_render:
            # hints.append(rabbit.sum(axis=2)[..., None] / 3)
            hints.append(rabbit)
            grid.append(rabbit)
            controls += 1
        if self.use_distance:
            # hints.append(distance_map.astype(np.float32)[:, :, 0][..., None] / 255.)
            hints.append(distance_map.astype(np.float32) / 255.)
            grid.append(distance_map.astype(np.float32) / 255.)
            controls += 1
        if self.use_depth_maps:
            hints.append((processed[..., 4:]).astype(np.float32))
            grid.append((processed[..., 4:]).astype(np.float32))
            controls += 1
        
        
        hint = np.concatenate(hints, axis=2)
        example['hint'] = hint

        # hint = np.concatenate([
        #     processed[..., 3:],
        #     rabbit.sum(axis=2)[..., None] / 3,
        #     distance_map.astype(np.float32)[:, :, 0][..., None] / 255.,
        #     ], axis=2)

        
        
        example['control_grid'] = create_image_grid(np.stack(grid), (1, controls + 1))




        # example['distance_map'] = self.get_distance_map()
        # example['sun_map'] = self.get_sun_map()

        return example


class LaionTrain(LaionLightBase):
    # default to random_crop=True
    def __init__(self, size=None, random_resized_crop=False,# random_crop=True,
                interpolation="bicubic", scale=[1.0, 1.0], control_mode='canny',
                data_root=None, data_csv=None, use_edges=True, use_render=True, use_distance=True,
                use_depth_maps=False,
                ):
        super().__init__(
            size=size, interpolation=interpolation,
            scale=scale, random_resized_crop=random_resized_crop,
            control_mode=control_mode, data_root=data_root, data_csv=data_csv,
            use_edges=use_edges, use_render=use_render, use_distance=use_distance,
            use_depth_maps=use_depth_maps,
                          )

    def get_split(self):
        return "train"


class LaionValidation(LaionLightBase):
    def __init__(self, size=None,# random_crop=False,
                  interpolation="bicubic", control_mode='canny', data_root=None, data_csv=None,
                  use_edges=True, use_render=True, use_distance=True, use_depth_maps=False,
                  ):
        super().__init__(size=size, interpolation=interpolation,
                         control_mode=control_mode, data_root=data_root,
                         data_csv=data_csv, use_edges=use_edges, use_render=use_render,
                         use_distance=use_distance, use_depth_maps=use_depth_maps)

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
