import os
import numpy as np
import cv2
import csv 
import albumentations
from PIL import Image
from torch.utils.data import Dataset
import json

import torchvision.transforms as tt
import torch
import einops
import OpenEXR
import Imath 

# from annotator.canny import CannyDetector
# from annotator.util import HWC3

# from light_util import get_perspective

from ldm.data.util import create_image_grid
from skimage import feature
from glob import glob 
from natsort import natsorted
import yaml


class BlockWorld(Dataset):
    def __init__(self, size=None, random_resized_crop=False, 
                 interpolation='bicubic', scale=[1.0, 1.0], control=['Depth'], 
                 full_set=True, data_csv=False, data_root=None, 
                 use_pillow=True, use_edges=True, use_render=True, use_distance=True, 
                 use_depth_maps=False):
        
        self.use_edges = use_edges
        self.use_render = use_render
        self.use_distance = use_distance
        self.use_depth_maps = use_depth_maps

        if data_csv is None:
            # TODO: implement splitting dataset
            self.split = self.get_split()
            self.data_csv = {"train": "data/sun360_light_train_split.txt",
                            "validation": "data/sun360_light_train_split.txt"}[self.split]
        else: 
            self.data_csv = data_csv
            if not full_set:
                self.split = self.get_split()

        if data_root is None:
            self.data_root = '/export/data/vislearn/rother_subgroup/feiden/data/ControlNet/training/Generated_Data'
        else: 
            self.data_root = data_root

        self.use_pillow = use_pillow

        self.full_set = full_set

        self.Labels = self.__load_labels__()
        
        self.Render_folder = os.path.join(self.data_root, 'Render')
        self.Renders = natsorted(glob(os.path.join(self.Render_folder, 'Render_*.png') ))
        self.__len = len(self.Renders)
        
        self.hint_dic = {}
        for key in ['Depth', 'DisMap', 'Edges', 'SSphere', 'Env']:
            # Collecting all Folder Paths
            if key == 'Env':
                self.hint_dic[key+'_folder'] = os.path.join(self.data_root, 'Env_Maps')
            elif key == 'DisMap':
                self.hint_dic[key+'_folder'] = os.path.join(self.data_root, 'Distance_Map')
            elif key == 'SSphere':
                self.hint_dic[key+'_folder'] = os.path.join(self.data_root, 'SpikySphere')
            else: 
                self.hint_dic[key+'_folder'] = os.path.join(self.data_root, key)
            # Loading paths to Images
            if key == 'Env':
                self.hint_dic[key] = natsorted(glob(os.path.join(self.hint_dic[key+'_folder'], key + '_*.exr'))) 
            else: 
                self.hint_dic[key] = natsorted(glob(os.path.join(self.hint_dic[key+'_folder'], key + '_*.png')))
        
        self.consistant = self.__consistant__()

        self.control = control

        size = None if size is not None and size<=0 else size
        self.size = size

        if use_pillow:
            ########## PILLOW RESIZE VERSION ##########
            self.interpolation = interpolation
            self.interpolation = {
                    "nearest": tt.InterpolationMode.NEAREST,
                    "bilinear": tt.InterpolationMode.BILINEAR,
                    "bicubic": tt.InterpolationMode.BICUBIC,
                    "lanczos": tt.InterpolationMode.LANCZOS}[self.interpolation]
            self.env_rescaler = tt.Resize(
                    size=(256, 512),
                    interpolation=self.interpolation,
                    antialias=True
                    )
            if self.size is not None:
                
                self.image_rescaler = tt.Resize(
                    size=self.size,
                    interpolation=self.interpolation,
                    antialias=True
                    )
                new_env_size = self.size
                new_env_size = (int(new_env_size/2), int(new_env_size))
                self.env_rescaler = tt.Resize(
                    size=new_env_size,
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
            self.interpolation = interpolation
            self.interpolation = {
                    "nearest": cv2.INTER_NEAREST,
                    "bilinear": cv2.INTER_LINEAR,
                    "bicubic": cv2.INTER_CUBIC,
                    "area": cv2.INTER_AREA,
                    "lanczos": cv2.INTER_LANCZOS4}[self.interpolation]
            self.env_rescaler = albumentations.SmallestMaxSize(max_size=(256, 512),
                                                                    interpolation=self.interpolation)
            

            if self.size is not None:
                
                
                new_env_size = self.size
                new_env_size = (int(new_env_size/2), int(new_env_size))

                self.env_rescaler = albumentations.SmallestMaxSize(max_size=new_env_size,
                                                                    interpolation=self.interpolation)

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



    def __load_exr__(self, exr_path):
        '''
        loads exr data: 
        
        returns:
            sun: np.array (hight, width, channels)
        '''
        path = os.path.join(exr_path)
        exr = OpenEXR.InputFile(path)
        dw = exr.header()['dataWindow']
        size = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

        # Get correct data type
        if exr.header()['channels']['R'] == Imath.Channel(Imath.PixelType(Imath.PixelType.HALF)):
            dt = np.float16
        else:
            dt = np.float32

        # Fill numpy array
        arr = np.zeros((size[0], size[1], 3), dt)
        for i, c in enumerate(['R', 'G', 'B']):
            arr[:,:,i] = np.frombuffer(exr.channel(c), dt).reshape(size)
        
        exr = arr.astype(np.float32)
                
        return exr

    def __consistant__(self):
        '''
        Checks if the Dataset is self consistant

        returns:   
            consitant: bool 
        '''
        amount_data = len(self.Renders)
        consis = True 
        if amount_data != len(self.Labels):
            consis = False

        for key in ['Depth', 'DisMap', 'Edges', 'SSphere', 'Env']:
            if amount_data != len(self.hint_dic[key]):
                consis = False
        
        if not consis:
            print('WARNING: There are inconsitencys in the datase: ')
            print(f'Renders: {len(self.Renders)}')
            print(f'Labels: {len(self.Labels)}')
            for key in ['Depth', 'DisMap', 'Edges', 'SSphere', 'Env']:
                print(f'{key}: {len(self.hint_dic[key])}') 
            return False
        else: 
            return True
    
    def __load_labels__(self):
        '''
        Loads Labels cvs as list of dictionarys.
        '''
        param_dic = {}
        with open(os.path.join(self.data_root, 'Labels.csv'), newline='') as tab:
            reader = csv.DictReader(tab, delimiter=',', )
            for row in reader: 
                dic = {}
                for key in row: 
                    if key == 'wsun' or key == 'wsky':
                        vec = np.array([float(row[key].split(' ')[0]),
                                            float(row[key].split(' ')[1]),
                                            float(row[key].split(' ')[2]),])
                        dic[key] = vec
                    elif key == 'name' or key == 'Prompt':
                        dic[key] = row[key]
                    else:
                        dic[key] = float(row[key])
                param_dic[row['name']] = dic

        return param_dic
        
    def __len__(self):
        if self.consistant:
            return self.__len
        else:
            return len(self.Labels)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image_render = Image.open( self.Renders[idx] )
        if not image_render.mode == 'RGB':
            image_render = image_render.convert('RGB')
        image_render = np.array(image_render).astype(np.uint8)


        instance_ = {}
        instance_['render'] = tt.ToTensor()(image_render)
        for key in self.control: 
            if key == 'Env':
                img = torch.from_numpy(np.moveaxis(self.__load_exr__(self.hint_dic[key][idx]), -1, 0))
                
                img = self.env_rescaler(img)
                if self.size is not None:
                    full_img = torch.zeros((3, self.size, self.size))
                    img_shape = img.size()
                    full_img[:, :img_shape[1], :img_shape[2]] = img
                
                else: 
                    full_img = torch.zeros_like(instance_['render'])
                    full_img[:, 128:384, :] = img
                instance_[key] = full_img
            
            elif key == 'DisMap':
                img = Image.open(self.hint_dic[key][idx])
                if not img.mode == 'RGB':
                    img = img.convert('RGB')
                img = np.array(img).astype(np.uint8)
                img = tt.ToTensor()(img) 

                img = self.env_rescaler(img)
                if self.size is not None:
                    full_img = torch.zeros((3, self.size, self.size))
                    img_shape = img.size()
                    full_img[:, :img_shape[1], :img_shape[2]] = img
                
                else: 
                    full_img = torch.zeros_like(instance_['render'])
                    full_img[:, 128:384, :] = img
                instance_[key] = full_img

            else: 
                img = Image.open(self.hint_dic[key][idx])
                if not img.mode == 'RGB':
                    img = img.convert('RGB')
                img = np.array(img).astype(np.uint8)

                img = tt.ToTensor()(img)
            
                full_img = img
                instance_[key] = full_img

        if self.size is not None: 
            for key in instance_: 
                instance_[key] = self.image_rescaler(instance_[key])

        for key in instance_:
            instance_[key] = np.array(einops.rearrange(instance_[key], 'c h w -> h w c')).astype(np.float32)
        
        hints = []
        controls = 0
        grid = [instance_['render']]
        for key in self.control:
            if key == 'Env':
                hints.append(instance_[key])
                grid.append(instance_[key].clip(0, 1))
                controls +=1
            else:
                hints.append(instance_[key])
                grid.append(instance_[key])
                controls +=1
        
        example = {}
        example['hint'] = np.concatenate(hints, axis=2)
        example['image'] = instance_['render']/127.5 - 1.0
        
        keys = self.hint_dic['Env'][idx].split('.exr')[0]
        if '/' in keys:
            keys = keys.split('/')[-1]
        example['caption'] = self.Labels[keys]['Prompt']

        example['control_grid'] = create_image_grid(np.stack(grid), (1, controls + 1))

        return example
                

class BlockWorldTrain(BlockWorld):
    # default to random_crop=True
    def __init__(self, size=None, random_resized_crop=False, 
                 interpolation='bicubic', scale=[1.0, 1.0], control=['Depth'], 
                 full_set=True, data_csv=False, data_root=None, 
                 use_pillow=True, use_edges=True, use_render=True, use_distance=True, 
                 use_depth_maps=False
                ):
        super().__init__(
            size=size, interpolation=interpolation,
            scale=scale, random_resized_crop=random_resized_crop,
            control=control, data_root=data_root, data_csv=data_csv,
            use_edges=use_edges, use_render=use_render, use_distance=use_distance,
            use_depth_maps=use_depth_maps,
                          )

    def get_split(self):
        return "train"


class BlockWorldValidation(BlockWorld):
    def __init__(self, size=None, random_resized_crop=False, 
                 interpolation='bicubic', scale=[1.0, 1.0], control=['Depth'], 
                 full_set=True, data_csv=False, data_root=None, 
                 use_pillow=True, use_edges=True, use_render=True, use_distance=True, 
                 use_depth_maps=False
                  ):
        super().__init__(size=size, interpolation=interpolation,
                         control=control, data_root=data_root,
                         data_csv=data_csv, use_edges=use_edges, use_render=use_render,
                         use_distance=use_distance, use_depth_maps=use_depth_maps)

    def get_split(self):
        return "validation"

