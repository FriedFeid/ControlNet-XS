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

from annotator.canny import CannyDetector
from annotator.util import HWC3

# from light_util import get_perspective
from ldm.data.util import create_image_grid

from skimage import feature
from glob import glob 
from natsort import natsorted
import yaml


class BlockWorld(Dataset):
    def __init__(self, size=None, random_resized_crop=False, 
                 interpolation='bicubic', scale=[1.0, 1.0], control=['Depth'], 
                 full_set=False, data_csv=None, data_root=None, 
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

        self.data_root = data_root or '/export/data/vislearn/rother_subgroup/feiden/data/ControlNet/training/Generated_Data'
        self.use_pillow = use_pillow

        self.full_set = full_set

        self.Labels = self.__load_labels__()
        
        self.Render_folder = os.path.join(self.root_folder, 'Render')
        self.Renders = natsorted(glob(os.path.join(self.Render_folder, 'Render_*.png') ))
        self.__len = len(self.Renders)
        
        self.hint_dic = {}
        for key in ['Depth', 'DisMap', 'Edges', 'SSphere', 'Env']:
            # Collecting all Folder Paths
            if key == 'Env':
                self.hint_dic[key+'_folder'] = os.path.join(self.root_folder, 'Env_Maps')
            elif key == 'DisMap':
                self.hint_dic[key+'_folder'] = os.path.join(self.root_folder, 'Distance_Map')
            elif key == 'SSphere':
                self.hint_dic[key+'_folder'] = os.path.join(self.root_folder, 'SpikySphere')
            else: 
                self.hint_dic[key+'_folder'] = os.path.join(self.root_folder, key)
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
        with open(os.path.join(self.root_folder, 'Labels.csv'), newline='') as tab:
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
        return self.__len
    
    