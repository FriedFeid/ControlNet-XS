import numpy as np 
import matplotlib.pyplot as plt 
import os 
import torch 
from glob import glob 
import csv

import cv2

import PIL
import subprocess

from PIL import Image

import OpenEXR
import Imath
from time import time 
from natsort import natsorted
from tqdm import tqdm

from lm_model import LMSkyModel

from torch.utils.data import Dataset



def save_exr(path, data):
    """
    Args: 
        path: str 
        data: np.array (float32) (hight, width, channels)
    ---------------------------------------------
    Save data to an EXR file. For optimal results this should be an float32 array.
    """
    header = OpenEXR.Header(data.shape[1], data.shape[0])
    dt = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    header['channels'] = dict([(c, dt) for c in 'RGB'])
    exr = OpenEXR.OutputFile(path, header)
    R = (data[:,:,0]).astype(np.float32).tobytes()
    G = (data[:,:,1]).astype(np.float32).tobytes()
    B = (data[:,:,2]).astype(np.float32).tobytes()
    exr.writePixels({'R' : R, 'G' : G, 'B' : B})
    exr.close()

def load_exr(path):
    """
    Args: 
        path: str 
    ----------------------------------------------
    Load an exr file from path.
    --------------------------------------------
    returns: np.array (float32) (hight, width, channels)
    """
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
            
    return arr.astype(np.float32)

def load_renders(render_folder):
    '''
    Args: 
        render_folder: str (path to rendered images folder)
    -------------------------------------------------------
    loads predicted and true RGB renders of Bunnys. 
    The names have to include pred, if it is an predicted env. map 
                              true, if it is an true env. map
    ---------------------------------------------------------
    returns:    
        true (np.array) (num_imgs, hight, width, channels)
        pred (np.array) (num_imgs, hight, width, channels)
    '''

    pred_paths = natsorted(glob(os.path.join(render_folder, '*_Render.png')))

    
    # find out img resolution and number of images: 
    true_imgs = None
    
    # same for predicted images:
    if pred_paths:
        img = np.asarray(PIL.Image.open(pred_paths[0]), dtype=np.uintc)
        pred_imgs = np.zeros( (len(pred_paths), *img.shape))

        for i, pred_path in enumerate(pred_paths):
            pred_imgs[i] = np.asarray(PIL.Image.open(pred_path), dtype=np.uintc)
    
    else: 
        assert len(pred_paths) != 0, 'No predictions found! Please check if polder path is correct or if predictions have .exr format'

    return true_imgs, pred_imgs


def sphere_distance(azimuth, zenith, curr_azimuth, curr_zenith):
    '''
    Args:   
        azimuth: float 
        zenith: float 
        curr_azimuth: float 
        curr_zenith: float
    -----------------------------------------------
    Calculates the Haversine Distance on a sphere: https://en.wikipedia.org/wiki/Haversine_formula
    between two points, given by P1 (azimuth, zenith) and P2 (curr_azimuth, curr_zenith)
    ------------------------------------------------
    reutrns: 
        distance: float
    '''
    distance = 2 * np.arcsin( np.sqrt(
                                                np.sin( (zenith - curr_zenith)/2 )**2 + 
                                                np.cos(curr_zenith) * np.cos(zenith) * 
                                                np.sin( (azimuth - curr_azimuth )/2 )**2 
                                            )
                                )

    return distance

def calc_distance_map(param_dic, sun_model, threshold= 5.):
    '''
    Args: 
        param_dic: dict (dictionary of gt parameters)
        sun_model: np.array() hdr_sun_model
        threshold: float
    ------------------------------------------------
    returns an numpy grayscale array of the distance to the sun. 
    In order to keep it easy we just calculate the distance from the center of the sun 
    while setting all vlaues higher than the threshold to distnace 1.

    This means that there might be a discontinuety at the edge of the sun and 
    distance 
    -------------------------------------------------
    returns: 
        np.array (hight, width, channels) distances
    '''

    # calculate angles out of u, v 
    azimuth = (2 * np.pi * param_dic['sunpos_u']) - np.pi
    zenith = (np.pi * param_dic['sunpos_v']) - np.pi/2

    intensity = np.sqrt(np.sum(sun_model**2, axis=2))
    distance_map = np.ones_like(intensity)
    # distance_map[int(distance_map.shape[0]/2):, :] = 0.

    mask = intensity > threshold
    distance_map[mask] = 0.

    width = intensity.shape[1]
    hight = intensity.shape[0]


    # speed this up by using an array computation 
    width = np.arange(intensity.shape[1])
    hight = np.arange(intensity.shape[0])

    curr_width, curr_hight = np.meshgrid(width, hight)

    curr_azimuth = (2 * np.pi * curr_width/intensity.shape[1]) - np.pi
    curr_zenith = (np.pi * curr_hight/intensity.shape[0]) - np.pi/2

    azimuth = np.ones_like(curr_width).astype(float) * azimuth
    zenith = np.ones_like(curr_hight).astype(float) * zenith

    distance_map = sphere_distance(azimuth, zenith,
                                   curr_azimuth, curr_zenith)
    # for curr_width in range(width):
    #     for curr_hight in range(int(hight)):

    #         if distance_map[curr_hight, curr_width] != 0.:
    #             # normalise to u and v 
    #             curr_u = curr_width/width
    #             curr_v = curr_hight/hight

    #             # get angles
    #             curr_azimuth = (2 * np.pi * curr_u) - np.pi
    #             curr_zenith = (np.pi * curr_v) - np.pi/2

    #             distance_map[curr_hight, curr_width] = sphere_distance(azimuth, zenith, 
    #                                                                    curr_azimuth, curr_zenith)
    
    distance_map[mask] = 0. 

    return distance_map

def find_ellipse(sun_model, threshold = 5.):
    '''
    Args: 
        picture: np.array (Sun model)
        threshold: float
    -----------------------------------------------
    Should find the ellipse sorrounding the sun. 

    -----------------------------------------------
    returns: 
        ellipse 
    
    '''
    # Create contour out of HDR image 
    intensity = np.sqrt(np.sum(sun_model**2, axis=2))
    sun_mask = np.zeros_like(intensity)
    sun_mask[intensity > threshold] = 1. 

    # Create open-cv array
    thresh = sun_mask.astype(np.uint8)*255

    # Find contours; If we find two we should try to rotate the image, in order
    # to not split the sun in half at the edge of the image
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 1: 
        print('rotate until we have only one')
    
    # fit ellipse 
    ellipse = cv2.fitEllipse(contours[0])

    cv2.ellipse(thresh, ellipse, (0, 0, 255), 3)

    # cv2.imshow("Ellipse", thresh)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()




class SUN360(Dataset):
    '''
    Datalaoder which loads SUN360 panos in full resolution along with the 11 Parameters Ground Truth 
    of the LM Sky model as well as renders of Spiky_Sphere and SUN_model. 
    '''

    def __init__(self, root_SUN360_dir, root_GT_params, root_Render_dir, transform=None):
        '''
        Args: 
            root_SUN360_dir: str (Path to SUN360)
            root_GT_params: str (gt_params.csv)
            root_Render_dir: str (Path to Renders)
            transform: (callable, optimal) Transfomations which should be applied
        ---------------------------------------------------------------------------
        returns: 
            ldr_picture (np.array)
            11_params_dict['elev', 'wsun', 'kappa',
                           'beta', 't', 'wsky', 
                           'sunpos_u', 'sunpos_v', 
                           'name']
            Renders: dict ['fov*_ang*': np.array, ... ]
            SUN_model: (np.array)
        '''
        self.root_sun_dir = root_SUN360_dir 
        self.root_gt_params = root_GT_params
        self.root_render_dir = root_Render_dir

        self.ldr_paths = natsorted(glob(os.path.join(self.root_sun_dir, '*.jpg')))
        self.len_data = len(self.ldr_paths)

        self.param_dic = self.load_tabular()
        self.len_gt = len(self.param_dic)

        if self.len_data != self.len_gt: 
            print('WARNING: data or label missing')
        
        self.transforms = transform 

    def load_tabular(self):
        '''
        Args: 

        ---------------------------------------
        loads the csv data and converts it to an dictionary sorted as follows :
        param_dic['name'] = dict['elev', 'wsun', 'kappa',
                                 'beta', 't', 'wsky', 
                                 'sunpos_u', 'sunpos_v', 'name']
        '''
        param_dic = {}
        with open(self.root_gt_params, newline='') as tab:
            reader = csv.DictReader(tab, delimiter=',', )
            for row in reader: 
                dic = {}
                for key in row: 
                    if key == 'wsun' or key == 'wsky':
                        vec = np.array([float(row[key].split(' ')[0]),
                                            float(row[key].split(' ')[1]),
                                            float(row[key].split(' ')[2]),])
                        dic[key] = vec
                    elif key == 'name':
                        dic[key] = row[key]
                    else:
                        dic[key] = float(row[key])
                param_dic[row['name']] = dic
            
        return param_dic
    
    def __len__(self):
        return self.len_data
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # load ldr picture 

        ldr_path = self.ldr_paths[idx]

        ldr = np.asarray(Image.open(ldr_path), dtype=np.float32)/255.

        instance_name = ldr_path.split('/')[-1]
        instance_name = instance_name.split('.jpg')[0]

        # load Renders 
        render_paths = natsorted( glob(os.path.join(self.root_render_dir, self.param_dic[instance_name]['name']+
                                                                            '_fov*_ang*_SpikySphere.png')) )
        renders = {}
        for i in range(len(render_paths)):
            renders[render_paths[i].split('_')[5]+'_'+render_paths[i].split('_')[6]] = np.asarray(Image.open(render_paths[i]), dtype=np.float32)/255.

        # load SunModel 
        sun_model = load_exr(os.path.join(self.root_render_dir, self.param_dic[instance_name]['name']+
                                                                            '_sunModel.exr'))

        if self.transforms: 
            ldr = self.transforms(ldr)

        return ldr, self.param_dic[instance_name], renders, sun_model
    
SUN360_path = '/export/home/ffeiden/Dokumente/Datasets/SUN360/gt_pano/gt_panos/'
GT_params = '/export/home/ffeiden/Dokumente/11Params/sun360-label.csv'

Render_path = 'SUN360_Renders_no_offset/'

Params_loader = SUN360(SUN360_path, GT_params, Render_path)

lm_model = LMSkyModel(height = 128)

threshold = 5.

rot_angles = np.array([0, 55, 65, 60, 55, 65])
fov = np.array(      [50, 60,  70,  50,  60,  70])

for curr_inst in range(30):
    ldr_pic, param_dic, render_dic, sunmodel = Params_loader[curr_inst]

    # # do not think we need this 
    azimuth = (2 * np.pi * param_dic['sunpos_u']) - np.pi
    zenith = (np.pi * param_dic['sunpos_v'])# - np.pi/2

    sun = lm_model.get_sun(wsun=param_dic['wsun'], 
                           azimuth= azimuth, 
                           zenith = zenith,
                           beta = param_dic['beta'],
                           kappa = param_dic['kappa'], 
                           )
    
    sky = lm_model.get_sky(wsky = param_dic['wsky'],
                           azimuth= azimuth, 
                           zenith = zenith,
                           turbidity= param_dic['t'])
    
    hdr_pred = sun + sky

    # find_ellipse(sunmodel, threshold=threshold)

    fig = plt.figure(figsize=(24, 15))

    ax1 = plt.subplot2grid(shape=(4, 6), loc=(0, 0), colspan=3)
    ax1_1 = plt.subplot2grid(shape=(4, 6), loc=(0, 3), colspan=3)

    ax2_l = []
    for i in range(6):
        ax2_l.append(plt.subplot2grid(shape=(4, 6), loc=(1, i)) )

    ax3 = plt.subplot2grid(shape=(4, 6), loc=(2, 0), colspan=2)
    ax3_1 = plt.subplot2grid(shape=(4, 6), loc=(2, 2), colspan=2)
    ax3_2 = plt.subplot2grid(shape=(4, 6), loc=(2, 4), colspan=2)

    ax4 = plt.subplot2grid(shape=(4, 6), loc=(3, 0), colspan=2)
    ax4_1 = plt.subplot2grid(shape=(4, 6), loc=(3, 2), colspan=2)
    ax4_2 = plt.subplot2grid(shape=(4, 6), loc=(3, 4), colspan=2)


    ax1.imshow(ldr_pic)
    ax1.plot(ldr_pic.shape[1]* param_dic['sunpos_u'], ldr_pic.shape[0]*param_dic['sunpos_v'], marker='x', color='red')
    ax1.set_title('Org. SUN360 image')

    ax1_1.imshow(sunmodel.clip(0, 1))
    ax1_1.plot(sunmodel.shape[1]* param_dic['sunpos_u'], sunmodel.shape[0]*param_dic['sunpos_v'], marker='x', color='red')
    ax1_1.set_title('Sun model')

    angle = 0
    for i, fov_curr in enumerate(fov):
        angle += rot_angles[i]
        title = 'fov'+str(fov_curr)+'_ang'+str(angle)
        ax2_l[i].imshow(render_dic[title])
        ax2_l[i].set_title(title)

    # Create contour out of HDR image 
    intensity = np.sqrt(np.sum(sunmodel**2, axis=2))
    sun_mask = np.zeros_like(intensity)
    sun_mask[intensity > threshold] = 1. 

    thresh = sun_mask.astype(np.uint8)

    im = ax3.imshow(intensity)
    ax3.plot(intensity.shape[1]* param_dic['sunpos_u'], intensity.shape[0]*param_dic['sunpos_v'], marker='x', color='red')
    ax3.set_title('Sun Model Intensity')
    cb = plt.colorbar(im)

    ax3_1.imshow(thresh, cmap='gray')
    ax3_1.plot(thresh.shape[1]* param_dic['sunpos_u'], thresh.shape[0]*param_dic['sunpos_v'], marker='x', color='red')
    ax3_1.set_title('Sun Mask')

    dis_map = calc_distance_map(param_dic, sunmodel, threshold=threshold)
    # im = ax3_2.imshow(dis_map/np.max(dis_map))
    # cb = plt.colorbar(im)

    cs = ax3_2.contour(dis_map)
    ax3_2.plot(dis_map.shape[1]* param_dic['sunpos_u'], dis_map.shape[0]*param_dic['sunpos_v'], marker='x', color='red')
    ax3_2.invert_yaxis()
    ax3_2.clabel(cs, inline=True, fontsize=10)
    ax3_2.set_title('Contour map of distances')

    im = ax4.imshow(sun.clip(0, 1))
    ax4.plot(sun.shape[1]* param_dic['sunpos_u'], sun.shape[0]*param_dic['sunpos_v'], marker='x', color='red')
    ax4.set_title('Sun Model')

    ax4_1.imshow(sky.clip(0, 1))
    ax4_1.plot(sky.shape[1]* param_dic['sunpos_u'], sky.shape[0]*param_dic['sunpos_v'], marker='x', color='red')
    ax4_1.set_title('Sky Model')

    ax4_2.imshow(hdr_pred.clip(0, 1))
    ax4_2.plot(hdr_pred.shape[1]* param_dic['sunpos_u'], hdr_pred.shape[0]*param_dic['sunpos_v'], marker='x', color='red')
    ax4_2.set_title('HDR sky Prediction')




    plt.savefig(os.path.join('/export/home/ffeiden/Dokumente/Datasets/SUN360/Example_plots', 'SUN360_refined_img_{}'.format(curr_inst)))
    # plt.show()
    plt.close('all')
    # if curr_inst == 4: 
    #     break

# thresholds = np.array([1, 2, 3, 5, 10, 20, 40, 80, 100])

# for pic in range(30): 
#     fig, ax = plt.subplots(3, 3, figsize=(10, 5))
#     for i , threshold_ in enumerate(thresholds):
#         l, m = divmod(i, 3)

#         ldr_pic, param_dic, render_dic, sunmodel = Params_loader[pic]
        
#         intensity = np.sqrt(np.sum(sunmodel**2, axis=2))

#         sun_mask = np.zeros_like(intensity)
#         sun_mask[intensity > threshold_] = 1. 

#         thresh = sun_mask.astype(np.uint8)

#         im = ax[l][m].imshow(thresh)
#         ax[l][m].plot(thresh.shape[1]* param_dic['sunpos_u'], thresh.shape[0]*param_dic['sunpos_v'], marker='x', color='red')
#         ax[l][m].set_title('Sun mask; Threshold: {}'.format(threshold_))
#     fig.tight_layout()
#     plt.savefig(os.path.join('/export/home/ffeiden/Dokumente/Datasets/SUN360/Example_plots', 'Threshold_img_{}'.format(pic)))
#     plt.close('all')



    
