import numpy as np
import os
import torch.utils.data as data
import torch
import natsort as natsort
import torch.utils.data as data
from PIL import Image
import matplotlib.pyplot as plt
from utils import *
from albumentations import Compose
import cv2
import random
try:  # SciPy >= 0.19
    from scipy.special import comb
except ImportError:
    from scipy.misc import comb
import math

def normalize(array):
    
    array = (array - np.min(array)) / (np.max(array) - np.min(array))
    array = np.nan_to_num(array)
    
    return array

def normalize_z(array):
    
    mean = np.mean(array)
    std = np.std(array)
    array = (array - mean) / std
    
    return array

def padding(array, size=224):
    
    h, w = array.shape
    h_pad = int((size - h) / 2)
    w_pad = int((size - w) / 2)
    
    array_pad = np.pad(array, ((h_pad, h_pad), (w_pad, w_pad)), mode='constant', constant_values=0)
    return array_pad

def crop_image(flair_mri, t1_mri, t2_mri, label):
    
    coords = np.argwhere(flair_mri)
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    cropped_flair_mri = flair_mri[y_min:y_max+1, x_min:x_max+1]
    cropped_t1_mri = t1_mri[y_min:y_max+1, x_min:x_max+1]
    cropped_t2_mri = t2_mri[y_min:y_max+1, x_min:x_max+1]
    cropped_Rater = label[y_min:y_max+1, x_min:x_max+1]
    
    return cropped_flair_mri, cropped_t1_mri, cropped_t2_mri, cropped_Rater

def crop_image_only(img):
    
    coords = np.argwhere(img)
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    cropped_img = img[y_min:y_max+1, x_min:x_max+1]
    
    return cropped_img

class ToTensor:
    def __call__(self, mri):
        # mri, label = sample['image'], sample['label']
        if len(mri.shape) == 2:
            mri = np.expand_dims(mri, axis=-1)
        mri = mri.transpose((2, 0, 1))
        
        return torch.from_numpy(mri).float()
        
        # return {'image': torch.from_numpy(mri).float(),
        #         'label': torch.tensor(label).long()}

class ResizeTransform:
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        image = Image.fromarray((image * 255).astype(np.uint8))  # Convert to PIL Image
        image = transforms.Resize(self.size)(image)
        image = np.array(image).astype(np.float32) / 255.0  # Normalize back to 0-1
        return image

def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i

def bezier_curve(points, nTimes=1000):
    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals

def nonlinear_transformation(x, prob=0.5):
    if random.random() >= prob:
        return x
    points = [[0, 0], [random.random(), random.random()], [random.random(), random.random()], [1, 1]]
    xpoints = [p[0] for p in points]
    ypoints = [p[1] for p in points]

    xvals, yvals = bezier_curve(points, nTimes=100000)
    xvals, yvals = np.sort(xvals), np.sort(yvals)


    nonlinear_x = np.interp(x, xvals, yvals)

    return nonlinear_x

class Microbleed_png_dataset(data.Dataset):
    
    def __init__(self, img_path, transform=None, aug=False):
        
        self.img_path = img_path
        self.gt_path = img_path + '/gt'
        self.transform = transform
        self.aug = aug
        
        self.img_files = [file for file in natsort.natsorted(list_file(self.img_path, '.png'))]
        self.gt_files = [file for file in natsort.natsorted(list_file(self.gt_path, '.png'))]
    
    def __getitem__(self, index):
        
        img_3ch_path = os.path.join(self.img_path, self.img_files[index])
        label_path = os.path.join(self.gt_path, self.gt_files[index])
        
        x_img = cv2.imread(img_3ch_path)
        y_img = cv2.imread(label_path, 0)
        
        x_img = x_img / 255

        #stacked_result = np.stack((x_img, laplacian_result_normalized), axis=-1)  # Shape: (H, W, 2)

        y_img = convert_gt(y_img)
        #y_img = np.expand_dims(y_img, axis=0) # (1, 224, 224)

        if self.transform:
            augmentations = self.transform(image=x_img, mask=y_img)
            x_img = augmentations["image"]
            y_img = augmentations["mask"]
            
        if self.aug == True :
            x_img = nonlinear_transformation(x_img, 0.3)

        #blurred = cv2.GaussianBlur(x_img, (5, 5), sigmaX=1.0)
        #laplacian_result = cv2.Laplacian(x_img, cv2.CV_64F)
        #laplacian_result_normalized = cv2.normalize(laplacian_result, None, 0.0, 1.0, cv2.NORM_MINMAX)

        #stacked_result = np.concatenate((x_img, laplacian_result_normalized), axis=-1)

        x_img = x_img.transpose(2, 0, 1) # (3, 224, 224)
        y_img = np.expand_dims(y_img, axis=0) # (1, 224, 224)
        

        x_img = torch.from_numpy(x_img)
        y_img = torch.from_numpy(y_img)

        return x_img, y_img
    
    def __len__(self):
        return len(self.gt_files)

class Microbleed_png_dataset2(data.Dataset):
    
    def __init__(self, img_path):
        
        self.img_path = img_path
        
        self.img_files = [file for file in natsort.natsorted(list_file(self.img_path, '.png'))]
    
    def __getitem__(self, index):
        
        img_3ch_path = os.path.join(self.img_path, self.img_files[index])
        
        x_img = cv2.imread(img_3ch_path)
        
        x_img = x_img / 255

        x_img = x_img.transpose(2, 0, 1) # (3, 224, 224)
        
        x_img = torch.from_numpy(x_img)

        return x_img
    
    def __len__(self):
        return len(self.img_files)
