from torch.utils.data import Dataset
from torchvision import transforms


import random
import os
import numpy as np
import glob
import PIL
from tqdm import tqdm 


def rgb2int(arr):
    """
    Convert (N,...M,3)-array of dtype uint8 to a (N,...,M)-array of dtype int32
    """
    return arr[...,0]*(256**2)+arr[...,1]*256+arr[...,2]

def rgb2vals(color, color2ind):
   
    int_colors = rgb2int(color)
    int_keys = rgb2int(np.array(list(color2ind.keys()), dtype='uint8'))
    int_array = np.r_[int_colors.ravel(), int_keys]
    uniq, index = np.unique(int_array, return_inverse=True)
    color_labels = index[:int_colors.size]
    key_labels = index[-len(color2ind):]

    colormap = np.empty_like(int_keys, dtype='int32')
    colormap[key_labels] = list(color2ind.values())
    out = colormap[color_labels].reshape(color.shape[:2])

    return out


class TASDataset(Dataset):
    def __init__(self, data_folder, eval=False, mode=None, augment_data=False):
        self.data_folder = data_folder
        self.eval = eval
        self.mode = mode
        self.augment_data = augment_data

        # You can use any valid transformations here
        # added augment_data flag, which is false by default, if true adds the following transformations.
        if augment_data:
            self.augment = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomApply(transforms=[transforms.RandomRotation(degrees=(0,30),fill=0)], p=0.3),
                # transforms.RandomApply(transforms=transforms.Compose([transforms.RandomCrop(size=(120,60)), transforms.Resize((768,384))]),p=0.3),                
                # transforms.RandomRotation(degrees=(0,30),fill=255),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomVerticalFlip(p=0.3),
            ])
            self.transform = transforms.Compose([transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                            ])
        else:
            self.transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                                ])                        

        # we will use the following width and height to resize
        self.width = 768
        self.height = 384

        self.color2class = {
                #terrain
                (192,192,192): 0, (105,105,105): 0, (160, 82, 45):0, (244,164, 96): 0, \
                #vegatation
                ( 60,179,113): 1, (34,139, 34): 1, ( 154,205, 50): 1, ( 0,128,  0): 1, (0,100,  0):1, ( 0,250,154):1, (139, 69, 19): 1,\
                #construction
                (1, 51, 73):2, ( 190,153,153): 2, ( 0,132,111): 2,\
                #vehicle
                (0,  0,142):3, ( 0, 60,100):3, \
                #sky
                (135,206,250):4,\
                #object
                ( 128,  0,128): 5, (153,153,153):5, (255,255,  0 ):5, \
                #human
                (220, 20, 60):6, \
                #animal
                ( 255,182,193):7,\
                #void
                (220,220,220):8, \
                #undefined
                (0,  0,  0):9
        }

        self.input_folder = os.path.join(self.data_folder, 'train')
        self.label_folder = os.path.join(self.data_folder, 'train_labels')

        if self.eval:
            self.input_folder = os.path.join(self.data_folder, 'val')
            self.label_folder = os.path.join(self.data_folder, 'val_labels')
        
        image_names = os.listdir(self.input_folder)
        
        invalid_labels = ['1537962190852671077.png','1539600515553691119.png', '1539600738459369245.png','1539600829359771415.png','1567611260762673589.png']
            
        if self.mode == 'test':
            image_names = sorted(list(set(image_names).difference(set(invalid_labels))))
        else:
            image_names = list(set(image_names).difference(set(invalid_labels)))
            
        self.paths = [(os.path.join(self.input_folder, i), os.path.join(self.label_folder, i)) for i in image_names]
        
        if self.mode == 'val': # use first 50 images for validation
            self.paths = self.paths[:50]
            
        elif self.mode == 'test': # use last 50 images for test
            self.paths = self.paths[50:]

    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):        
            
        image = np.asarray(PIL.Image.open(self.paths[idx][0]).resize((self.width, self.height)))
        mask_image = np.asarray(PIL.Image.open(self.paths[idx][1]).resize((self.width, self.height), PIL.Image.NEAREST))
        mask =  rgb2vals(mask_image, self.color2class)

        # Since we are using flipping and rotations, we need to transform the binary mask as well.
        if self.augment_data:
            image = self.augment(image)
            mask = self.augment(mask)
            mask = mask.reshape((mask.shape[1],mask.shape[2]))

        if self.transform:
            image = self.transform(image).float()

        return image, mask