import os
import numpy as np
from PIL import Image
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import torchvision
import torchvision.transforms as transforms

import utils

def side_to_bound(side):
    if side == 'cacheline':
        v_max = 0xFFFF_FFFF >> 6
        v_min = 0
    elif side == 'pagetable':
        v_max = 0xFFFF_FFFF >> 12 # i.e., addr & (~4095)
        v_min = 0
    else:
        print('Side Channel %s is NOT Defined!' % side)
        return None, None
    return v_max, v_min

class ImageDataset(Dataset):
    def __init__(self, args, split):
        super(ImageDataset).__init__()
        self.args = args
        self.npz_dir = args['trace_dir'] + ('train/' if split == 'train' else 'test/')
        self.img_dir = args['image_dir'] + ('train/' if split == 'train' else 'test/')

        self.npz_list = sorted(os.listdir(self.npz_dir))
        self.img_list = sorted(os.listdir(self.img_dir))

        self.transform = transforms.Compose([
                       transforms.Resize(args['image_size']),
                       transforms.CenterCrop(args['image_size']),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
               ])

        self.v_max, self.v_min = side_to_bound(args['side_channel'])
        
        print('Total %d Data Points.' % len(self.npz_list))

    def __len__(self):
        return len(self.npz_list)

    def __getitem__(self, index):
        npz_name = self.npz_list[index]
        img_name = '-'.join(npz_name.split('-')[1:]).split('.')[0] + '.jpg'

        npz = np.load(self.npz_dir + npz_name)
        trace = npz['arr_0']
        trace = trace.astype(np.float32)
        #trace = torch.from_numpy(trace).view([3, 512, 512]) # [K, N, N]
        trace = torch.from_numpy(trace).view([
                                    self.args['trace_size_K'],
                                    self.args['trace_size_N'],
                                    self.args['trace_size_N']
                                ])
        ###############################################################
        # The length of side channel trace may be slightly different, #
        # set the value of (K, N, N) by yourself. Actually this value #
        # nearly has no effect on the final result.                   #
        ###############################################################
        trace = utils.norm_scale(v=trace,
                            v_max=self.v_max,
                            v_min=self.v_min
                            )

        image = Image.open(self.img_dir + img_name)
        image = self.transform(image)
        
        return trace, image


class DataLoader(object):
    def __init__(self, args):
        self.args = args
        self.init_param()

    def init_param(self):
        self.gpus = torch.cuda.device_count()

    def get_loader(self, dataset):
        data_loader = torch.utils.data.DataLoader(
                            dataset, 
                            batch_size=self.args['batch_size'] * self.gpus,
                            num_workers=int(self.args['num_workers']),
                            shuffle=True
                        )
        return data_loader

if __name__ == '__main__':
    pass