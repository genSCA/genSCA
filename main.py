import argparse
import os
import random
import time
import progressbar

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from vae_lp import VAE_LP
from ae_gan import AE_GAN
from data_loader import ImageDataset
from data_loader import DataLoader
import utils

args = utils.load_params(json_file='params.json')

data_loader = DataLoader(args)
vae = VAE_LP(args)
gan = AE_GAN(args)

train_dataset = ImageDataset(args, split='train')
test_dataset = ImageDataset(args, split='test')

train_loader = data_loader.get_loader(train_dataset)
test_loader = data_loader.get_loader(test_dataset)

for i in range(args['vae_epoch']):
    vae.train(train_loader)
    if i % args['test_freq'] == 0:
        vae.test(test_loader)
        vae.save_model('%s/ckpt/%03d.pth' % (args['vae_dir'], i))

gan.set_trace2image(vae.inference)

for i in range(args['gan_epoch']):
    gan.train(train_loader)
    if i % args['test_freq'] == 0:
        gan.test(test_loader)
        gan.save_model('%s/ckpt/%03d.pth' % (args['gan_dir'], i))