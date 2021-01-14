import argparse
import os
import random
import time
import progressbar

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from models import TraceEncoder, ImageEncoder, ImageDecoder
from models import Generator as G
from models import Discriminator as D
import utils

class AE_GAN(object):
    def __init__(self, args):
        self.args = args
        self.epoch = 0
        self.bce = nn.BCELoss().cuda()
        self.real_label = torch.FloatTensor(1).cuda().fill_(1)
        self.fake_label = torch.FloatTensor(1).cuda().fill_(0)
        self.init_model_optimizer()

    def loss(self, output, label):
        return self.bce(output, label.expand_as(output))

    def init_model_optimizer(self):
        print('Initializing Model & Optimizer...')
        self.G = G(nc=self.args['nc'], dim=self.args['gan_dim'])
        self.G = torch.nn.DataParallel(self.G).cuda()
        
        self.optimizerG = torch.optim.Adam(
                            self.G.module.parameters(),
                            lr=self.args['gan_lr'],
                            betas=(self.args['beta1'], 0.999)
                            )

        self.D = D(nc=self.args['nc'], dim=self.args['gan_dim'])
        self.D = torch.nn.DataParallel(self.D).cuda()
        
        self.optimizerD = torch.optim.Adam(
                            self.D.module.parameters(),
                            lr=self.args['gan_lr'],
                            betas=(self.args['beta1'], 0.999)
                            )

    def set_trace2image(self, func):
        self.trace2image = func

    def load_model(self, path):
        print('Loading Model from %s ...' % (path))
        ckpt = torch.load(path)
        self.G.module.load_state_dict(checkpoint['Generator'])
        self.D.module.load_state_dict(checkpoint['Discriminator'])

    def save_model(self, path):
        print('Saving Model on %s ...' % (path))
        state = {
            'Generator': self.G.module.state_dict(),
            'Discriminator': self.D.module.state_dict()
        }
        torch.save(state, path)

    def train(self, data_loader):
        print('Training...')
        with torch.autograd.set_detect_anomaly(True):
            self.epoch += 1
            self.G.train()
            self.D.train()
            record_G = utils.Record()
            record_D = utils.Record()
            start_time = time.time()
            progress = progressbar.ProgressBar(maxval=len(data_loader)).start()
            for i, (trace, image) in enumerate(data_loader):
                progress.update(i + 1)
                trace = trace.cuda()
                image = image.cuda()
                
                self.D.zero_grad()
                # update D with real images
                real_output = self.D(image)
                err_D_real = self.loss(real_output, self.real_label)
                D_x = real_output.data.mean()
                # update D with reconstructed images
                fake_input, *_ = self.trace2image(trace)
                fake_refine = self.G(fake_input)
                fake_output = self.D(fake_refine.detach())
                err_D_fake = self.loss(fake_output, self.fake_label)
                D_G_z = fake_output.data.mean()

                err_D = err_D_fake + err_D_real
                err_D.backward()
                self.optimizerD.step()

                self.G.zero_grad()
                # update G
                fake_output = self.D(fake_refine)
                err_G = self.loss(fake_output, self.real_label)
                
                err_G.backward()
                self.optimizerG.step()

                record_D.add(err_D.item())
                record_G.add(err_G.item())
            progress.finish()
            utils.clear_progressbar()
            print('----------------------------------------')
            print('Epoch: %d' % self.epoch)
            print('Costs time: %.2f s' % (time.time() - start_time))
            print('Loss of G: %f' % (record_G.mean()))
            print('Loss of D: %f' % (record_D.mean()))
            print('D(x): %f, D(G(z)): %f' % (D_x, D_G_z))
            print('----------------------------------------')
            utils.save_image(image.data, ('%s/image/test/target_%03d.jpg' % (self.args['gan_dir'], self.epoch)))
            utils.save_image(trace2image.data, ('%s/image/test/tr2im_%03d.jpg' % (self.args['gan_dir'], self.epoch)))
            utils.save_image(image2image.data, ('%s/image/test/im2im_%03d.jpg' % (self.args['gan_dir'], self.epoch)))

    def test(self, data_loader):
        print('Testing...')
        with torch.no_grad():
            self.G.eval()
            self.D.eval()
            record_G = utils.Record()
            record_D = utils.Record()
            start_time = time.time()
            progress = progressbar.ProgressBar(maxval=len(data_loader)).start()
            for i, (trace, image) in enumerate(data_loader):
                progress.update(i + 1)
                trace = trace.cuda()
                image = image.cuda()
                
                real_output = self.D(image)
                err_D_real = self.loss(real_output, self.real_label)
                D_x = real_output.data.mean()

                fake_input, *_ = self.trace2image(trace)
                fake_refine = self.G(fake_input)
                fake_output = self.D(fake_refine.detach())
                err_D_fake = self.loss(fake_output, self.fake_label)
                D_G_z = fake_output.data.mean()

                err_D = err_D_fake + err_D_real

                fake_output = self.D(fake_refine)
                err_G = self.loss(fake_output, self.real_label)

                record_D.add(err_D.item())
                record_G.add(err_G.item())
            progress.finish()
            utils.clear_progressbar()
            print('----------------------------------------')
            print('Test at Epoch %d' % self.epoch)
            print('Costs time: %.2f s' % (time.time() - start_time))
            print('Loss of G: %f' % (record_G.mean()))
            print('Loss of D: %f' % (record_D.mean()))
            print('D(x): %f, D(G(z)): %f' % (D_x, D_G_z))
            print('----------------------------------------')
            utils.save_image(image.data, ('%s/image/test/target_%03d.jpg' % (self.args['gan_dir'], self.epoch)))
            utils.save_image(trace2image.data, ('%s/image/test/tr2im_%03d.jpg' % (self.args['gan_dir'], self.epoch)))
            utils.save_image(image2image.data, ('%s/image/test/im2im_%03d.jpg' % (self.args['gan_dir'], self.epoch)))

    def inference(self, trace):
        with torch.no_grad():
            self.G.eval()
            recov_image = self.trace2image(trace)
            final_image = self.G(recov_image)
        return final_image