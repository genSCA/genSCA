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

class VAE_LP(object):
    def __init__(self, args):
        self.args = args
        self.epoch = 0
        self.l1 = nn.L1Loss().cuda()
        self.l2 = nn.MSELoss().cuda()
        self.kld = utils.kl_criterion
        self.init_model_optimizer()

    def init_model_optimizer(self):
        print('Initializing Model & Optimizer...')
        self.TraceEncoder = TraceEncoder(nc=self.args['trace_size_K'], dim=self.args['vae_dim'])
        self.TraceEncoder = torch.nn.DataParallel(self.TraceEncoder).cuda()
        
        self.ImageEncoder = ImageEncoder(nc=self.args['nc'], dim=self.args['vae_dim'])
        self.ImageEncoder = torch.nn.DataParallel(self.ImageEncoder).cuda()
        
        self.Decoder = ImageDecoder(nc=self.args['nc'], dim=self.args['vae_dim'])
        self.Decoder = torch.nn.DataParallel(self.Decoder).cuda()
        
        self.optimizer = torch.optim.Adam(
                            list(self.TraceEncoder.module.parameters()) + \
                            list(self.ImageEncoder.module.parameters()) + \
                            list(self.Decoder.module.parameters()),
                            lr=self.args['vae_lr'],
                            betas=(self.args['beta1'], 0.999)
                            )

    def load_model(self, path):
        print('Loading Model from %s ...' % (path))
        ckpt = torch.load(path)
        self.TraceEncoder.module.load_state_dict(checkpoint['TraceEncoder'])
        self.ImageEncoder.module.load_state_dict(checkpoint['ImageEncoder'])
        self.Decoder.module.load_state_dict(checkpoint['Decoder'])

    def save_model(self, path):
        print('Saving Model on %s ...' % (path))
        state = {
            'TraceEncoder': self.TraceEncoder.module.state_dict(),
            'ImageEncoder': self.ImageEncoder.module.state_dict(),
            'Decoder': self.Decoder.module.state_dict()
        }
        torch.save(state, path)

    def set_train(self):
        self.TraceEncoder.train()
        self.ImageEncoder.train()
        self.Decoder.train()

    def set_eval(self):
        self.TraceEncoder.eval()
        self.ImageEncoder.eval()
        self.Decoder.eval()

    def zero_grad(self):
        self.TraceEncoder.zero_grad()
        self.ImageEncoder.zero_grad()
        self.Decoder.zero_grad()

    def train(self, data_loader):
        print('Training...')
        with torch.autograd.set_detect_anomaly(True):
            self.epoch += 1
            self.set_train()
            record_trace = utils.Record()
            record_image = utils.Record()
            record_inter = utils.Record()
            record_kld = utils.Record()
            start_time = time.time()
            progress = progressbar.ProgressBar(maxval=len(data_loader)).start()
            for i, (trace, image) in enumerate(data_loader):
                progress.update(i + 1)
                trace = trace.cuda()
                image = image.cuda()
                self.zero_grad()
                trace_embed = self.TraceEncoder(trace)
                image_embed = self.ImageEncoder(image)
                trace_mu, trace_logvar = trace_embed, trace_embed
                image_mu, image_logvar = image_embed, image_embed
                trace_z = utils.reparameterize(trace_mu, trace_logvar)
                image_z = utils.reparameterize(image_mu, image_logvar)
                trace2image, trace_inter = self.Decoder(trace_z)
                image2image, image_inter = self.Decoder(image_z)

                err_trace = self.l1(trace2image, image)
                err_image = self.l1(image2image, image)
                #err_inter = self.l2(trace_inter, image_inter)
                err_kld = self.kld(image_mu, image_logvar, trace_mu, trace_logvar)

                #(err_trace + err_image + err_inter + self.args['beta'] * err_kld).backward()
                (err_trace + err_image + self.args['beta'] * err_kld).backward()
                
                self.optimizer.step()

                record_trace.add(err_trace)
                record_image.add(err_image)
                #record_inter.add(err_inter)
                record_kld.add(err_kld)
            progress.finish()
            utils.clear_progressbar()
            print('----------------------------------------')
            print('Epoch: %d' % self.epoch)
            print('Costs time: %.2fs' % (time.time() - start_time))
            print('Loss of Trace to Image: %f' % (record_trace.mean()))
            print('Loss of Image to Image: %f' % (record_image.mean()))
            print('Loss of KL-Divergence: %f' % (record_kld.mean()))
            print('----------------------------------------')
            utils.save_image(image.data, ('%s/image/train/target_%03d.jpg' % (self.args['vae_dir'], self.epoch)))
            utils.save_image(trace2image.data, ('%s/image/train/tr2im_%03d.jpg' % (self.args['vae_dir'], self.epoch)))
            utils.save_image(image2image.data, ('%s/image/train/im2im_%03d.jpg' % (self.args['vae_dir'], self.epoch)))

    def test(self, data_loader):
        print('Testing...')
        with torch.no_grad():
            self.set_eval()
            record_trace = utils.Record()
            record_image = utils.Record()
            record_inter = utils.Record()
            record_kld = utils.Record()
            start_time = time.time()
            progress = progressbar.ProgressBar(maxval=len(data_loader)).start()
            for i, (trace, image) in enumerate(data_loader):
                progress.update(i + 1)
                trace = trace.cuda()
                image = image.cuda()

                trace_embed = self.TraceEncoder(trace)
                image_embed = self.ImageEncoder(image)
                trace_mu, trace_logvar = trace_embed, trace_embed
                image_mu, image_logvar = image_embed, image_embed
                trace_z = utils.reparameterize(trace_mu, trace_logvar)
                image_z = utils.reparameterize(image_mu, image_logvar)
                trace2image, trace_inter = self.Decoder(trace_z)
                image2image, image_inter = self.Decoder(image_z)

                err_trace = self.l1(trace2image, image)
                err_image = self.l1(image2image, image)
                #err_inter = self.l2(trace_inter, image_inter)
                err_kld = self.kld(image_mu, image_logvar, trace_mu, trace_logvar)

                record_trace.add(err_trace)
                record_image.add(err_image)
                #record_inter.add(err_inter)
                record_kld.add(err_kld)
            progress.finish()
            utils.clear_progressbar()
            print('----------------------------------------')
            print('Test at Epoch %d' % self.epoch)
            print('Costs time: %.2fs' % (time.time() - start_time))
            print('Loss of Trace to Image: %f' % (record_trace.mean()))
            print('Loss of Image to Image: %f' % (record_image.mean()))
            print('Loss of KL-Divergence: %f' % (record_kld.mean()))
            print('----------------------------------------')
            utils.save_image(image.data, ('%s/image/test/target_%03d.jpg' % (self.args['vae_dir'], self.epoch)))
            utils.save_image(trace2image.data, ('%s/image/test/tr2im_%03d.jpg' % (self.args['vae_dir'], self.epoch)))
            utils.save_image(image2image.data, ('%s/image/test/im2im_%03d.jpg' % (self.args['vae_dir'], self.epoch)))

    def inference(self, x):
        with torch.no_grad():
            self.TraceEncoder.eval()
            self.Decoder.eval()
            trace_embed = self.TraceEncoder(x)
            recov_image = self.Decoder(trace_embed)
        return recov_image