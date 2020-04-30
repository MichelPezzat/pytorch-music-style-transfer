import os
import time
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image

from data_loader import TestSet
from model import Discriminator, DomainClassifier, Generator
from utility import *
#from preprocess import FRAMES, SAMPLE_RATE, FFTSIZE
import random
from sklearn.preprocessing import LabelBinarizer
#from pyworld import decode_spectral_envelope, synthesize
import librosa
import ast


class Solver(object):
    """docstring for Solver."""
    def __init__(self, data_loader, config):
        
        self.config = config
        self.data_loader = data_loader
        # Model configurations.
        
        self.lambda_cycle = config.lambda_cycle
        self.lambda_cls = config.lambda_cls
        self.lambda_identity = config.lambda_identity
        self.sigma_d = config.sigma_d

        # Training configurations.
        self.data_dir = config.data_dir
        self.test_dir = config.test_dir
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.c_lr = config.c_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        

        # Test configurations.
        self.test_iters = config.test_iters
        self.trg_style = ast.literal_eval(config.trg_style)
        self.src_style = config.src_style

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.stls_enc = LabelBinarizer().fit(styles)
        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()
    
    def build_model(self):
        self.G = Generator()
        self.D = Discriminator()
        self.C = DomainClassifier()

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.c_optimizer = torch.optim.Adam(self.C.parameters(), self.c_lr,[self.beta1, self.beta2])
        
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')
        self.print_network(self.C, 'C')
            
        self.G.to(self.device)
        self.D.to(self.device)
        self.C.to(self.device)
    
    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr, c_lr):
        """Decay learning rates of the generator and discriminator and classifier."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr
        for param_group in self.c_optimizer.param_groups:
            param_group['lr'] = c_lr

    def train(self):
        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr
        c_lr = self.c_lr

        start_iters = 0
        if self.resume_iters:
            pass
        
        #norm = Normalizer()
        data_iter = iter(self.data_loader)

        print('Start training......')
        start_time = datetime.now()

        for i in range(start_iters, self.num_iters):
            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #
             # Fetch real images and labels.
            try:
                x_real, style_idx_org, label_org = next(data_iter)
            except:
                data_iter = iter(self.data_loader)
                x_real, style_idx_org, label_org = next(data_iter)  

            #generate gaussian noise for robustness improvement

            gaussian_noise = self.sigma_d*torch.randn(x_real.size())         

            # Generate target domain labels randomly.
            rand_idx = torch.randperm(label_org.size(0))
            label_trg = label_org[rand_idx]
            style_idx_trg = style_idx_org[rand_idx]
            
            x_real = x_real.to(self.device)           # Input images.
            label_org = label_org.to(self.device)     # Original domain one-hot labels.
            label_trg = label_trg.to(self.device)     # Target domain one-hot labels.
            style_idx_org = style_idx_org.to(self.device) # Original domain labels
            style_idx_trg = style_idx_trg.to(self.device) #Target domain labels

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #
            # Compute loss with real audio frame.
            CELoss = nn.CrossEntropyLoss()
            cls_real = self.C(x_real)
            cls_loss_real = CELoss(input=cls_real, target=style_idx_org)

            self.reset_grad()
            cls_loss_real.backward()
            self.c_optimizer.step()
             # Logging.
            loss = {}
            loss['C/C_loss'] = cls_loss_real.item()

            out_r = self.D(x_real + gaussian_noise, label_org)
            # Compute loss with fake audio frame.
            x_fake = self.G(x_real, label_trg)
            out_f = self.D(x_fake.detach() + gaussian_noise, label_trg)
            d_loss_t = F.mse_loss(input=out_f,target=torch.zeros_like(out_f, dtype=torch.float)) + \
                F.mse_loss(input=out_r, target=torch.ones_like(out_r, dtype=torch.float))
           
            out_cls = self.C(x_fake)
            d_loss_cls = CELoss(input=out_cls, target=speaker_idx_trg)

            # Compute loss for gradient penalty.
            #alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
            #x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            #out_src = self.D(x_hat, label_trg)
            #d_loss_gp = self.gradient_penalty(out_src, x_hat)

            d_loss = d_loss_t + self.lambda_cls * d_loss_cls 
                            # \
                            #+ 5*d_loss_gp

            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()


            # loss['D/d_loss_t'] = d_loss_t.item()
            # loss['D/loss_cls'] = d_loss_cls.item()
            # loss['D/D_gp'] = d_loss_gp.item()
            loss['D/D_loss'] = d_loss.item()

            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #        
            if (i+1) % self.n_critic == 0:
                # Original-to-target domain.
                x_fake = self.G(x_real, label_trg)
                g_out_src = self.D(x_fake + gaussian_noise, label_trg)
                g_loss_fake = F.mse_loss(input=g_out_src, target=torch.ones_like(g_out_src, dtype=torch.float))
                
                out_cls = self.C(x_real)
                g_loss_cls = CELoss(input=out_cls, target=speaker_idx_org)

                # Target-to-original domain.
                x_reconst = self.G(x_fake, label_org)
                g_loss_rec = F.l1_loss(x_reconst, x_real )

                # Original-to-Original domain(identity).
                x_fake_iden = self.G(x_real, label_org)
                id_loss = F.l1_loss(x_fake_iden, x_real )

                # Backward and optimize.
                g_loss = g_loss_fake + self.lambda_cycle * g_loss_rec +\
                 self.lambda_cls * g_loss_cls + self.lambda_identity * id_loss
                 
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/loss_cls'] = g_loss_cls.item()
                loss['G/loss_id'] = id_loss.item()
                loss['G/g_loss'] = g_loss.item()
            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #
            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = datetime.now() - start_time
                et = str(et)[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i+1)

            # Translate fixed images for debugging.
            if (i+1) % self.sample_step == 0:
                with torch.no_grad():
                    d, style = TestSet(self.test_dir).test_data()
                    
                    label_o = self.stls_enc.transform([style])[0]
                    label_o = np.asarray([label_o])

                    target = random.choice([x for x in styles if x != style])
                    label_t = self.stls_enc.transform([target])[0]
                    label_t = np.asarray([label_t])

                    for filename, content in d.items():
                        
                        
                        filename = filename.split('.')[0]
                            
                        one_seg = torch.FloatTensor(content).to(self.device)
                        one_seg = one_seg.view(1,one_seg.size(0), one_seg.size(1),one_seg.size(2))
                        l_t = torch.FloatTensor(label_t)
                        one_seg = one_seg.to(self.device)
                        l_t = l_t.to(self.device)


                        one_set_transfer = self.G(one_seg, l_t)

                        l_o = torch.FloatTensor(label_o)
                        l_o = l_o.to(self.device)
                        
                        one_set_cycle = self.G(one_set_transfer, l_o).data.cpu().numpy()
                        one_set_transfer = one_set_return.data.cpu().numpy()



                        one_set_return_binary = to_binary(one_set_transfer,0.5)
                        one_set_cycle_binary = to_binary(one_set_cycle,0.5)
                        

                        name_origin = f'{style}-{target}_iter{i+1}_{filename}_origin'
                        name_transfer = f'{style}-{target}_iter{i+1}_{filename}_transfer'
                        name_cycle = f'{style}-{target}_iter{i+1}_{filename}_cycle'

                        path_origin = os.path.join(self.sample_dir, name_origin)
                        path_transfer = os.path.join(self.sample_dir, name_transfer)
                        path_cycle = os.path.join(self.sample_dir, name_cycle)

                        print(f'[save]:{path_origin},{path_transfer},{path_cycle}')
                        
                        save_midis(one_seg,path_origin)
                        save_midis(one_set_transfer,path_transfer)
                        save_midis(one_set_cycle,path_cycle)
                        
            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                C_path = os.path.join(self.model_save_dir, '{}-C.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                torch.save(self.C.state_dict(), C_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                c_lr -= (self.c_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr, c_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()
        self.c_optimizer.zero_grad()

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        C_path = os.path.join(self.model_save_dir, '{}-C.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
        self.C.load_state_dict(torch.load(C_path, map_location=lambda storage, loc: storage))

    

    def test(self):
        """Translate music using StarGAN ."""
        # Load the trained generator.
        self.restore_model(self.test_iters)
        

        # Set data loader.
        d, style = TestSet(self.test_dir).test_data(self.src_style)
        targets = self.trg_style

        label_o = self.stls_enc.transform([style])[0]
        label_o = np.asarray([label_o])
       
        for target in targets:
            print(target)
            assert target in styles
            label_t = self.stls_enc.transform([target])[0]
            label_t = np.asarray([label_t])
            
            with torch.no_grad():

                for filename, content in d.items():

                    filename = filename.split('.')[0]
                    

                            
                    one_seg = torch.FloatTensor(content).to(self.device)
                    one_seg = one_seg.view(1,one_seg.size(0), one_seg.size(1),one_seg.size(2))
                    l_t = torch.FloatTensor(label_t)
                    one_seg = one_seg.to(self.device)
                    l_t = l_t.to(self.device)


                    one_set_transfer = self.G(one_seg, l_t)

                    l_o = torch.FloatTensor(label_o)
                    l_o = l_o.to(self.device)
                        
                    one_set_cycle = self.G(one_set_transfer, l_o).data.cpu().numpy()
                    one_set_transfer = one_set_return.data.cpu().numpy()



                    one_set_return_binary = to_binary(one_set_transfer,0.5)
                    one_set_cycle_binary = to_binary(one_set_cycle,0.5)
                        

                    name_origin = f'{style}-{target}_iter{i+1}_{filename}_origin'
                    name_transfer = f'{style}-{target}_iter{i+1}_{filename}_transfer'
                    name_cycle = f'{style}-{target}_iter{i+1}_{filename}_cycle'

                    path_origin = os.path.join(self.sample_dir, name_origin)
                    path_transfer = os.path.join(self.sample_dir, name_transfer)
                    path_cycle = os.path.join(self.sample_dir, name_cycle)

                    print(f'[save]:{path_origin},{path_transfer},{path_cycle}')
                        
                    save_midis(one_seg,path_origin)
                    save_midis(one_set_transfer,path_transfer)
                    save_midis(one_set_cycle,path_cycle)
                    

                   


    

if __name__ == '__main__':
    pass
