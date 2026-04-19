import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.adam import Adam
from torchvision.models.vgg import vgg19
import numpy as np
import cv2

from model import basic_model as model
from importlib import import_module

from model.networks import Discriminator_model
from model.gan import Patch_Discriminator
# from fvcore.nn import FlopCountAnalysis, flop_count_table

class SRModel(nn.Module):
    def __init__(self, args, train=True):
        super(SRModel, self).__init__()

        self.args = args
        self.scale = args.scale
        self.gpu = args.gpu
        self.error_last = 1e8

        self.phase = args.phase
        self.training_type = args.training_type
        print('sr model : ', self.args.sr_model)
        print('training type : ', self.training_type)

        # define model, optimizer, scheduler, loss
        self.gen = model.Model(args)
        # print(self.gen)
        # self.gen.eval()
        # flops = FlopCountAnalysis(self.gen, torch.rand(1, 3, 224, 224).to(self.gpu))
        # print(flop_count_table(flops))

        if args.pretrain_sr is not None:
            checkpoint = torch.load(args.pretrain_sr, map_location=lambda storage, loc: storage)['gen']
            self.gen.load_state_dict(checkpoint)
            print('Load pretrained SR model from {}'.format(args.pretrain_sr))

        if train:
            self.gen_opt = Adam(self.gen.parameters(), lr=(args.lr_sr), betas=(0.9, 0.999))
            self.gen_sch = torch.optim.lr_scheduler.StepLR(self.gen_opt, args.decay_batch_size_sr, gamma=0.5)  # args.gamma)

            self.content_criterion = nn.L1Loss().to(self.gpu)

            if self.training_type == 'endosr':
                self.dis = Discriminator_model(scale=int(args.scale)).to(self.gpu)
                self.dis_opt = Adam(self.dis.parameters(), lr=args.lr_sr, betas=(0.9, 0.999))
                self.dis_sch = torch.optim.lr_scheduler.StepLR(self.dis_opt, args.decay_batch_size_sr, gamma=0.1)

                self.GAN_Loss = GANLoss().to(self.gpu)
                # self.Wavelet_Loss = DWT_LOSS().to(self.gpu)
                self.Edge_Loss = EdgeLoss().to(self.gpu)
                self.TV_Loss = BTVLoss(1e-2).to(self.gpu)
                self.dis.train()
            elif self.training_type == 'gan':
                self.dis = Patch_Discriminator(in_nc=3, base_nf=64).to(self.gpu)
                self.dis_opt = Adam(self.dis.parameters(), lr = (args.lr_sr), betas=(0.9,0.999))
                self.dis_sch = torch.optim.lr_scheduler.StepLR(self.dis_opt, args.decay_batch_size_sr, gamma=0.1)
                self.GAN_Loss = GANLoss1(gan_type='lsgan').to(self.gpu)
                # self.Perceptual_loss = PerceptualLoss().to(self.gpu)
                # self.TV_Loss = TotalVariationLoss().to(self.gpu)
                self.dis.train()

            self.gen.train()

        self.gen_loss = 0
        if self.training_type == 'endosr':
            self.recon_loss = 0
        elif self.training_type == 'gan':
            self.dis_loss = 0

    ## update images to the model
    def update_img(self, lr, hr=None, lr_t=None):
        self.img_lr = lr
        self.img_hr = hr
        if lr_t is not None:
            self.img_t = lr_t.to(self.gpu).detach()

        self.gen_loss = 0
        if self.training_type == 'endosr':
            self.recon_loss = 0
        elif self.training_type == 'gan':
            self.dis_loss = 0

    def generate_HR(self):
        # self.img_lr *= 255
        if self.training_type == 'endosr' and self.phase == 'train':
            self.feature_s, self.img_gen = self.gen(self.img_lr, 0)
            self.feature_t, _ = self.gen(self.img_t, 0)
        elif self.training_type == 'endosr' and self.phase == 'test':
            self.feature_s, self.img_gen = self.gen(self.img_lr, 0)
        elif self.training_type == 'gan' and self.phase == 'train':
            self.img_gen = self.gen(self.img_lr, 0)
        elif self.training_type == 'gan' and self.phase == 'test':
            self.img_gen = self.gen(self.img_lr, 0)
        else:
            self.img_gen = self.gen(self.img_lr, 0)
        # self.img_gen /= 255

    def update_G(self):
        # EDSR style
        if self.training_type == 'edsr':
            self.gen_opt.zero_grad()

            self.recon_loss = self.content_criterion(self.img_gen, self.img_hr) * 255.0  # compensate range of 0 to 1

            self.recon_loss.backward()
            self.gen_opt.step()
            self.gen_loss = self.recon_loss.item()
            self.gen_sch.step()

        elif self.training_type == 'endosr':
            if self.args.cycle_recon:
                raise NotImplementedError('Do not support using cycle reconstruction loss in EndoSR training')

            # training generator
            self.gen_opt.zero_grad()

            score_fake = self.dis(self.feature_s)
            score_real = self.dis(self.feature_t)

            adversarial_loss_rf = self.GAN_Loss(score_fake, 'mean')
            adversarial_loss_fr = self.GAN_Loss(score_real, 'mean')
            adversarial_loss = adversarial_loss_fr + adversarial_loss_rf

            content_loss = self.content_criterion(self.img_gen, self.img_hr)  # compensate range of 0 to 1
            # wavelet_loss = self.Wavelet_Loss(self.img_gen, self.img_hr)
            tv_loss = self.TV_Loss(self.img_gen)
            edge_loss = self.Edge_Loss(self.img_hr, self.img_gen)


            gen_loss = adversarial_loss * self.args.adv_w + tv_loss + content_loss * self.args.con_w + self.args.edge_loss_w * edge_loss

            gen_loss.backward()
            self.gen_loss = gen_loss.item()
            self.edge_loss = self.args.edge_loss_w * (edge_loss.item())
            self.tv_loss = tv_loss.item()
            self.gen_opt.step()

            # training discriminator
            self.dis_opt.zero_grad()

            score_real = self.dis(self.feature_s.detach())
            score_fake = self.dis(self.feature_t.detach())

            adversarial_loss_rf = self.GAN_Loss(score_real, 'real')
            adversarial_loss_fr = self.GAN_Loss(score_fake, 'fake')
            discriminator_loss = adversarial_loss_fr + adversarial_loss_rf

            discriminator_loss.backward()
            self.dis_opt.step()

            self.gen_sch.step()
            self.dis_sch.step()
        # gan style
        elif self.training_type == 'gan':

            # training generator 
            self.gen_opt.zero_grad()

            self.img_dis_hr = self.dis(self.img_hr)
            self.img_dis_sr = self.dis(self.img_gen)

            content_loss = self.content_criterion(self.img_gen, self.img_hr)  # compensate range of 0 to 1
            # tv_loss = self.TV_Loss(self.img_gen)
            # perceptual_loss = self.Perceptual_loss(self.img_hr,self.img_gen)
            adversarial_loss_sr_gen = self.GAN_Loss(self.img_dis_sr, True)
            adversarial_loss_hr_gen = self.GAN_Loss(self.img_dis_hr, True)
            adversarial_loss = adversarial_loss_sr_gen + adversarial_loss_hr_gen
            # percept_w = 0.01


            gen_loss = adversarial_loss * self.args.adv_w + content_loss * self.args.con_w 

            gen_loss.backward()
            self.gen_loss = gen_loss.item()
            # self.edge_loss = self.args.edge_loss_w * (edge_loss.item())
            # self.tv_loss = tv_loss.item()
            self.gen_opt.step()
            

            # training discriminator 
            self.dis_opt.zero_grad()

            self.img_dis_hr = self.dis(self.img_hr.detach())
            self.img_dis_sr = self.dis(self.img_gen.detach())

            adversarial_loss_true = self.GAN_Loss(self.img_dis_hr, True)
            adversarial_loss_false = self.GAN_Loss(self.img_dis_sr, False)
            discriminator_loss = adversarial_loss_true + adversarial_loss_false

            discriminator_loss.backward()
            self.dis_loss = discriminator_loss.item()
            # print(self.dis_loss)
            self.dis_opt.step()

            self.gen_sch.step()
            self.dis_sch.step()

        else:
            raise NotImplementedError('training type is not possible')
    
    # def update_D(self):
    #     if self.training_type == 'gan':
    #         # training discriminator 
    #         self.dis_opt.zero_grad()

    #         self.img_dis_hr = self.dis(self.img_hr)
    #         self.img_dis_sr = self.dis(self.img_gen)
    #         adversarial_loss_true = self.GAN_Loss(self.img_dis_hr, True)
    #         adversarial_loss_false = self.GAN_Loss(self.img_dis_sr, False)
    #         discriminator_loss = adversarial_loss_true + adversarial_loss_false

    #         discriminator_loss.backward()
    #         self.dis_loss = discriminator_loss.item()
    #         self.dis_opt.step()

    #         # self.gen_sch.step()
    #         # self.dis_sch.step()

    #     else:
    #         raise NotImplementedError('training type is not possible')

    def resume(self, model_dir, train=True):
        checkpoint = torch.load(model_dir, map_location=lambda storage, loc: storage)
        self.gen.load_state_dict(checkpoint['gen'])
        if train:
            self.gen_opt.load_state_dict(checkpoint['gen_opt'])
            if self.training_type == 'endosr' or self.training_type == 'gan':
                self.dis.load_state_dict(checkpoint['dis'])
                self.dis_opt.load_state_dict(checkpoint['dis_opt'])
        return checkpoint['ep'], checkpoint['total_it']

    def state_save(self, filename, ep, total_it):
        state = {'gen': self.gen.state_dict(),
                 'gen_opt': self.gen_opt.state_dict(),
                 'ep': ep,
                 'total_it': total_it
                 }
        if self.training_type == 'endosr' or self.training_type == 'gan':
            state['dis'] = self.dis.state_dict()
            state['dis_opt'] = self.dis_opt.state_dict()
        time.sleep(5)
        torch.save(state, filename)
        return

    def model_save(self, filename, ep, total_it):
        state = {'gen': self.gen.state_dict()}
        if self.training_type == 'endosr' or self.training_type == 'gan':
            state['dis'] = self.dis.state_dict()
        time.sleep(5)
        torch.save(state, filename)
        return
    
    def model_save_best(self, filename, ep, total_it):
        state = {'gen': self.gen.state_dict()}
        if self.training_type == 'endosr' or self.training_type == 'gan':
            state['dis'] = self.dis.state_dict()
        time.sleep(5)
        torch.save(state, filename)
        return


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()

        vgg = vgg19(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:35]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.l1_loss = nn.L1Loss()

    def forward(self, high_resolution, fake_high_resolution):
        perception_loss = self.l1_loss(self.loss_network(high_resolution), self.loss_network(fake_high_resolution))
        return perception_loss


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.L1Loss()  # nn.L1Loss
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real == 'real':
            target_tensor = self.real_label
        elif target_is_real == 'fake':
            target_tensor = self.fake_label
        else:
            target_tensor = (self.real_label + self.fake_label) / 2
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

class GANLoss1(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss1, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':

            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss

class DWT_LOSS(nn.Module):
    def __init__(self):
        super(DWT_LOSS, self).__init__()
        self.requires_grad = False

    def dwt_init(self, x):
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        x_LL = x1 + x2 + x3 + x4
        x_HL = -x1 - x2 + x3 + x4
        x_LH = -x1 + x2 - x3 + x4
        x_HH = x1 - x2 - x3 + x4

        return x_LL, x_HL, x_LH, x_HH

    def forward(self, x, target):
        x_LL1, x_HL1, x_LH1, x_HH1 = self.dwt_init(x)
        y_LL1, y_HL1, y_LH1, y_HH1 = self.dwt_init(target)
        HL1_loss = F.l1_loss(x_HL1, y_HL1)
        LH1_loss = F.l1_loss(x_LH1, y_LH1)
        HH1_loss = F.l1_loss(x_HH1, y_HH1)
        x_LL2, x_HL2, x_LH2, x_HH2 = self.dwt_init(x_LL1)
        y_LL2, y_HL2, y_LH2, y_HH2 = self.dwt_init(y_LL1)
        HL2_loss = F.l1_loss(x_HL2, y_HL2)
        LH2_loss = F.l1_loss(x_LH2, y_LH2)
        HH2_loss = F.l1_loss(x_HH2, y_HH2)
        x_LL3, x_HL3, x_LH3, x_HH3 = self.dwt_init(x_LL2)
        y_LL3, y_HL3, y_LH3, y_HH3 = self.dwt_init(y_LL2)
        HL3_loss = F.l1_loss(x_HL3, y_HL3)
        LH3_loss = F.l1_loss(x_LH3, y_LH3)
        HH3_loss = F.l1_loss(x_HH3, y_HH3)
        return 0.12 * HL2_loss + 0.12 * LH2_loss + 0.12 * HH2_loss + 0.05 * HL3_loss + 0.05 * LH3_loss + 0.05 * HH3_loss + 0.16 * LH1_loss + 0.16 * HL1_loss + 0.16 * HH1_loss
    
class CannyEdgeDetection(nn.Module):
    """Class for applying Canny edge detection on images."""
    def __init__(self, low_threshold: int = 40, high_threshold: int = 85):
        super(CannyEdgeDetection, self).__init__()
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a PyTorch tensor")
        
        if x.dim() != 4:
            raise ValueError("Input tensor must have 4 dimensions (B, C, H, W)")
        
        batch_size, channels, height, width = x.shape
        device = x.device

        edge_maps = []
        for img in x:
            img_np = self._tensor_to_numpy(img)
            img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(img_gray, self.low_threshold, self.high_threshold)
            edges_tensor = torch.from_numpy(edges).float().to(device).unsqueeze(0).unsqueeze(0)
            edge_maps.append(edges_tensor)

        return torch.cat(edge_maps, dim=0)

    @staticmethod
    def _tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
        img_np = tensor.permute(1, 2, 0).cpu().detach().numpy()
        normalized = (img_np + 1) / 2
        return np.floor(np.clip(normalized * 256, 0, 255)).astype(np.uint8)

class EdgeLoss(nn.Module):
    """Class for calculating Edge Loss between two batches of RGB images."""
    def __init__(self, low_threshold: int = 40, high_threshold: int = 85):
        super(EdgeLoss, self).__init__()
        self.edge_detector = CannyEdgeDetection(low_threshold, high_threshold)

    def forward(self, hr_image: torch.Tensor, sr_image: torch.Tensor) -> torch.Tensor:
        if not (isinstance(hr_image, torch.Tensor) and isinstance(sr_image, torch.Tensor)):
            raise TypeError("Inputs must be PyTorch tensors")
        
        if hr_image.shape != sr_image.shape:
            raise ValueError("HR and SR images must have the same shape")

        device = hr_image.device
        self.edge_detector = self.edge_detector.to(device)

        hr_edge_map = self.edge_detector(hr_image)
        sr_edge_map = self.edge_detector(sr_image)

        hr_image_gray = hr_image.mean(dim=1, keepdim=True)
        sr_image_gray = sr_image.mean(dim=1, keepdim=True)

        reconstruction_error = torch.abs(hr_image_gray - sr_image_gray)
        edge_loss_hr = torch.mean(hr_edge_map * reconstruction_error)
        edge_loss_sr = torch.mean(sr_edge_map * reconstruction_error)

        return edge_loss_hr + edge_loss_sr

    @torch.jit.script
    def _compute_edge_loss(edge_map: torch.Tensor, reconstruction_error: torch.Tensor) -> torch.Tensor:
        return torch.mean(edge_map * reconstruction_error)
    

class BTVLoss(nn.Module):
    def __init__(self, weight, neighborhood_size=3, epsilon=1e-6):
        super(BTVLoss, self).__init__()
        self.weight = weight
        self.neighborhood_size = neighborhood_size
        self.epsilon = epsilon

    def forward(self, x):
        batch_size, c, h, w = x.size()
        btv_loss = 0.0
        for k in range(-self.neighborhood_size, self.neighborhood_size + 1):
            for l in range(-self.neighborhood_size, self.neighborhood_size + 1):
                if k == 0 and l == 0:
                    continue
                shifted_x = torch.roll(x, shifts=(k, l), dims=(2, 3))
                diff = torch.abs(x - shifted_x)
                btv_loss += torch.sqrt(diff * diff + self.epsilon).sum()
        return self.weight * btv_loss / (batch_size * c * h * w)

class TotalVariationLoss(nn.Module):
    def __init__(self):
        super(TotalVariationLoss, self).__init__()

    def forward(self, img):
        """
        Compute total variation loss.
        
        Args:
            img (torch.Tensor): Image tensor of shape (B, C, H, W)
        
        Returns:
            torch.Tensor: Total variation loss
        """
        tv_h = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]).sum()
        tv_w = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]).sum()
        return tv_h + tv_w
