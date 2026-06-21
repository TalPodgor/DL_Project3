"""Paired high-fidelity translation model (pix2pixHD-style) — Wave 4 drop-in.

DEPLOYMENT
----------
Copy into the cluster CUT checkout:
    ~/chess_cut_project/contrastive-unpaired-translation/models/paired_hd_model.py
Train with `--model paired_hd`.

WHY THIS EXISTS (Wave 4)
------------------------
Waves 2-3 (single unconditional PatchGAN + L1 [+VGG]) plateaued at: cold/desaturated
colour, blur, weak piece identity.  Root causes (see REFACTOR_LOG Wave 4):
  * colour was ILL-POSED (per-game white balance) -> fixed in the DATA (v2 dataset,
    chroma-canonicalised targets); this model trains on datasets/chess_paired_v2.
  * blur/identity were limited by 256px + a single weak D.

This model is the standard recipe for high-fidelity *paired* translation on one GPU:

    loss_G = lambda_GAN * LSGAN(D(fake))                         # realism
           + lambda_feat * FM(D(fake), D(real))                 # multi-scale feature matching (sharp, stable)
           + lambda_VGG  * VGG19(fake, real)                    # perceptual texture/identity
           + lambda_L1   * ||fake - real||_1                    # hard occupancy/structure anchor (priority #1)

with a MULTI-SCALE discriminator (num_D PatchGANs at successively halved resolutions),
which gives both local-texture and larger-context realism pressure -> better piece
coherence.  The generator is the SAME resnet_9blocks as before, so it warm-starts from
the Wave 2 generator via --g_init_path (the new multi-scale D trains from scratch).

The VGG net is a frozen feature extractor (not saved/loaded, not optimised).
"""
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .base_model import BaseModel
from . import networks


# --------------------------------------------------------------------------- #
# VGG19 perceptual loss (pix2pixHD-style). Inputs in [-1, 1].
# Runs at a capped resolution to bound memory at 512px training.
# --------------------------------------------------------------------------- #
class VGGPerceptualLoss(nn.Module):
    def __init__(self, device, max_size=256):
        super().__init__()
        vgg = torchvision.models.vgg19(pretrained=True).features.to(device).eval()
        for p in vgg.parameters():
            p.requires_grad = False
        idxs = [0, 2, 7, 12, 21, 30]  # relu1_1..relu5_1 boundaries
        self.slices = nn.ModuleList()
        for i in range(5):
            seq = nn.Sequential()
            for j in range(idxs[i], idxs[i + 1]):
                seq.add_module(str(j), vgg[j])
            self.slices.append(seq)
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.criterion = nn.L1Loss()
        self.max_size = max_size
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1))

    def _prep(self, x):
        if self.max_size and x.shape[-1] > self.max_size:
            x = F.interpolate(x, size=(self.max_size, self.max_size), mode='bilinear', align_corners=False)
        x = (x + 1.0) / 2.0
        return (x - self.mean) / self.std

    def forward(self, fake, real):
        f, r = self._prep(fake), self._prep(real)
        loss = 0.0
        for w, slc in zip(self.weights, self.slices):
            f, r = slc(f), slc(r)
            loss = loss + w * self.criterion(f, r.detach())
        return loss


# --------------------------------------------------------------------------- #
# PatchGAN discriminator that ALSO returns intermediate features (for FM loss).
# --------------------------------------------------------------------------- #
class NLayerDiscriminatorFeat(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        kw, padw = 4, 1
        sequence = [[nn.Conv2d(input_nc, ndf, kw, 2, padw), nn.LeakyReLU(0.2, True)]]
        nf = ndf
        for n in range(1, n_layers):
            nf_prev, nf = nf, min(nf * 2, 512)
            sequence += [[nn.Conv2d(nf_prev, nf, kw, 2, padw), norm_layer(nf), nn.LeakyReLU(0.2, True)]]
        nf_prev, nf = nf, min(nf * 2, 512)
        sequence += [[nn.Conv2d(nf_prev, nf, kw, 1, padw), norm_layer(nf), nn.LeakyReLU(0.2, True)]]
        sequence += [[nn.Conv2d(nf, 1, kw, 1, padw)]]
        # register each block so we can read its output
        self.n_blocks = len(sequence)
        for i, blk in enumerate(sequence):
            setattr(self, 'model' + str(i), nn.Sequential(*blk))

    def forward(self, x):
        feats = [x]
        for i in range(self.n_blocks):
            feats.append(getattr(self, 'model' + str(i))(feats[-1]))
        return feats[1:]  # [feat_1, ..., feat_{N-1}, logits]


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, num_D=2):
        super().__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        for i in range(num_D):
            setattr(self, 'disc' + str(i), NLayerDiscriminatorFeat(input_nc, ndf, n_layers))
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

    def forward(self, x):
        result = []
        cur = x
        for i in range(self.num_D):
            result.append(getattr(self, 'disc' + str(i))(cur))
            if i != self.num_D - 1:
                cur = self.downsample(cur)
        return result  # list[num_D] of list[n_layers+1] feature maps


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #
class PairedHDModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--lambda_GAN', type=float, default=1.0)
        parser.add_argument('--lambda_feat', type=float, default=10.0, help='multi-scale D feature-matching weight')
        parser.add_argument('--lambda_VGG', type=float, default=5.0, help='VGG19 perceptual weight')
        parser.add_argument('--lambda_L1', type=float, default=10.0, help='pixel-L1 occupancy/structure anchor')
        parser.add_argument('--num_D', type=int, default=2, help='number of discriminator scales')
        parser.add_argument('--vgg_max', type=int, default=256, help='cap VGG input resolution (memory)')
        parser.add_argument('--g_init_path', type=str, default='', help='path to a net_G .pth to warm-start G (D stays fresh)')
        parser.set_defaults(dataset_mode='aligned', direction='AtoB',
                            netG='resnet_9blocks', ngf=64, normG='instance', no_dropout=True,
                            gan_mode='lsgan', pool_size=0)
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['G_GAN', 'G_FM', 'G_VGG', 'G_L1', 'D_real', 'D_fake']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.model_names = ['G', 'D'] if self.isTrain else ['G']

        self.netG = networks.define_G(
            opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG,
            not opt.no_dropout, opt.init_type, opt.init_gain,
            opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)

        # Warm-start G from a previous generator checkpoint (identical architecture).
        if self.isTrain and getattr(opt, 'g_init_path', ''):
            self._load_g_init(opt.g_init_path)

        if self.isTrain:
            netD = MultiscaleDiscriminator(opt.output_nc, opt.ndf, opt.n_layers_D, opt.num_D)
            self.netD = networks.init_net(netD, opt.init_type, opt.init_gain, self.gpu_ids)

            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = nn.L1Loss()
            self.criterionFeat = nn.L1Loss()
            self.criterionVGG = VGGPerceptualLoss(self.device, max_size=opt.vgg_max)

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers += [self.optimizer_G, self.optimizer_D]

    def _load_g_init(self, path):
        sd = torch.load(path, map_location=self.device)
        if hasattr(sd, '_metadata'):
            del sd._metadata
        net = self.netG.module if isinstance(self.netG, nn.DataParallel) else self.netG
        missing, unexpected = net.load_state_dict(sd, strict=False)
        print(f'[paired_hd] warm-started G from {path} '
              f'(missing={len(missing)}, unexpected={len(unexpected)})')

    def data_dependent_initialize(self, data):
        return

    def parallelize(self):
        # Single-GPU: nets are already on device (define_G / init_net). Skip DataParallel
        # so the multi-scale D's nested-list output is never scatter/gathered.
        return

    def save_networks(self, epoch):
        # DataParallel-agnostic save: base_model.save_networks() assumes net.module
        # (DataParallel), but parallelize() is a no-op here, so nets are unwrapped.
        import os
        for name in self.model_names:
            if not isinstance(name, str):
                continue
            save_path = os.path.join(self.save_dir, '%s_net_%s.pth' % (epoch, name))
            net = getattr(self, 'net' + name)
            if isinstance(net, torch.nn.DataParallel):
                net = net.module
            if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                torch.save(net.cpu().state_dict(), save_path)
                net.cuda(self.gpu_ids[0])
            else:
                torch.save(net.cpu().state_dict(), save_path)

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.fake_B = self.netG(self.real_A)

    def _gan_over_scales(self, preds, target_is_real):
        loss = 0.0
        for scale in preds:
            loss = loss + self.criterionGAN(scale[-1], target_is_real).mean()
        return loss / len(preds)

    def compute_D_loss(self):
        pred_fake = self.netD(self.fake_B.detach())
        self.loss_D_fake = self._gan_over_scales(pred_fake, False)
        pred_real = self.netD(self.real_B)
        self.loss_D_real = self._gan_over_scales(pred_real, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_G_loss(self):
        pred_fake = self.netD(self.fake_B)
        with torch.no_grad():
            pred_real = self.netD(self.real_B)

        self.loss_G_GAN = self._gan_over_scales(pred_fake, True) * self.opt.lambda_GAN

        # feature matching across all scales and all intermediate D layers (exclude logits)
        fm = 0.0
        feat_w = 4.0 / (self.opt.n_layers_D + 1)
        d_w = 1.0 / self.opt.num_D
        for i in range(len(pred_fake)):
            for j in range(len(pred_fake[i]) - 1):
                fm = fm + d_w * feat_w * self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach())
        self.loss_G_FM = fm * self.opt.lambda_feat

        self.loss_G_VGG = self.criterionVGG(self.fake_B, self.real_B) * self.opt.lambda_VGG
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1

        self.loss_G = self.loss_G_GAN + self.loss_G_FM + self.loss_G_VGG + self.loss_G_L1
        return self.loss_G

    def optimize_parameters(self):
        self.forward()
        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.compute_D_loss().backward()
        self.optimizer_D.step()
        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.compute_G_loss().backward()
        self.optimizer_G.step()
