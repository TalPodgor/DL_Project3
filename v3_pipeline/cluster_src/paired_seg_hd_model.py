"""Segmentation-conditioned paired HD model (Wave 5, geometry-first v3).

Same pix2pixHD recipe as paired_hd_model (multi-scale D + feature matching + VGG +
L1), but the generator is conditioned on BOTH the view-aligned synthetic RGB *and* a
one-hot FEN-derived semantic mask:

    G_input = concat( A_synthetic(3) , onehot(seg, K) )   # K = num_seg_classes
    fake_B  = G(G_input)
    loss_G  = GAN + feat-matching + VGG + maskweighted-L1

The mask carries exact piece identity/occupancy from the FEN, so empty squares stay
empty and piece type is supplied (not inferred). Pixel-L1 is down-weighted inside
piece interiors (--l1_piece_w) because the substitute 3D set need not pixel-match the
real pieces; structure there is carried by the mask + GAN.  Train: `--model paired_seg_hd`.
"""
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .base_model import BaseModel
from . import networks
from .paired_hd_model import VGGPerceptualLoss, MultiscaleDiscriminator


class PairedSegHDModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--lambda_GAN', type=float, default=1.0)
        parser.add_argument('--lambda_feat', type=float, default=10.0)
        parser.add_argument('--lambda_VGG', type=float, default=5.0)
        parser.add_argument('--lambda_L1', type=float, default=10.0)
        parser.add_argument('--l1_piece_w', type=float, default=0.3,
                            help='L1 weight inside piece pixels (<1 lets GAN/mask carry pieces)')
        parser.add_argument('--num_seg_classes', type=int, default=15,
                            help='one-hot channels for the seg mask (ids 0..14)')
        parser.add_argument('--num_D', type=int, default=2)
        parser.add_argument('--vgg_max', type=int, default=256)
        parser.add_argument('--g_init_path', type=str, default='')
        parser.set_defaults(dataset_mode='seg_aligned', direction='AtoB',
                            netG='resnet_9blocks', ngf=64, normG='instance', no_dropout=True,
                            gan_mode='lsgan', pool_size=0)
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.K = opt.num_seg_classes
        self.loss_names = ['G_GAN', 'G_FM', 'G_VGG', 'G_L1', 'D_real', 'D_fake']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.model_names = ['G', 'D'] if self.isTrain else ['G']

        g_in = opt.input_nc + self.K  # synthetic RGB + one-hot mask
        self.netG = networks.define_G(
            g_in, opt.output_nc, opt.ngf, opt.netG, opt.normG,
            not opt.no_dropout, opt.init_type, opt.init_gain,
            opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)

        if self.isTrain and getattr(opt, 'g_init_path', ''):
            self._load_g_init(opt.g_init_path)

        if self.isTrain:
            netD = MultiscaleDiscriminator(opt.output_nc, opt.ndf, opt.n_layers_D, opt.num_D)
            self.netD = networks.init_net(netD, opt.init_type, opt.init_gain, self.gpu_ids)
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
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
        own = net.state_dict()
        filt = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
        missing, unexpected = net.load_state_dict(filt, strict=False)
        print(f'[paired_seg_hd] warm-started G from {path}: loaded {len(filt)}/{len(sd)} tensors '
              f'(skipped shape-mismatch e.g. first conv); missing={len(missing)}')

    def data_dependent_initialize(self, data):
        return

    def parallelize(self):
        return

    def save_networks(self, epoch):
        import os
        for name in self.model_names:
            if not isinstance(name, str):
                continue
            save_path = os.path.join(self.save_dir, '%s_net_%s.pth' % (epoch, name))
            net = getattr(self, 'net' + name)
            if isinstance(net, torch.nn.DataParallel):
                net = net.module
            if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                torch.save(net.cpu().state_dict(), save_path); net.cuda(self.gpu_ids[0])
            else:
                torch.save(net.cpu().state_dict(), save_path)

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)      # 3ch (for visuals)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        seg = input['seg'].to(self.device).long()                      # (N,H,W)
        self.seg = seg
        onehot = F.one_hot(seg.clamp(0, self.K - 1), self.K).permute(0, 3, 1, 2).float()
        self.gen_input = torch.cat([self.real_A, onehot], dim=1)       # (N,3+K,H,W)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.fake_B = self.netG(self.gen_input)

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

        fm = 0.0
        feat_w = 4.0 / (self.opt.n_layers_D + 1); d_w = 1.0 / self.opt.num_D
        for i in range(len(pred_fake)):
            for j in range(len(pred_fake[i]) - 1):
                fm = fm + d_w * feat_w * self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach())
        self.loss_G_FM = fm * self.opt.lambda_feat

        self.loss_G_VGG = self.criterionVGG(self.fake_B, self.real_B) * self.opt.lambda_VGG

        # mask-weighted L1: full weight on board/empty, down-weighted on piece pixels
        wmap = torch.ones_like(self.seg, dtype=torch.float32)
        wmap[self.seg >= 3] = self.opt.l1_piece_w
        wmap = wmap.unsqueeze(1)  # (N,1,H,W)
        self.loss_G_L1 = (wmap * (self.fake_B - self.real_B).abs()).mean() * self.opt.lambda_L1

        self.loss_G = self.loss_G_GAN + self.loss_G_FM + self.loss_G_VGG + self.loss_G_L1
        return self.loss_G

    def optimize_parameters(self):
        self.forward()
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.compute_D_loss().backward()
        self.optimizer_D.step()
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.compute_G_loss().backward()
        self.optimizer_G.step()
