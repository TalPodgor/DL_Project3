"""Paired model + VGG perceptual loss — Wave 3 drop-in for the CUT repo.

DEPLOYMENT
----------
Copy into the cluster CUT checkout:
    ~/chess_cut_project/contrastive-unpaired-translation/models/paired_perc_cut_model.py
Train with `--model paired_perc_cut`.

WHY THIS EXISTS (Wave 3)
------------------------
Wave 2 (PairedCutModel: LSGAN + pixel-L1) removed phantom pieces but pixel-L1 pulls
toward the per-pixel mean, which slightly BLURS the output. Wave 3 keeps the L1 anchor
(so empty squares stay empty) and ADDS a VGG19 perceptual loss, which matches deep
features instead of raw pixels and restores high-frequency sharpness/texture:

    loss_G = lambda_GAN*LSGAN(D(G(A))) + lambda_L1*L1(G(A),B) + lambda_VGG*VGG(G(A),B)

The generator and (unconditional) discriminator are built IDENTICALLY to Wave 2 /
cut_model.py, so this fine-tunes directly from the Wave 2 checkpoint
(checkpoints/chess_paired/latest_net_{G,D}.pth) via --continue_train.

The VGG network is a FROZEN, pretrained feature extractor used only for the loss; it is
not in model_names (never saved/loaded) and its parameters are not optimized.
"""
import torch
import torch.nn as nn
import torchvision
from .base_model import BaseModel
from . import networks


class VGGPerceptualLoss(nn.Module):
    """pix2pixHD-style VGG19 feature-space L1. Inputs are expected in [-1, 1]."""

    def __init__(self, device):
        super().__init__()
        vgg = torchvision.models.vgg19(pretrained=True).features.to(device).eval()
        for p in vgg.parameters():
            p.requires_grad = False
        # Slice boundaries capturing relu1_1, relu2_1, relu3_1, relu4_1, relu5_1.
        idxs = [0, 2, 7, 12, 21, 30]
        self.slices = nn.ModuleList()
        for i in range(5):
            seq = nn.Sequential()
            for j in range(idxs[i], idxs[i + 1]):
                seq.add_module(str(j), vgg[j])
            self.slices.append(seq)
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.criterion = nn.L1Loss()
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1))

    def _prep(self, x):
        # [-1,1] -> [0,1] -> ImageNet-normalized (what VGG expects)
        x = (x + 1.0) / 2.0
        return (x - self.mean) / self.std

    def forward(self, fake, real):
        f, r = self._prep(fake), self._prep(real)
        loss = 0.0
        for w, slc in zip(self.weights, self.slices):
            f, r = slc(f), slc(r)
            loss = loss + w * self.criterion(f, r.detach())
        return loss


class PairedPercCutModel(BaseModel):
    """Supervised synth->real translation: LSGAN + L1 + VGG perceptual."""

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--lambda_GAN', type=float, default=1.0,
                            help='weight for the LSGAN loss on G(A)')
        parser.add_argument('--lambda_L1', type=float, default=5.0,
                            help='weight for pixel-L1 (occupancy/structure anchor). Lower than Wave 2 (10) '
                                 'because VGG now carries part of the supervision and lower L1 = less blur.')
        parser.add_argument('--lambda_VGG', type=float, default=10.0,
                            help='weight for VGG19 perceptual loss (sharpness/texture).')
        parser.set_defaults(dataset_mode='aligned', direction='AtoB',
                            netG='resnet_9blocks', ngf=64, normG='instance', no_dropout=True,
                            netD='basic', n_layers_D=3, ndf=64, normD='instance',
                            gan_mode='lsgan', pool_size=0)
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['G_GAN', 'G_L1', 'G_VGG', 'D_real', 'D_fake']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.model_names = ['G', 'D'] if self.isTrain else ['G']

        self.netG = networks.define_G(
            opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG,
            not opt.no_dropout, opt.init_type, opt.init_gain,
            opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)

        if self.isTrain:
            self.netD = networks.define_D(
                opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD,
                opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionVGG = VGGPerceptualLoss(self.device)

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def data_dependent_initialize(self, data):
        return

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.fake_B = self.netG(self.real_A)

    def compute_D_loss(self):
        fake = self.fake_B.detach()
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        pred_real = self.netD(self.real_B)
        self.loss_D_real = self.criterionGAN(pred_real, True).mean()
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_G_loss(self):
        pred_fake = self.netD(self.fake_B)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        self.loss_G_VGG = self.criterionVGG(self.fake_B, self.real_B) * self.opt.lambda_VGG
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_VGG
        return self.loss_G

    def optimize_parameters(self):
        self.forward()
        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()
        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
