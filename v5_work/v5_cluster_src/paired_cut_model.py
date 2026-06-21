"""Paired (supervised) translation model — drop-in for the CUT repo.

DEPLOYMENT
----------
Copy this file into the cluster CUT checkout:
    ~/chess_cut_project/contrastive-unpaired-translation/models/paired_cut_model.py
Then train with `--model paired_cut` (see train_paired.sh).

WHY THIS EXISTS (Wave 2 of the refactor)
----------------------------------------
The deployed model was trained with the *unpaired* CUT objective (PatchNCE + GAN),
which has no per-square occupancy supervision -> phantom pieces on empty squares.
Our data is in fact paired (Wave 1 produced an aligned [synthetic | real] dataset),
so we switch to a *supervised* objective:

    loss_G = lambda_GAN * LSGAN(D(G(A))) + lambda_L1 * L1(G(A), B)

The L1 term ties every output pixel to the real target of the *same* FEN, which is
exactly the occupancy anchor the unpaired objective lacked.

WEIGHT REUSE (the whole point — fine-tune, do not retrain)
----------------------------------------------------------
The generator and the discriminator are built EXACTLY as in cut_model.py:
  * G = resnet_9blocks, ngf=64, instance norm, antialiasing on  (input_nc=3, output_nc=3)
  * D = `basic` PatchGAN, n_layers=3, ndf=64, instance norm, **UNCONDITIONAL**
        (sees only the image, 3-channel input — NOT the pix2pix 6-channel concat).
Keeping D unconditional is deliberate: it lets `latest_net_D.pth` load directly.
With `--continue_train --epoch latest`, BaseModel.load_networks() loads
`latest_net_G.pth` and `latest_net_D.pth` from checkpoints/<name>/ unchanged, so the
fine-tune starts from the trained model. (latest_net_F.pth is simply ignored — this
model has no F/MLP network.)
"""
import torch
from .base_model import BaseModel
from . import networks


class PairedCutModel(BaseModel):
    """Supervised synth->real translation: LSGAN + L1, reusing CUT's G and D weights."""

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Defaults are pinned to the trained model so latest_net_{G,D}.pth load cleanly."""
        parser.add_argument('--lambda_GAN', type=float, default=1.0,
                            help='weight for the GAN (LSGAN) loss on G(A)')
        parser.add_argument('--lambda_L1', type=float, default=10.0,
                            help='weight for the supervised L1 loss between G(A) and the real target B. '
                                 'Higher = stronger occupancy/structure anchor (pix2pix uses 100). '
                                 'Key ablation knob — sweep {10,50,100}.')

        # Pin architecture + data mode to match the trained checkpoint.
        parser.set_defaults(dataset_mode='aligned')   # reads dataroot/{train,test}/*.png ([A|B] side-by-side)
        parser.set_defaults(direction='AtoB')          # A = left = synthetic input, B = right = real target
        parser.set_defaults(netG='resnet_9blocks', ngf=64, normG='instance', no_dropout=True)
        parser.set_defaults(netD='basic', n_layers_D=3, ndf=64, normD='instance')
        parser.set_defaults(gan_mode='lsgan', pool_size=0)
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # Losses tracked by the visualizer (loss_<name> attributes must exist).
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # Images dumped to the HTML/visuals.
        self.visual_names = ['real_A', 'fake_B', 'real_B']

        if self.isTrain:
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']  # discriminator not needed at test time

        # --- Generator: built identically to cut_model.py so latest_net_G.pth loads ---
        self.netG = networks.define_G(
            opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG,
            not opt.no_dropout, opt.init_type, opt.init_gain,
            opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)

        if self.isTrain:
            # --- Discriminator: UNCONDITIONAL (output_nc channels), as in cut_model.py ---
            self.netD = networks.define_D(
                opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD,
                opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

            # Losses
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()

            # Fresh optimizers (CUT does not save optimizer state) — fine-tune at a low lr.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def data_dependent_initialize(self, data):
        """train.py calls this once on the first batch. CUT uses it to build the F/MLP
        network from a sample feature map; this model has no such network, so it is a
        no-op. Defined only to satisfy the train loop's call signature."""
        return

    def set_input(self, input):
        """Unpack an aligned batch. A=synthetic (input), B=real (target)."""
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.fake_B = self.netG(self.real_A)

    def compute_D_loss(self):
        """LSGAN discriminator loss on detached fakes vs. real targets."""
        fake = self.fake_B.detach()
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()

        pred_real = self.netD(self.real_B)
        self.loss_D_real = self.criterionGAN(pred_real, True).mean()

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_G_loss(self):
        """G = adversarial (fool D) + supervised L1 to the paired real target."""
        pred_fake = self.netD(self.fake_B)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
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
