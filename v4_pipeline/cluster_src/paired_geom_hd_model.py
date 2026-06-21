"""Geometry-conditioned pix2pixHD model for v4.

Generator input:
  concat(A_synthetic RGB, onehot(seg), geom)

The loss keeps strong reconstruction pressure on the board/background, lowers
pixel loss on piece interiors where real/synthetic silhouettes are imperfect, and
optionally adds a frozen real-square classifier loss for piece identity.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .base_model import BaseModel
from . import networks
from .paired_hd_model import VGGPerceptualLoss, MultiscaleDiscriminator


CLASS_FROM_SEG = {
    3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6,
    9: 7, 10: 8, 11: 9, 12: 10, 13: 11, 14: 12,
}


def resnet18_square_classifier(num_classes=13):
    try:
        return torchvision.models.resnet18(weights=None, num_classes=num_classes)
    except TypeError:
        return torchvision.models.resnet18(pretrained=False, num_classes=num_classes)


class FrozenPieceClassifierLoss(nn.Module):
    def __init__(self, ckpt, device, crop=112, min_pixels=20, max_crops=256):
        super().__init__()
        self.crop = crop
        self.min_pixels = min_pixels
        self.max_crops = max_crops
        self.net = resnet18_square_classifier().to(device)
        self.net.load_state_dict(torch.load(ckpt, map_location=device))
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.ce = nn.CrossEntropyLoss()

    def _target_for_cell(self, seg_cell):
        counts = torch.bincount(seg_cell.reshape(-1), minlength=15)
        piece_counts = counts[3:15]
        mx, rel = torch.max(piece_counts, dim=0)
        if int(mx.item()) < self.min_pixels:
            return 0
        seg_id = int(rel.item()) + 3
        return CLASS_FROM_SEG.get(seg_id, 0)

    def forward(self, fake, seg):
        # fake: N,3,512,512 in [-1,1]; seg: N,512,512 ids.
        n, _, h, w = fake.shape
        if h != 512 or w != 512:
            return fake.new_tensor(0.0)
        pad = self.crop // 2
        padded = F.pad(fake, (pad, pad, pad, pad), mode="replicate")
        crops, targets = [], []
        for bi in range(n):
            for row in range(8):
                for col in range(8):
                    y0, y1 = row * 64, (row + 1) * 64
                    x0, x1 = col * 64, (col + 1) * 64
                    target = self._target_for_cell(seg[bi, y0:y1, x0:x1])
                    if target == 0:
                        continue
                    cy = row * 64 + 32 + pad
                    cx = col * 64 + 32 + pad
                    half = self.crop // 2
                    crop = padded[bi:bi + 1, :, cy - half:cy + half, cx - half:cx + half]
                    crop = F.interpolate(crop, size=(64, 64), mode="bilinear", align_corners=False)
                    crops.append(crop)
                    targets.append(target)
        if not crops:
            return fake.new_tensor(0.0)
        if len(crops) > self.max_crops:
            idx = torch.randperm(len(crops), device=fake.device)[:self.max_crops].cpu().tolist()
            crops = [crops[i] for i in idx]
            targets = [targets[i] for i in idx]
        xb = torch.cat(crops, dim=0)
        yb = torch.tensor(targets, dtype=torch.long, device=fake.device)
        logits = self.net(xb)
        return self.ce(logits, yb)


class PairedGeomHDModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument("--lambda_GAN", type=float, default=1.0)
        parser.add_argument("--lambda_feat", type=float, default=10.0)
        parser.add_argument("--lambda_VGG", type=float, default=5.0)
        parser.add_argument("--lambda_L1", type=float, default=10.0)
        parser.add_argument("--lambda_piece_cls", type=float, default=0.0)
        parser.add_argument("--l1_piece_w", type=float, default=0.1)
        parser.add_argument("--num_seg_classes", type=int, default=15)
        parser.add_argument("--num_geom_channels", type=int, default=3)
        parser.add_argument("--num_D", type=int, default=2)
        parser.add_argument("--vgg_max", type=int, default=256)
        parser.add_argument("--g_init_path", type=str, default="")
        parser.add_argument("--piece_cls_path", type=str, default="./checkpoints/square_eval.pth")
        parser.add_argument("--piece_cls_min_pixels", type=int, default=20)
        parser.add_argument("--piece_cls_max_crops", type=int, default=256)
        parser.set_defaults(dataset_mode="geom_aligned", direction="AtoB",
                            netG="resnet_9blocks", ngf=64, normG="instance", no_dropout=True,
                            gan_mode="lsgan", pool_size=0)
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.K = opt.num_seg_classes
        self.GEOM = opt.num_geom_channels
        self.loss_names = ["G_GAN", "G_FM", "G_VGG", "G_L1", "G_PCLS", "D_real", "D_fake"]
        self.visual_names = ["real_A", "fake_B", "real_B"]
        self.model_names = ["G", "D"] if self.isTrain else ["G"]

        g_in = opt.input_nc + self.K + self.GEOM
        self.netG = networks.define_G(
            g_in, opt.output_nc, opt.ngf, opt.netG, opt.normG,
            not opt.no_dropout, opt.init_type, opt.init_gain,
            opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)

        if self.isTrain and getattr(opt, "g_init_path", ""):
            self._load_g_init(opt.g_init_path)

        if self.isTrain:
            netD = MultiscaleDiscriminator(opt.output_nc, opt.ndf, opt.n_layers_D, opt.num_D)
            self.netD = networks.init_net(netD, opt.init_type, opt.init_gain, self.gpu_ids)
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionFeat = nn.L1Loss()
            self.criterionVGG = VGGPerceptualLoss(self.device, max_size=opt.vgg_max)
            self.criterionPiece = None
            if opt.lambda_piece_cls > 0:
                if not os.path.exists(opt.piece_cls_path):
                    raise FileNotFoundError(opt.piece_cls_path)
                self.criterionPiece = FrozenPieceClassifierLoss(
                    opt.piece_cls_path, self.device,
                    crop=112,
                    min_pixels=opt.piece_cls_min_pixels,
                    max_crops=opt.piece_cls_max_crops)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers += [self.optimizer_G, self.optimizer_D]

    def _load_g_init(self, path):
        sd = torch.load(path, map_location=self.device)
        if hasattr(sd, "_metadata"):
            del sd._metadata
        net = self.netG.module if isinstance(self.netG, nn.DataParallel) else self.netG
        own = net.state_dict()
        filt = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
        missing, _ = net.load_state_dict(filt, strict=False)
        print(f"[paired_geom_hd] warm-started G from {path}: loaded {len(filt)}/{len(sd)} tensors; "
              f"missing={len(missing)}")

    def data_dependent_initialize(self, data):
        return

    def parallelize(self):
        return

    def save_networks(self, epoch):
        for name in self.model_names:
            if not isinstance(name, str):
                continue
            save_path = os.path.join(self.save_dir, "%s_net_%s.pth" % (epoch, name))
            net = getattr(self, "net" + name)
            if isinstance(net, torch.nn.DataParallel):
                net = net.module
            if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                torch.save(net.cpu().state_dict(), save_path)
                net.cuda(self.gpu_ids[0])
            else:
                torch.save(net.cpu().state_dict(), save_path)

    def set_input(self, input):
        AtoB = self.opt.direction == "AtoB"
        self.real_A = input["A" if AtoB else "B"].to(self.device)
        self.real_B = input["B" if AtoB else "A"].to(self.device)
        self.geom = input["geom"].to(self.device)
        self.seg = input["seg"].to(self.device).long()
        onehot = F.one_hot(self.seg.clamp(0, self.K - 1), self.K).permute(0, 3, 1, 2).float()
        self.gen_input = torch.cat([self.real_A, onehot, self.geom], dim=1)
        self.image_paths = input["A_paths" if AtoB else "B_paths"]

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
        feat_w = 4.0 / (self.opt.n_layers_D + 1)
        d_w = 1.0 / self.opt.num_D
        for i in range(len(pred_fake)):
            for j in range(len(pred_fake[i]) - 1):
                fm = fm + d_w * feat_w * self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach())
        self.loss_G_FM = fm * self.opt.lambda_feat

        self.loss_G_VGG = self.criterionVGG(self.fake_B, self.real_B) * self.opt.lambda_VGG

        wmap = torch.ones_like(self.seg, dtype=torch.float32)
        wmap[self.seg >= 3] = self.opt.l1_piece_w
        self.loss_G_L1 = (wmap.unsqueeze(1) * (self.fake_B - self.real_B).abs()).mean() * self.opt.lambda_L1

        if self.criterionPiece is not None:
            self.loss_G_PCLS = self.criterionPiece(self.fake_B, self.seg) * self.opt.lambda_piece_cls
        else:
            self.loss_G_PCLS = self.fake_B.new_tensor(0.0)

        self.loss_G = self.loss_G_GAN + self.loss_G_FM + self.loss_G_VGG + self.loss_G_L1 + self.loss_G_PCLS
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
