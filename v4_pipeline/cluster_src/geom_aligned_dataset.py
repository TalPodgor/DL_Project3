"""Geometry-conditioned aligned dataset for v4.

Each item:
  {name}.png      = [A_synthetic | B_real]
  {name}_seg.png  = class-id mask, ids 1..14
  {name}_geom.png = 3ch geometry hints derived from the rendered mask
"""
import os

import numpy as np
import torch
from PIL import Image

from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset


class GeomAlignedDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))
        self.AB_paths = [
            p for p in paths
            if not p.endswith("_seg.png") and not p.endswith("_geom.png")
        ]
        assert self.opt.load_size >= self.opt.crop_size
        self.input_nc = self.opt.input_nc
        self.output_nc = self.opt.output_nc

    def _seg_transform(self, seg, params):
        pp = self.opt.preprocess
        if "resize" in pp:
            seg = seg.resize((self.opt.load_size, self.opt.load_size), Image.NEAREST)
        if "crop" in pp:
            x, y = params["crop_pos"]
            cs = self.opt.crop_size
            seg = seg.crop((x, y, x + cs, y + cs))
        if (not self.opt.no_flip) and params.get("flip", False):
            seg = seg.transpose(Image.FLIP_LEFT_RIGHT)
        return torch.from_numpy(np.array(seg, dtype=np.int64))

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert("RGB")
        w, h = AB.size
        w2 = w // 2
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))
        seg = Image.open(AB_path[:-4] + "_seg.png")
        geom = Image.open(AB_path[:-4] + "_geom.png").convert("RGB")

        params = get_params(self.opt, A.size)
        A = get_transform(self.opt, params, grayscale=(self.input_nc == 1))(A)
        B = get_transform(self.opt, params, grayscale=(self.output_nc == 1))(B)
        geom = get_transform(self.opt, params, grayscale=False)(geom)
        seg = self._seg_transform(seg, params)
        return {"A": A, "B": B, "seg": seg, "geom": geom,
                "A_paths": AB_path, "B_paths": AB_path}

    def __len__(self):
        return len(self.AB_paths)
