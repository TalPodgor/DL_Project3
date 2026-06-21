"""Segmentation-conditioned aligned dataset (Wave 5, geometry-first v3).

Each training item is:
    {name}.png      = [A_synthetic | B_real]  (side-by-side, like AlignedDataset)
    {name}_seg.png  = class-id mask for A/B (single channel, ids 0..14)

Returns A (3ch, [-1,1]), B (3ch, [-1,1]), seg (long HxW class ids), with the SAME
geometric augmentation (resize/crop/flip) applied to all three so they stay
registered. Train with `--dataset_mode seg_aligned`.
"""
import os
import numpy as np
import torch
from PIL import Image
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset


class SegAlignedDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))
        self.AB_paths = [p for p in paths if not p.endswith('_seg.png')]
        assert self.opt.load_size >= self.opt.crop_size
        self.input_nc = self.opt.input_nc
        self.output_nc = self.opt.output_nc

    def _seg_transform(self, seg, params):
        # mirror get_transform geometry (NEAREST, no normalize) -> long index tensor
        pp = self.opt.preprocess
        if 'resize' in pp:
            seg = seg.resize((self.opt.load_size, self.opt.load_size), Image.NEAREST)
        if 'crop' in pp:
            x, y = params['crop_pos']; cs = self.opt.crop_size
            seg = seg.crop((x, y, x + cs, y + cs))
        if (not self.opt.no_flip) and params.get('flip', False):
            seg = seg.transpose(Image.FLIP_LEFT_RIGHT)
        return torch.from_numpy(np.array(seg, dtype=np.int64))

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        w, h = AB.size; w2 = w // 2
        A = AB.crop((0, 0, w2, h)); B = AB.crop((w2, 0, w, h))
        seg = Image.open(AB_path[:-4] + '_seg.png')  # mode L, class ids

        params = get_params(self.opt, A.size)
        A = get_transform(self.opt, params, grayscale=(self.input_nc == 1))(A)
        B = get_transform(self.opt, params, grayscale=(self.output_nc == 1))(B)
        seg = self._seg_transform(seg, params)
        return {'A': A, 'B': B, 'seg': seg, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        return len(self.AB_paths)
