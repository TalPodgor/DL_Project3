"""Aligned (paired) dataset — drop-in for the CUT repo.

DEPLOYMENT
----------
Copy this file into the cluster CUT checkout:
    ~/chess_cut_project/contrastive-unpaired-translation/data/aligned_dataset.py
Then train with `--dataset_mode aligned` (see train_paired.sh).

WHY THIS EXISTS (Wave 2 of the refactor)
----------------------------------------
The official CUT repo ships ONLY `unaligned_dataset.py` (it is an unpaired method).
Our supervised fine-tune needs PAIRED batches, so we add the canonical pix2pix
`AlignedDataset`: it reads one folder of side-by-side [A | B] images, splits each
into A (left half) and B (right half), and applies the SAME crop+flip params to
both halves so the pair stays geometrically registered.

This matches the Wave 1 output layout exactly:
    datasets/chess_paired/{train,test}/*.png   where each PNG is [synthetic | real]
        -> A = left  = synthetic (input)
        -> B = right = real      (target)
    Train with `--direction AtoB`.
"""
import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image


class AlignedDataset(BaseDataset):
    """A dataset of side-by-side [A | B] images for paired image-to-image translation."""

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        # dataroot/{phase}/  e.g. datasets/chess_paired/train  and  .../test
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))
        assert self.opt.load_size >= self.opt.crop_size, \
            "load_size must be >= crop_size"
        # With AtoB: input = A = synthetic (input_nc), output = B = real (output_nc).
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')

        # Split the combined image into the A (left) and B (right) halves.
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

        # Same transform parameters for A and B -> identical crop + flip -> pair stays aligned.
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        return len(self.AB_paths)
