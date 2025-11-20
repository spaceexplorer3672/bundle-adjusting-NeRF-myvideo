# data/myvideo.py
import os
from PIL import Image
import numpy as np
import torch
from easydict import EasyDict as edict

from .base import Dataset
from torch.utils.data._utils.collate import default_collate

class MyVideoDataset(Dataset):
    """
    Simple, robust dataset for BARF using video frames from a folder.

    Expects:
        opt.data.root = "data/myvideo" (or similar)
        and inside that folder:
            images/
              frame_00001.jpg
              frame_00002.jpg
              ...

    This class is defensive about the type of elements in self.list:
      - filename string (normal)
      - tuple like (filename, ...)
      - already-loaded PIL.Image (in case something mutated self.list)

    Camera pose convention (IMPORTANT):
      - BARF expects poses with shape [..., 3, 4] = [R|t]
      - R: [..., 3, 3], t: [..., 3]
    """

    def __init__(self, opt, split="train"):
        self.opt = opt
        self.split = split

        # figure out root and images directory
        root = getattr(opt.data, "root", None) or "data/myvideo"
        self.root = os.path.abspath(root)
        self.img_dir = os.path.join(self.root, "images")
        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"MyVideoDataset: images directory not found: {self.img_dir}")

        # initial list of filenames (strings)
        files = sorted(
            f for f in os.listdir(self.img_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        if len(files) == 0:
            raise RuntimeError(f"MyVideoDataset: no images found in {self.img_dir}")

        self.list = files  # keep as filenames

        # detect H, W from first frame
        sample_path = os.path.join(self.img_dir, self.list[0])
        sample_img = Image.open(sample_path).convert("RGB")
        w, h = sample_img.size
        self.raw_H = h
        self.raw_W = w

        # focal: if not set, use simple heuristic
        if not hasattr(opt.data, "focal") or opt.data.focal is None:
            opt.data.focal = 0.5 * float(self.raw_W)

        # call base Dataset init (handles cropping, opt.H/opt.W, etc.)
        super().__init__(opt, split)

        self.all = None  # will hold preloaded (collated) data

    # ----- required interface helpers ----------------------------------

    def get_list(self, opt):
        return np.arange(len(self.list))

    def get_image(self, opt, idx):
        """
        Robust loader returning a PIL.Image for the given index.

        Handles:
          - self.list[idx] is a filename string/path
          - self.list[idx] is a tuple/list whose first element is a filename or Image
          - self.list[idx] is already a PIL.Image
        """
        item = self.list[idx]

        # case: already an Image
        if isinstance(item, Image.Image):
            return item.convert("RGB")

        # case: tuple/list like (filename, pose, ...) or (Image, ...)
        if isinstance(item, (tuple, list)):
            cand = item[0]
            if isinstance(cand, Image.Image):
                return cand.convert("RGB")
            if isinstance(cand, (str, bytes, os.PathLike)):
                full = os.path.join(self.img_dir, str(cand))
                return Image.open(full).convert("RGB")
            # fallback: try stringifying
            full = os.path.join(self.img_dir, str(cand))
            return Image.open(full).convert("RGB")

        # case: filename string/path
        if isinstance(item, (str, bytes, os.PathLike)):
            full = os.path.join(self.img_dir, str(item))
            return Image.open(full).convert("RGB")

        # anything else: error
        raise RuntimeError(
            f"MyVideoDataset.get_image: unsupported type in self.list[{idx}]: {type(item)}"
        )

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        opt = self.opt
        pil_img = self.get_image(opt, idx)

        # augmentation if enabled
        aug = self.generate_augmentation(opt) if self.augment else None

        # preprocess (crop/resize/aug -> tensor)
        image_tensor = self.preprocess_image(opt, pil_img, aug=aug)

        intr = torch.tensor([
            [opt.data.focal, 0.0, opt.W / 2.0],
            [0.0, opt.data.focal, opt.H / 2.0],
            [0.0, 0.0, 1.0]
        ], dtype=torch.float32)

        # ---- pose: 3x4, [R|t] with identity R and zero t ----
        R = torch.eye(3, dtype=torch.float32)
        t = torch.zeros(3, dtype=torch.float32)
        pose = torch.cat([R, t[:, None]], dim=-1)  # (3,4)

        sample = edict(
            image=image_tensor,
            intr=intr,
            pose=pose,
            idx=idx
        )
        return sample

    # ----- prefetching --------------------------------------------------

    def prefetch_all_data(self, opt):
        """
        Preload all samples and collate them into batched tensors.

        This matches Blender's behavior so that:
            self.train_data.all = edict(util.move_to_device(self.train_data.all, opt.device))
        works as expected (self.all is a dict).
        """
        samples = [self[i] for i in range(len(self))]
        self.all = default_collate(samples)
        return self.all

    # ----- camera pose & intrinsics for validation ---------------------

    def get_all_camera_poses(self, opt):
        """
        Return 'ground-truth' camera poses for all training images.

        Since we don't have real GT poses for this video, we return identity
        3x4 poses [I | 0]. BARF will still train, but pose error metrics will
        not be meaningful.
        """
        N = len(self)

        R = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(N, 1, 1)  # (N,3,3)
        t = torch.zeros(N, 3, dtype=torch.float32)                           # (N,3)
        pose_GT = torch.cat([R, t.unsqueeze(-1)], dim=-1)                    # (N,3,4)

        return pose_GT

    def get_all_intrinsics(self, opt):
        """
        Return intrinsics for all images as an (N, 3, 3) tensor.
        All frames share the same intrinsics in this simple setup.
        """
        N = len(self)
        intr_single = torch.tensor([
            [opt.data.focal, 0.0, opt.W / 2.0],
            [0.0, opt.data.focal, opt.H / 2.0],
            [0.0, 0.0, 1.0]
        ], dtype=torch.float32)
        intr_all = intr_single.unsqueeze(0).repeat(N, 1, 1)
        return intr_all
