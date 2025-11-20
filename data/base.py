import numpy as np
import os,sys,time
import torch
import torch.nn.functional as torch_F
import torchvision
import torchvision.transforms.functional as torchvision_F
import torch.multiprocessing as mp
import PIL
import tqdm
import threading,queue
from easydict import EasyDict as edict

import util
from util import log,debug

class Dataset(torch.utils.data.Dataset):

    def __init__(self, opt, split="train", **kwargs):
        """
        Base Dataset initializer.

        - Accepts extra keyword args (e.g., subset, val_sub) to be robust
        against different dataset factory call-sites.
        - Subclasses are expected to set self.raw_H and self.raw_W *before*
        calling super().__init__(...) OR set them immediately after instantiation
        (many dataset implementations call super() at the end).
        """
        super().__init__()
        self.opt = opt
        self.split = split
        self.augment = (split == "train") and getattr(opt.data, "augment", False)

        # accept optional subset parameters passed from caller (train_sub / val_sub)
        # store them for later use by subclasses or loader logic
        self.subset = kwargs.get("subset", None)   # e.g., opt.data.train_sub
        self.val_sub = kwargs.get("val_sub", None)

        # define image sizes
        # Note: many subclasses set self.raw_H/self.raw_W before calling this.
        # If they haven't been set yet, fall back to opt values or None temporarily.
        raw_H = getattr(self, "raw_H", None)
        raw_W = getattr(self, "raw_W", None)

        if opt.data.center_crop is not None and raw_H is not None and raw_W is not None:
            self.crop_H = int(raw_H * opt.data.center_crop)
            self.crop_W = int(raw_W * opt.data.center_crop)
        else:
            # If subclass hasn't provided raw_H/raw_W yet, try to use opt.H/opt.W or set crop to None.
            if raw_H is None:
                raw_H = getattr(opt, "H", None)
            if raw_W is None:
                raw_W = getattr(opt, "W", None)
            self.crop_H = int(raw_H * opt.data.center_crop) if (opt.data.center_crop is not None and raw_H is not None) else raw_H
            self.crop_W = int(raw_W * opt.data.center_crop) if (opt.data.center_crop is not None and raw_W is not None) else raw_W

        # If crop sizes are still None, set them to opt.H/opt.W if available
        if self.crop_H is None:
            self.crop_H = getattr(opt, "H", None)
        if self.crop_W is None:
            self.crop_W = getattr(opt, "W", None)

        # If opt.H / opt.W are not set (None/0), set them to crop sizes if we have them
        if not getattr(opt, "H", None) or not getattr(opt, "W", None):
            if self.crop_H is not None and self.crop_W is not None:
                opt.H, opt.W = int(self.crop_H), int(self.crop_W)


    def setup_loader(self,opt,shuffle=False,drop_last=False):
        loader = torch.utils.data.DataLoader(self,
            batch_size=opt.batch_size or 1,
            num_workers=opt.data.num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
            pin_memory=False, # spews warnings in PyTorch 1.9 but should be True in general
        )
        print("number of samples: {}".format(len(self)))
        return loader

    def get_list(self,opt):
        raise NotImplementedError

    def preload_worker(self,data_list,load_func,q,lock,idx_tqdm):
        while True:
            idx = q.get()
            data_list[idx] = load_func(self.opt,idx)
            with lock:
                idx_tqdm.update()
            q.task_done()

    def preload_threading(self,opt,load_func,data_str="images"):
        data_list = [None]*len(self)
        q = queue.Queue(maxsize=len(self))
        idx_tqdm = tqdm.tqdm(range(len(self)),desc="preloading {}".format(data_str),leave=False)
        for i in range(len(self)): q.put(i)
        lock = threading.Lock()
        for ti in range(opt.data.num_workers):
            t = threading.Thread(target=self.preload_worker,
                                 args=(data_list,load_func,q,lock,idx_tqdm),daemon=True)
            t.start()
        q.join()
        idx_tqdm.close()
        assert(all(map(lambda x: x is not None,data_list)))
        return data_list

    def __getitem__(self,idx):
        raise NotImplementedError

    def get_image(self,opt,idx):
        raise NotImplementedError

    def generate_augmentation(self,opt):
        brightness = opt.data.augment.brightness or 0.
        contrast = opt.data.augment.contrast or 0.
        saturation = opt.data.augment.saturation or 0.
        hue = opt.data.augment.hue or 0.
        color_jitter = torchvision.transforms.ColorJitter.get_params(
            brightness=(1-brightness,1+brightness),
            contrast=(1-contrast,1+contrast),
            saturation=(1-saturation,1+saturation),
            hue=(-hue,hue),
        )
        aug = edict(
            color_jitter=color_jitter,
            flip=np.random.randn()>0 if opt.data.augment.hflip else False,
            rot_angle=(np.random.rand()*2-1)*opt.data.augment.rotate if opt.data.augment.rotate else 0,
        )
        return aug

    def preprocess_image(self,opt,image,aug=None):
        if aug is not None:
            image = self.apply_color_jitter(opt,image,aug.color_jitter)
            image = torchvision_F.hflip(image) if aug.flip else image
            image = image.rotate(aug.rot_angle,resample=PIL.Image.BICUBIC)
        # center crop
        if opt.data.center_crop is not None:
            self.crop_H = int(self.raw_H*opt.data.center_crop)
            self.crop_W = int(self.raw_W*opt.data.center_crop)
            image = torchvision_F.center_crop(image,(self.crop_H,self.crop_W))
        else: self.crop_H,self.crop_W = self.raw_H,self.raw_W
        # resize
        if opt.data.image_size[0] is not None:
            image = image.resize((opt.W,opt.H))
        image = torchvision_F.to_tensor(image)
        return image

    def preprocess_camera(self,opt,intr,pose,aug=None):
        intr,pose = intr.clone(),pose.clone()
        # center crop
        intr[0,2] -= (self.raw_W-self.crop_W)/2
        intr[1,2] -= (self.raw_H-self.crop_H)/2
        # resize
        intr[0] *= opt.W/self.crop_W
        intr[1] *= opt.H/self.crop_H
        return intr,pose

    def apply_color_jitter(self,opt,image,color_jitter):
        mode = image.mode
        if mode!="L":
            chan = image.split()
            rgb = PIL.Image.merge("RGB",chan[:3])
            rgb = color_jitter(rgb)
            rgb_chan = rgb.split()
            image = PIL.Image.merge(mode,rgb_chan+chan[3:])
        return image

    def __len__(self):
        return len(self.list)
