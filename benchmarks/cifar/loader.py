#############################################
#                DataLoader                 #
#############################################

import torch
import os
import torchvision
from math import ceil
import torchvision.transforms as T
import torch.nn.functional as F

import torch._dynamo
torch._dynamo.config.suppress_errors = True

CIFAR_MEAN = torch.tensor((0.4914, 0.4822, 0.4465), dtype=torch.half)
CIFAR_STD = torch.tensor((0.247, 0.2435, 0.2616), dtype=torch.half)

# @torch.compile()
def batch_color_jitter(inputs, brightness_range: float, contrast_range: float):
    B = inputs.shape[0]
    device = inputs.device
    dtype = inputs.dtype
    brightness_shift = (
        torch.rand(B, 1, 1, 1, device=device, dtype=dtype) * 2 - 1
    ) * brightness_range
    contrast_scale = (
        torch.rand(B, 1, 1, 1, device=device, dtype=dtype) * 2 - 1
    ) * contrast_range + 1
    inputs = inputs + brightness_shift
    inputs = inputs * contrast_scale
    return inputs

# @torch.compile()
def batch_flip_lr(inputs):
    flip_mask = (torch.rand(len(inputs), device=inputs.device) < 0.5).view(-1, 1, 1, 1)
    return torch.where(flip_mask, inputs.flip(-1), inputs)

# @torch.compile()
def batch_crop(images, crop_size):
    B, C, H_padded, W_padded = images.shape
    r = (H_padded - crop_size) // 2
    y_offsets = (torch.rand(B, device=images.device) * (2 * r + 1)).long()
    x_offsets = (torch.rand(B, device=images.device) * (2 * r + 1)).long()
    base_y_coords = torch.arange(crop_size, device=images.device).view(
        1, 1, crop_size, 1
    )
    base_x_coords = torch.arange(crop_size, device=images.device).view(
        1, 1, 1, crop_size
    )
    y_start_coords_expanded = y_offsets.view(B, 1, 1, 1)
    x_start_coords_expanded = x_offsets.view(B, 1, 1, 1)
    y_indices = y_start_coords_expanded + base_y_coords
    y_indices = y_indices.expand(B, C, crop_size, crop_size)
    x_indices = x_start_coords_expanded + base_x_coords
    x_indices = x_indices.expand(B, C, crop_size, crop_size)
    batch_indices = (
        torch.arange(B, device=images.device).view(B, 1, 1, 1).expand_as(y_indices)
    )
    channel_indices = (
        torch.arange(C, device=images.device).view(1, C, 1, 1).expand_as(y_indices)
    )
    cropped_images = images[batch_indices, channel_indices, y_indices, x_indices]
    return cropped_images

class CifarLoader:
    def __init__(self, path, train=True, batch_size=500, aug=None):
        data_path = os.path.join(path, "train.pt" if train else "test.pt")
        if not os.path.exists(data_path):
            dset = torchvision.datasets.CIFAR10(path, download=True, train=train)
            images = torch.tensor(dset.data)
            labels = torch.tensor(dset.targets)
            torch.save({"images": images, "labels": labels, "classes": dset.classes}, data_path)
        data = torch.load(data_path, map_location=torch.device("cpu"), weights_only=True) # cuda when gpu
        self.images, self.labels, self.classes = (
            data["images"],
            data["labels"],
            data["classes"],
        )
        self.images = (
            (self.images.half() / 255)
            .permute(0, 3, 1, 2)
            .to(memory_format=torch.channels_last)
        )
        self.normalize = T.Normalize(CIFAR_MEAN, CIFAR_STD)
        self.proc_images = {}
        self.epoch = 0
        self.aug = aug or {}
        self.batch_size = batch_size
        self.drop_last = train
        self.shuffle = train
        # Pre-allocate indices tensor for better performance
        self._indices = torch.empty(len(self.images), dtype=torch.long, device="cpu") # cuda when gpu

    def __len__(self):
        return (
            len(self.images) // self.batch_size
            if self.drop_last
            else ceil(len(self.images) / self.batch_size)
        )

    def __iter__(self):

        if self.epoch == 0:
            images = self.proc_images["norm"] = self.normalize(self.images)
            # Pre-flip images in order to do every-other epoch flipping scheme
            if self.aug.get("flip", False):
                images = self.proc_images["flip"] = batch_flip_lr(images)
            # Pre-pad images to save time when doing random translation
            pad = self.aug.get("translate", 0)
            if pad > 0:
                # self.proc_images["pad"] = F.pad(images, (pad,)*4, "reflect") # does not work on cpu
                self.proc_images["pad"] = F.pad(images.float(), (pad,)*4, "reflect").half()

        if self.aug.get("translate", 0) > 0:
            images = batch_crop(self.proc_images["pad"], self.images.shape[-2])
        elif self.aug.get("flip", False):
            images = self.proc_images["flip"]
        else:
            images = self.proc_images["norm"]
        # Flip all images together every other epoch. This increases diversity relative to random flipping
        if self.aug.get("flip", False):
            if self.epoch % 2 == 1:
                images = images.flip(-1)

        color_jitter_config = self.aug.get("color_jitter", {"enabled": False})
        if color_jitter_config.get("enabled", False):
            brightness = color_jitter_config.get("brightness_range", 0.1)
            contrast = color_jitter_config.get("contrast_range", 0.1)
            images = batch_color_jitter(images, brightness, contrast)

        self.epoch += 1

        if self.shuffle:
            torch.randperm(len(self._indices), out=self._indices)
            indices = self._indices
        else:
            indices = torch.arange(len(self.images), device=self.images.device)
        for i in range(len(self)):
            idxs = indices[i * self.batch_size : (i + 1) * self.batch_size]
            yield (images[idxs], self.labels[idxs])