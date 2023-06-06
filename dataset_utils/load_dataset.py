import torch.utils.data as data
from PIL import Image
from torchvision import transforms
import torch
import numpy as np
import torchvision.transforms.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import get_cfg_defaults 
cfg = get_cfg_defaults()

class LoadDataset(data.Dataset):
    def __init__(self, X, y, train=True):

        self.image_paths, self.class_ids = X, y
        self.train = train

    def __len__(self):

        return len(self.image_paths)

    def __getitem__(self, idx):

        img_path = self.image_paths[idx]
        class_id = self.class_ids[idx]
        img = Image.open(img_path).convert('RGB')
        if self.train:
            image_transform = self.train_img_transforms()
        else:
            image_transform = self.test_img_transforms()
        img = image_transform(img)

        return img, class_id, img_path


    def train_img_transforms(self):

        trans = []
        trans.append(SquarePad())
        trans.append(transforms.Resize(size=cfg.DATASET.IMG_SIZE))
        #trans.append(transforms.RandomCrop(size=224, padding=0))
        #trans.append(transforms.RandomAffine(degrees=20, translate=(0.2,0.2), scale=(0.2,1.5), shear=0.2))
        trans.append(transforms.RandomHorizontalFlip(p=0.3))
        trans.append(transforms.RandomRotation(degrees=30))
        trans.append(transforms.GaussianBlur(kernel_size=3, sigma=1.0))
        trans.append(transforms.RandomVerticalFlip())
        trans.append(transforms.ToTensor())
        trans.append(transforms.Normalize(mean=cfg.DATASET.IMG_MEAN, std=cfg.DATASET.IMG_STD))
        trans.append(transforms.RandomErasing(p=0.5))

        trans = transforms.Compose(trans)

        return trans

    def test_img_transforms(self):

        trans = []
        trans.append(SquarePad())
        trans.append(transforms.Resize(size=cfg.DATASET.IMG_SIZE))
        #trans.append(transforms.CenterCrop(size=224))
        #trans.append(transforms.RandomHorizontalFlip())
        trans.append(transforms.ToTensor())
        trans.append(transforms.Normalize(mean=cfg.DATASET.IMG_MEAN, std=cfg.DATASET.IMG_STD))
        #trans.append(transforms.RandomErasing(p=self.random_erasing_prob))
        trans = transforms.Compose(trans)

        return trans


class LoadNoisyDataset(data.Dataset):
    def __init__(self, X, y, blur_kernel_size=1.0, rotation=0, random_erasing_prob=0.0, gaussian_noise_prob=0.0):

        self.image_paths, self.class_ids = X, y
        self.blur_kernel_size = blur_kernel_size
        self.rotation = rotation
        self.random_erasing_prob = random_erasing_prob
        self.gaussian_noise_prob = gaussian_noise_prob

    def __len__(self):

        return len(self.image_paths)

    def __getitem__(self, idx):

        img_path = self.image_paths[idx]
        class_id = self.class_ids[idx]
        img = Image.open(img_path).convert('RGB')
        image_transform = self.noisy_img_transforms()
        img = image_transform(img)

        return img, class_id, img_path


    def noisy_img_transforms(self):

        trans = []
        trans.append(SquarePad())
        trans.append(transforms.Resize(size=cfg.DATASET.IMG_SIZE))
        trans.append(transforms.GaussianBlur(kernel_size=self.blur_kernel_size, sigma=1.0))
        trans.append(transforms.RandomRotation(self.rotation))
        trans.append(transforms.ToTensor())
        trans.append(transforms.Normalize(mean=cfg.DATASET.IMG_MEAN, std=cfg.DATASET.IMG_STD))
        trans.append(transforms.RandomErasing(p=self.random_erasing_prob))
        trans.append(transforms.RandomApply([AddGaussianNoise(0., 1.)], p=self.gaussian_noise_prob))
        trans = transforms.Compose(trans)

        return trans

class SquarePad:
	def __call__(self, image):
		w, h = image.size
		max_wh = np.max([w, h])
		hp = int((max_wh - w) / 2)
		vp = int((max_wh - h) / 2)
		padding = (hp, vp, hp, vp)
		return F.pad(image, padding, 0, 'constant')


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)