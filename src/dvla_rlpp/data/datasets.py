"""Dataset builders.

Expected layout (see ``data/README.md``)::

    data/datasets/<dataset>/base/<class>/*.jpg     (training categories)
    data/datasets/<dataset>/val/<class>/*.jpg      (validation categories)
    data/datasets/<dataset>/novel/<class>/*.jpg    (test categories)
"""
import os
import os.path as osp

import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from .. import utils

GENERAL = ("miniImageNet", "tieredImageNet", "CIFAR-FS", "FC100")
FINE_GRAINED = ("FG-CUB", "FG-Cars", "FG-Dogs")
CROSS_DOMAIN = ("CD-CUB", "Places", "ChestX", "CD-Cars", "CropDiseases", "EuroSAT", "ISIC", "Plantae")
ALL_DATASETS = GENERAL + FINE_GRAINED + CROSS_DOMAIN


def dataset_root(data_root: str, dataset: str) -> str:
    return osp.join(data_root, dataset)


def class_display_name(dataset: str, folder_name: str) -> str:
    """Map an ImageFolder class directory to the class name used for semantics."""
    if dataset in ("FG-CUB", "CD-CUB") and "." in folder_name:
        return folder_name.split(".", 1)[-1]
    if dataset == "FG-Dogs" and "-" in folder_name:
        return folder_name.split("-", 1)[-1]
    return folder_name


class FewShotImageFolder(ImageFolder):
    """ImageFolder that also returns the sample index so that support image paths
    can be recovered (semantic caches are keyed by support identifiers)."""

    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        self.root_path = root

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        return image, label, index

    def relative_path(self, index: int) -> str:
        return osp.relpath(self.samples[index][0], self.root_path)

    def idx_to_name(self, dataset: str):
        return {idx: class_display_name(dataset, name) for name, idx in self.class_to_idx.items()}


def build_episodic_datasets(dataset: str, data_root: str, image_size: int = 224,
                            eval_split: str = "novel", train_split: str = "base"):
    """Return (train_dataset, eval_dataset) for episodic meta-tuning / testing.

    Cross-domain datasets are evaluated with a model trained on miniImageNet, so
    their ``train_split`` only provides the base class list of the source domain.
    """
    root = dataset_root(data_root, dataset)
    train_dir = osp.join(root, train_split)
    eval_dir = osp.join(root, eval_split)
    for d in (train_dir, eval_dir):
        if not osp.isdir(d):
            raise FileNotFoundError(f"Missing dataset split '{d}'. See data/README.md.")
    train_set = FewShotImageFolder(train_dir, transform=utils.build_transform(dataset, "train", image_size))
    eval_set = FewShotImageFolder(eval_dir, transform=utils.build_transform(dataset, "eval", image_size))
    return train_set, eval_set


def _timm_train_transform(image_size: int):
    from timm.data import create_transform
    return create_transform(input_size=image_size, is_training=True, color_jitter=0.4,
                            auto_augment="rand-m9-mstd0.5-inc1", interpolation="bicubic",
                            re_prob=0.25, re_mode="pixel", re_count=1)


class PretrainDataset(Dataset):
    """Base-class dataset for backbone pre-training (whole-image classification)."""

    def __init__(self, dataset: str, data_root: str, split: str = "train", image_size: int = 224,
                 augment: bool = True):
        split_dir = {"train": "base", "val": "val", "test": "novel"}[split]
        path = osp.join(dataset_root(data_root, dataset), split_dir)
        if dataset in ("CIFAR-FS", "FC100"):
            mean, std = utils.CIFAR_MEAN, utils.CIFAR_STD
        else:
            mean, std = utils.IMAGENET_MEAN, utils.IMAGENET_STD
        norm = transforms.Normalize(mean, std)
        if split == "train" and augment:
            self.transform = _timm_train_transform(image_size)
        elif split == "train":
            self.transform = transforms.Compose([transforms.Resize(image_size), transforms.CenterCrop(image_size),
                                                 transforms.RandomHorizontalFlip(), transforms.ToTensor(), norm])
        else:
            self.transform = transforms.Compose([transforms.Resize(int(image_size * 1.1)),
                                                 transforms.CenterCrop(image_size), transforms.ToTensor(), norm])
        self.dataset = torchvision.datasets.ImageFolder(path, self.transform)
        self.norm_params = {"mean": mean, "std": std}

    def __getitem__(self, i):
        return self.dataset[i]

    def __len__(self):
        return len(self.dataset)
