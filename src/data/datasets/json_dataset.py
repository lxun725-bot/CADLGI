#!/usr/bin/env python3

"""Barley image dataset with explicit train/val/test JSON indices"""

import os
import hashlib
import json
import torch
import torch.utils.data
import torchvision as tv
import numpy as np
from collections import Counter

from ..transforms import get_transforms, get_transforms_strong_augmentation
from ...utils import logging
from ...utils.io_utils import read_json
logger = logging.get_logger("cadlgi")


class JSONDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, split):
        assert split in {
            "train",
            "val",
            "test",
        }, "Split '{}' not supported for {} dataset".format(
            split, cfg.DATA.NAME)
        logger.info("Constructing {} dataset {}...".format(
            cfg.DATA.NAME, split))

        self.cfg = cfg
        self._split = split
        self.name = cfg.DATA.NAME
        self.data_dir = cfg.DATA.DATAPATH
        self.index_dir = cfg.DATA.INDEX_DIR or self.data_dir
        mapping_path = os.path.join(self.index_dir, 'category2index.json')
        mapping = read_json(mapping_path)
        if sorted(mapping.values()) != list(range(cfg.DATA.NUMBER_CLASSES)):
            raise ValueError('category2index.json must map all configured classes to contiguous IDs')
        self.class_mapping = mapping
        self.classes = [name for name, label in sorted(mapping.items(), key=lambda item: item[1])]
        self.class_names = self.classes
        self.data_percentage = cfg.DATA.PERCENTAGE
        # Bind all three split indices and labels to task checkpoints.
        digest = hashlib.sha256(json.dumps(mapping, sort_keys=True).encode())
        for name in ('train', 'val', 'test'):
            if name == 'train' and self.data_percentage < 1.0:
                name = f'train_{self.data_percentage}'
            path = os.path.join(self.index_dir, name + '.json')
            with open(path, 'rb') as stream:
                digest.update(name.encode()); digest.update(stream.read())
        self.protocol_fingerprint = digest.hexdigest()
        self._construct_imdb(cfg)
        use_strong_aug = cfg.DATA.USE_STRONG_AUGMENTATION
        if use_strong_aug and split == "train":
            self.transform = get_transforms_strong_augmentation(split, cfg.DATA.CROPSIZE)
        else:
            self.transform = get_transforms(split, cfg.DATA.CROPSIZE)

    def get_anno(self):
        anno_path = os.path.join(self.index_dir, "{}.json".format(self._split))
        if "train" in self._split:
            if self.data_percentage < 1.0:
                anno_path = os.path.join(
                    self.index_dir,
                    "{}_{}.json".format(self._split, self.data_percentage)
                )
        assert os.path.exists(anno_path), "{} dir not found".format(anno_path)

        return read_json(anno_path)

    def get_imagedir(self):
        raise NotImplementedError()

    def _construct_imdb(self, cfg):
        """Constructs the imdb."""

        img_dir = self.get_imagedir()
        assert os.path.exists(img_dir), "{} dir not found".format(img_dir)

        anno = self.get_anno()
        # Enforce a consistent, global class indexing across splits
        # Assume JSON already stores labels as 0..NUM_CLASSES-1
        num_classes = cfg.DATA.NUMBER_CLASSES
        self._class_ids = list(range(num_classes))

        # Construct the image db without remapping
        self._imdb = []
        for img_name, cls_id in anno.items():
            # robustify label
            try:
                label = int(cls_id)
            except Exception:
                raise ValueError(f"Invalid class id '{cls_id}' for image '{img_name}'")
            if not (0 <= label < num_classes):
                raise ValueError(
                    f"Class id {label} out of range [0, {num_classes-1}] for image '{img_name}'.\n"
                    f"Please ensure JSON labels are globally indexed and consistent across splits.")
            im_path = os.path.join(img_dir, img_name)
            resolved = os.path.abspath(im_path)
            if os.path.commonpath([os.path.abspath(img_dir), resolved]) != os.path.abspath(img_dir):
                raise ValueError(f'Image path escapes dataset root: {img_name}')
            if not os.path.isfile(resolved):
                raise FileNotFoundError(resolved)
            category = os.path.basename(os.path.dirname(resolved))
            if self.class_mapping.get(category) != label:
                raise ValueError(f'Index label does not match class directory: {img_name}')
            self._imdb.append({"im_path": im_path, "class": label, 'id': img_name.replace('\\', '/')})

        logger.info("Number of images: {}".format(len(self._imdb)))
        logger.info("Number of classes: {}".format(len(self._class_ids)))

    def get_info(self):
        num_imgs = len(self._imdb)
        return num_imgs, self.get_class_num()

    def get_class_num(self):
        return self.cfg.DATA.NUMBER_CLASSES
        # return len(self._class_ids)

    def get_class_weights(self, weight_type):
        """get a list of class weight, return a list float"""
        if "train" not in self._split:
            raise ValueError(
                "only getting training class distribution, " +
                "got split {} instead".format(self._split)
            )

        cls_num = self.get_class_num()
        if weight_type == "none":
            return [1.0] * cls_num

        id2counts = Counter(item['class'] for item in self._imdb)
        assert len(id2counts) == cls_num
        num_per_cls = np.array([id2counts[i] for i in self._class_ids])
        if np.any(num_per_cls == 0):
            raise ValueError('Cannot compute inverse weights when a training class is absent')

        if weight_type == 'inv':
            mu = -1.0
        elif weight_type == 'inv_sqrt':
            mu = -0.5
        weight_list = num_per_cls ** mu
        weight_list = np.divide(
            weight_list, np.linalg.norm(weight_list, 1)) * cls_num
        return weight_list.tolist()

    def __getitem__(self, index):
        # Load the image
        im = tv.datasets.folder.default_loader(self._imdb[index]["im_path"])
        label = self._imdb[index]["class"]
        im = self.transform(im)
        sample = {
            "image": im,
            "label": label,
            'id': self._imdb[index]["id"],
        }
        return sample

    def __len__(self):
        return len(self._imdb)


class BarleyDataset(JSONDataset):
    """barley dataset"""

    def __init__(self, cfg, split):
        super(BarleyDataset, self).__init__(cfg, split)

    def get_imagedir(self):
        return self.data_dir
