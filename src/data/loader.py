#!/usr/bin/env python3

"""Data loader."""
import torch
import random
import numpy as np
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

from ..utils import logging
from .datasets.json_dataset import BarleyDataset
logger = logging.get_logger("cadlgi")
_DATASET_CATALOG = {"barley": BarleyDataset}


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def _construct_loader(cfg, split, batch_size, shuffle, drop_last):
    """Constructs the data loader for the given dataset."""
    dataset_name = cfg.DATA.NAME

    if dataset_name not in _DATASET_CATALOG:
        raise ValueError(f"Unsupported dataset: {dataset_name}; expected barley")
    dataset = _DATASET_CATALOG[dataset_name](cfg, split)

    # Create a sampler for multi-process training
    sampler = DistributedSampler(dataset) if cfg.NUM_GPUS > 1 else None
    generator = torch.Generator()
    generator.manual_seed(int(cfg.SEED or 0) + {'train': 0, 'val': 1, 'test': 2}[split])
    # Sampler/worker RNG is independent of model initialization.
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(False if sampler else shuffle),
        sampler=sampler,
        num_workers=cfg.DATA.NUM_WORKERS,
        pin_memory=cfg.DATA.PIN_MEMORY,
        drop_last=drop_last,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    return loader


def construct_train_loader(cfg):
    """Train loader wrapper."""
    if cfg.NUM_GPUS > 1:
        drop_last = True
    else:
        drop_last = False
    return _construct_loader(
        cfg=cfg,
        split="train",
        batch_size=int(cfg.DATA.BATCH_SIZE / cfg.NUM_GPUS),
        shuffle=True,
        drop_last=drop_last,
    )




def construct_test_loader(cfg):
    """Test loader wrapper."""
    return _construct_loader(
        cfg=cfg,
        split="test",
        batch_size=int(cfg.DATA.BATCH_SIZE / cfg.NUM_GPUS),
        shuffle=False,
        drop_last=False,
    )


def construct_val_loader(cfg, batch_size=None):
    if batch_size is None:
        bs = int(cfg.DATA.BATCH_SIZE / cfg.NUM_GPUS)
    else:
        bs = batch_size
    """Validation loader wrapper."""
    return _construct_loader(
        cfg=cfg,
        split="val",
        batch_size=bs,
        shuffle=False,
        drop_last=False,
    )


def shuffle(loader, cur_epoch):
    """"Shuffles the data."""
    assert isinstance(
        loader.sampler, (RandomSampler, DistributedSampler)
    ), "Sampler type '{}' not supported".format(type(loader.sampler))
    # RandomSampler handles shuffling automatically
    if isinstance(loader.sampler, DistributedSampler):
        # DistributedSampler shuffles data based on epoch
        loader.sampler.set_epoch(cur_epoch)


def construct_train_eval_loader(cfg):
    """Same training IDs, deterministic validation transforms; never test images."""
    from .transforms import get_transforms
    loader = _construct_loader(cfg, 'train', cfg.DATA.BATCH_SIZE, False, False)
    loader.dataset.transform = get_transforms('val', cfg.DATA.CROPSIZE)
    return loader
