

#!/usr/bin/env python3
"""
major actions here: fine-tune the features and evaluate different settings
"""
import os
import torch
import warnings

import numpy as np
import random
from datetime import datetime, timezone

from time import sleep
from random import randint

import src.utils.logging as logging
from src.configs.config import get_cfg
from src.data import loader as data_loader
from src.engine.evaluator import Evaluator
from src.engine.trainer import Trainer
from src.models.build_model import build_model
from src.utils.file_io import PathManager

from launch import default_argument_parser, logging_train_setup


def setup(args):
    """
    Create configs and perform basic setups.
    """
    # Load the default configuration.
    cfg = get_cfg()
    # Merge the configuration file.
    cfg.merge_from_file(args.config_file)
    # Apply command-line overrides.
    cfg.merge_from_list(args.opts)
    if args.eval and not cfg.MODEL.WEIGHT_PATH:
        raise ValueError('Evaluation requires a frozen model checkpoint: MODEL.WEIGHT_PATH')
    if args.eval and args.eval_dataset == 'test':
        cfg.DATA.NO_TEST = False

    # setup output dir
    output_dir = cfg.OUTPUT_DIR
    data_name = cfg.DATA.NAME
    lr = cfg.SOLVER.BASE_LR
    wd = cfg.SOLVER.WEIGHT_DECAY

    if cfg.MODEL.TRANSFER_TYPE not in {"end2end", "linear"}:
        raise ValueError("Only end2end and linear transfer are retained")
    output_folder = os.path.join(cfg.MODEL.TRANSFER_TYPE, cfg.DATA.FEATURE,
                                f"seed{cfg.SEED}_lr{lr}_wd{wd}")

    # A new session must never pick up a stale best_model from an earlier run.
    run_id = datetime.now(timezone.utc).strftime('run_%Y%m%dT%H%M%S_%fZ')
    output_path = os.path.join(output_dir, data_name, output_folder, run_id)
    cfg.OUTPUT_DIR = output_path

    if not PathManager.exists(output_path):
        PathManager.mkdirs(output_path)

    cfg.freeze()
    with open(os.path.join(output_path, 'config.yaml'), 'w', encoding='utf-8') as stream:
        stream.write(cfg.dump())
    return cfg


def get_loaders(cfg, logger):
    logger.info("Loading training data...")
    train_loader = data_loader.construct_train_loader(cfg)

    # Load validation data.
    logger.info("Loading validation data...")
    val_loader = data_loader.construct_val_loader(cfg)
    logger.info("Loading test data...")
    if cfg.DATA.NO_TEST:
        logger.info("...no test data is constructed")
        test_loader = None
    else:
        test_loader = data_loader.construct_test_loader(cfg)
    return train_loader,  val_loader, test_loader


def train(cfg, args):
    # Clear cached memory only when explicitly enabled.
    if torch.cuda.is_available():
        if os.environ.get("CADLGI_SKIP_EMPTY_CACHE", "1") != "1":
            torch.cuda.empty_cache()

    # main training / eval actions here

    # fix the seed for reproducibility
    if cfg.SEED is not None:
        torch.manual_seed(cfg.SEED)
        np.random.seed(cfg.SEED)
        random.seed(cfg.SEED)

    # setup training env including loggers
    logging_train_setup(args, cfg)
    logger = logging.get_logger((cfg['DATA']['NAME']))

    train_loader, val_loader, test_loader = get_loaders(cfg, logger)
    logger.info("Constructing models...")
    model, cur_device = build_model(cfg)

    logger.info("Setting up Evalutator...")
    evaluator = Evaluator()
    evaluator.compute_f1 = args.compute_f1
    logger.info("Setting up Trainer...")
    trainer = Trainer(cfg, model, evaluator, cur_device)

    if args.eval:
        trainer.cls_weights = train_loader.dataset.get_class_weights(trainer.cfg.DATA.CLASS_WEIGHTS_TYPE)
        trainer.model.eval()
        if args.eval_dataset in ["test","val","train"]:
            logger.info(f'Testing the {args.eval_dataset} dataset...')
            selected_loader = {"train": train_loader, "val": val_loader, "test": test_loader}[args.eval_dataset]
            if selected_loader is None:
                raise ValueError('Requested evaluation split is disabled')
            trainer.validate_checkpoint_protocol(selected_loader.dataset)
            if args.eval_dataset == 'test':
                trainer.run_final_test_evaluation(selected_loader, checkpoint_path=cfg.MODEL.WEIGHT_PATH)
            else:
                trainer.eval_classifier(selected_loader, args.eval_dataset, False)
        else:
            logger.info(f"No {args.eval_dataset} loader presented. Exit")
    else:
        if train_loader:
            trainer.train_classifier(train_loader, val_loader, test_loader,
                                     train_eval_loader=data_loader.construct_train_eval_loader(cfg))
        else:
            logger.info("No train loader presented. Exit")


def main(args):
    """main function to call from workflow"""

    # set up cfg and args
    cfg = setup(args)

    # Perform training.
    train(cfg, args)


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    main(args)
