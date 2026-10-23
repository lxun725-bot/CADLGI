#!/usr/bin/env python3
"""
Model construction functions for CADLGI.
"""
import torch

from .resnet import ResNet
from .convnext import ConvNeXt
from .vit_models import ViT, Swin, CADLGISwin
from .cadlgi_rep import build_cadlgi_rep, build_pure_repvit

from ..utils import logging

logger = logging.get_logger("cadlgi")

# Supported model types
_MODEL_TYPES = {
    "resnet": ResNet,
    "convnext": ConvNeXt,
    "vit": ViT,
    "swin": Swin,
    "cadlgi_swin": CADLGISwin,        # Swin-T + CADLGI (CADPA + LGI)
    "cadlgi_rep": "cadlgi_rep",           # RepViT with the CADLGI block.
    "pure_repvit": "pure_repvit",     # RepViT classification baseline.
}


def build_model(cfg):
    """
    Build model based on config.
    """
    assert (
        cfg.MODEL.TYPE in _MODEL_TYPES.keys()
    ), "Model type '{}' not supported. Available: {}".format(cfg.MODEL.TYPE, list(_MODEL_TYPES.keys()))

    # A complete task checkpoint already contains backbone weights.
    if cfg.MODEL.WEIGHT_PATH:
        cfg = cfg.clone()
        cfg.defrost()
        cfg.MODEL.CADLGI.USE_PRETRAINED = False
        cfg.MODEL.MODEL_ROOT = ""
        cfg.freeze()

    # Construct the model
    train_type = cfg.MODEL.TYPE
    if train_type in {'cadlgi_rep', 'pure_repvit'} and cfg.MODEL.CADLGI.DROP_PATH_RATE != 0:
        raise ValueError('This RepViT backbone has no stochastic depth; DROP_PATH_RATE must be 0')

    # Special handling for CADLGIRep and PureRepViT
    if train_type == "cadlgi_rep":
        model = build_cadlgi_rep(cfg)
    elif train_type == "pure_repvit":
        model = build_pure_repvit(cfg)
    else:
        model = _MODEL_TYPES[train_type](cfg)

    if hasattr(model, 'cadlgi') and hasattr(model.cadlgi, 'algf'):
        fusion = model.cadlgi.algf
        if hasattr(fusion, 'configure_gate_mode'):
            mode = cfg.MODEL.CADLGI.ALGF_GATE_MODE
            if mode == 'joint' and not cfg.MODEL.CADLGI.USE_SPATIAL_ADAPTIVE:
                mode = 'sample'
            fusion.configure_gate_mode(mode)
    log_model_info(model, verbose=cfg.DBG)
    model, device = load_model_to_device(model, cfg)
    logger.info(f"Device used for model: {device}")

    return model, device


def log_model_info(model, verbose=False):
    """Logs model info"""
    if verbose:
        logger.info(f"Classification Model:\n{model}")
        for k, p in model.named_parameters():
            if p.requires_grad:
                logger.info("module {} {} requires gard".format(k, p.shape))
    model_total_params = sum(p.numel() for p in model.parameters())
    model_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Total Parameters: {0}\t Gradient Parameters: {1}".format( model_total_params, model_grad_params))
    logger.info("tuned percent:%.3f"%(model_grad_params/model_total_params*100))

def get_current_device():
    if torch.cuda.is_available():
        # Determine the GPU used by the current process
        cur_device = torch.device('cuda', torch.cuda.current_device())
    else:
        cur_device = torch.device('cpu')
    return cur_device


def load_model_to_device(model, cfg):
    cur_device = get_current_device()
    if torch.cuda.is_available():
        # Transfer the model to the current GPU device
        model = model.cuda(device=cur_device)
        # Use multi-process data parallel model in the multi-gpu setting
        if cfg.NUM_GPUS > 1:
            # Make model replica operate on the current device
            model = torch.nn.parallel.DistributedDataParallel(
                module=model, device_ids=[cur_device.index], output_device=cur_device.index,
                find_unused_parameters=True,
            )
    else:
        model = model.to(cur_device)
    return model, cur_device
