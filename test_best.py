#!/usr/bin/env python3
"""
Validate a trained model checkpoint (best.pth) on the test split.
Usage:
    python test_best.py \
        --config-file configs/repvit/cadlgi.yaml \
        --weight output/path/to/best_model.pth

The script reuses the project data-loading / evaluation utilities so results are
fully comparable with training-time metrics.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path

from src.configs.config import get_cfg
from src.data import loader as data_loader
from src.engine.evaluator import Evaluator
from src.engine.trainer import Trainer
from src.models.build_model import build_model
from src.utils import logging

logger = logging.get_logger("test_best")

def build_test_loader(cfg):
    """Construct only the test loader."""
    logger.info("Constructing test loader…")
    return data_loader.construct_test_loader(cfg)


def main():
    parser = argparse.ArgumentParser(description="Evaluate best checkpoint on test set")
    parser.add_argument("--config-file", required=True, help="Experiment YAML used for training")
    parser.add_argument("--weight", required=True, help="Path to best_model.pth")
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="Optional cfg opts")
    args = parser.parse_args()

    # 1. Build cfg
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    # Allow command-line overrides (e.g. different BATCH_SIZE for testing)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.MODEL.WEIGHT_PATH = args.weight
    run_id = datetime.now(timezone.utc).strftime('final_test_%Y%m%dT%H%M%S_%fZ')
    cfg.OUTPUT_DIR = str(Path(args.weight).resolve().parent / run_id)
    Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=False)
    Path(cfg.OUTPUT_DIR, 'config.yaml').write_text(cfg.dump(), encoding='utf-8')
    cfg.freeze()

    # 2. Build model & load weights
    model, device = build_model(cfg)
    # Trainer loads once, including migration and strict key validation.
    model.eval()

    # 3. Data loader & evaluator
    test_loader = build_test_loader(cfg)
    evaluator = Evaluator()
    trainer = Trainer(cfg, model, evaluator, device)
    trainer.validate_checkpoint_protocol(test_loader.dataset)

    # 4. Run evaluation
    results = trainer.run_final_test_evaluation(test_loader, checkpoint_path=args.weight)
    # Print the final metrics.
    print("\n=== Test Results ===\n", results, flush=True)
    print(f"Predictions and protocol saved to: {cfg.OUTPUT_DIR}", flush=True)
    logger.info("Evaluation completed. Full results: %s", results)


if __name__ == "__main__":
    main()
