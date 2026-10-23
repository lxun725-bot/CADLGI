#!/usr/bin/env python3
"""
 a trainer class
"""
import datetime
import time
import os
import torch
from torch.cuda import amp
import torch.nn as nn
from pathlib import Path
import pandas as pd
import numpy as np
import json
import hashlib
import platform
from importlib.metadata import version
from tqdm.auto import tqdm
from collections import OrderedDict

from fvcore.common.config import CfgNode
from fvcore.common.checkpoint import Checkpointer

from ..engine.evaluator import Evaluator
from ..engine.comprehensive_evaluator import ComprehensiveEvaluator
from ..solver.lr_scheduler import make_scheduler
from ..solver.optimizer import make_optimizer
from ..solver.losses import build_loss
from ..utils import logging
from ..utils.checkpoint import load_model_checkpoint, IMPLEMENTATION_VERSION, get_model_contract, get_source_fingerprint
from ..utils.mechanism_stats import collect_mechanism_stats
from ..utils.train_utils import AverageMeter, gpu_mem_usage

logger = logging.get_logger("cadlgi")

GLOBAL_RESULTS_CSV = "results/all_experiments.csv"

def get_gpu_memory_info():
    """Return the allocated and reserved GPU memory."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        free = total - reserved
        return allocated, reserved, total, free
    return 0, 0, 0, 0


class Trainer():
    """
    a trainer with below logics:

    1. Build optimizer, scheduler
    2. Load checkpoints if provided
    3. Train and eval at each epoch
    """
    def __init__(
        self,
        cfg: CfgNode,
        model: nn.Module,
        evaluator: Evaluator,
        device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.device = device
        # Enable performance features
        try:
            torch.backends.cudnn.benchmark = cfg.SOLVER.CUDNN_BENCHMARK
            torch.backends.cuda.matmul.allow_tf32 = cfg.SOLVER.TF32
            torch.backends.cudnn.allow_tf32 = cfg.SOLVER.TF32
        except Exception:
            pass

        # Mixed precision training to reduce memory and improve throughput
        self.use_amp = bool(cfg.SOLVER.AMP and device.type == 'cuda')
        self.grad_scaler = amp.GradScaler(enabled=self.use_amp)

        # solver related
        logger.info("\tSetting up the optimizer...")
        self.optimizer = make_optimizer([self.model], cfg.SOLVER)
        self.scheduler = make_scheduler(self.optimizer, cfg.SOLVER)
        self.cls_criterion = build_loss(self.cfg)

        self.checkpointer = Checkpointer(
            self.model,
            save_dir=cfg.OUTPUT_DIR,
            save_to_disk=True
        )

        self.loaded_checkpoint = None
        if len(cfg.MODEL.WEIGHT_PATH) > 0:
            # Checkpoint loading includes legacy paper-module key migration
            self.loaded_checkpoint = load_model_checkpoint(self.model, cfg.MODEL.WEIGHT_PATH,
                                  require_current=not cfg.MODEL.ALLOW_LEGACY_CHECKPOINT)
            self.model.pretrained_info = {
                'mode': 'task_checkpoint', 'path': str(Path(cfg.MODEL.WEIGHT_PATH).resolve()),
                'sha256': hashlib.sha256(Path(cfg.MODEL.WEIGHT_PATH).read_bytes()).hexdigest(),
                'source_implementation': self.loaded_checkpoint.get('implementation_version'),
                'source_initialization': self.loaded_checkpoint.get('initialization'),
            }
            logger.info(f"Model weight loaded from {cfg.MODEL.WEIGHT_PATH}")

        self.evaluator = evaluator
        self.cpu_device = torch.device("cpu")
        self._test_evaluated = False
        self.selection = None
        self.model_contract = get_model_contract(cfg)

        # prepare metrics csv (lazy create on first write)
        self.metrics_csv = Path(self.cfg.OUTPUT_DIR) / "metrics.csv"
        self.metrics_csv.parent.mkdir(parents=True, exist_ok=True)
        self.runtime_protocol = {
            'implementation_version': IMPLEMENTATION_VERSION, 'code_sha256': get_source_fingerprint(),
            'python': platform.python_version(), 'torch': str(torch.__version__),
            'torchvision': version('torchvision'), 'timm': version('timm'),
            'device': str(device), 'seed': cfg.SEED, 'amp_enabled': self.use_amp,
            'tf32_requested': cfg.SOLVER.TF32, 'cudnn_benchmark': cfg.SOLVER.CUDNN_BENCHMARK,
            'optimizer': type(self.optimizer).__name__,
            'optimizer_groups': [{k: v for k, v in group.items() if k != 'params'} | {
                'parameter_tensors': len(group['params']),
                'parameter_count': sum(p.numel() for p in group['params'])}
                for group in self.optimizer.param_groups],
            'initialization': getattr(self.model, 'pretrained_info', None),
            'config': cfg.dump(),
        }
        with open(Path(cfg.OUTPUT_DIR, 'run_protocol.json'), 'w', encoding='utf-8') as stream:
            json.dump(self.runtime_protocol, stream, ensure_ascii=False, indent=2)

    def validate_checkpoint_protocol(self, dataset):
        if self.loaded_checkpoint is None:
            raise ValueError('Evaluation requires an explicitly loaded checkpoint')
        selection = self.loaded_checkpoint.get('selection', {})
        if self.loaded_checkpoint.get('implementation_version') != IMPLEMENTATION_VERSION:
            raise ValueError('Legacy weights are for warm starts only; corrected evaluation requires retraining')
        if selection.get('split') != 'val':
            raise ValueError('Evaluation requires a validation-selected checkpoint')
        if selection.get('protocol_fingerprint') != dataset.protocol_fingerprint:
            raise ValueError('Dataset indices differ from the checkpoint protocol')
        if selection.get('class_mapping') != dataset.class_mapping:
            raise ValueError('Dataset class mapping differs from the checkpoint protocol')
        if self.loaded_checkpoint.get('model_contract') != self.model_contract:
            raise ValueError('Evaluation model configuration differs from the selected checkpoint')

    def _append_metrics_row(self, row: dict) -> None:
        """Append a row of metrics to CSV (create header if empty)."""
        base_df = None
        if self.metrics_csv.exists() and self.metrics_csv.stat().st_size > 0:
            base_df = pd.read_csv(self.metrics_csv)
        out_df = pd.DataFrame([row]) if base_df is None else pd.concat(
            [base_df, pd.DataFrame([row])], axis=0, ignore_index=True)
        temporary = self.metrics_csv.with_suffix('.csv.tmp')
        out_df.to_csv(temporary, index=False)
        temporary.replace(self.metrics_csv)

    def forward_one_batch(self, inputs, targets, is_train):
        """Run one batch and return its loss and predictions."""
        # move data to device
        inputs = inputs.to(self.device, non_blocking=True)    # (batchsize, 2048)
        targets = targets.to(self.device, non_blocking=True)  # (batchsize, )

        if self.cfg.DBG:
            logger.info(f"shape of inputs: {inputs.shape}")
            logger.info(f"shape of targets: {targets.shape}")

        # forward
        with torch.set_grad_enabled(is_train):
            # Use the same configured loss for all backbones, ablations and splits.
            with amp.autocast(enabled=self.use_amp):
                outputs = self.model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                loss = self.cls_criterion(outputs, targets, self.cls_weights)

            if not torch.isfinite(outputs).all() or not torch.isfinite(loss).all():
                raise FloatingPointError('Non-finite logits or loss detected; stopping training.')

        # =======backward and optim step only if in training phase... =========
        if is_train:
            self.optimizer.zero_grad()
            if self.use_amp:
                self.grad_scaler.scale(loss).backward()
                # Unscale gradients and clip to prevent explosion
                self.grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                old_scale = self.grad_scaler.get_scale()
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                if self.grad_scaler.get_scale() < old_scale:
                    logger.warning('AMP overflow: optimizer step skipped, loss scale reduced from %s to %s',
                                   old_scale, self.grad_scaler.get_scale())
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, error_if_nonfinite=True)
                self.optimizer.step()

        return loss, outputs

    def get_input(self, data):
        if not isinstance(data["image"], torch.Tensor):
            for k, v in data.items():
                data[k] = torch.from_numpy(v)

        inputs = data["image"].float()
        labels = data["label"]
        return inputs, labels

    def train_classifier(self, train_loader, val_loader, test_loader, train_eval_loader=None):
        """
        Train a classifier using epoch
        """

        self.model.eval()

        if val_loader is None:
            raise ValueError('Training requires a validation split for checkpoint selection')
        if (train_loader.dataset.protocol_fingerprint != val_loader.dataset.protocol_fingerprint
                or train_loader.dataset.class_mapping != val_loader.dataset.class_mapping):
            raise ValueError('Training and validation use different dataset protocols')

        # setup training epoch params
        total_epoch = self.cfg.SOLVER.TOTAL_EPOCH
        total_data = len(train_loader)
        best_epoch = -1
        best_metric = float('-inf')
        log_interval = self.cfg.SOLVER.LOG_EVERY_N

        losses = AverageMeter('Loss', ':.4e')
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')

        self.cls_weights = train_loader.dataset.get_class_weights(
            self.cfg.DATA.CLASS_WEIGHTS_TYPE)
        # logger.info(f"class weights: {self.cls_weights}")
        patience = 0  # if > self.cfg.SOLVER.PATIENCE, stop training

        for epoch in range(total_epoch):
            # reset averagemeters to measure per-epoch results
            losses.reset()
            batch_time.reset()
            data_time.reset()

            lr = self.scheduler.get_last_lr()[0]
            logger.info(
                "Training {} / {} epoch, with learning rate {}".format(
                    epoch + 1, total_epoch, lr
                )
            )

            # Enable training mode
            self.model.train()

            end = time.time()

            show_progress = os.environ.get("BATCH_TQDM_DISABLE", "0") != "1"
            task_id = int(os.environ.get("TASK_ID", "0"))
            with tqdm(total=total_data, desc=f"Epoch {epoch+1}/{total_epoch}", unit="batch",
                     disable=not show_progress,
                     dynamic_ncols=True,
                     leave=True,
                     position=task_id) as pbar:
                for idx, input_data in enumerate(train_loader):
                    if self.cfg.DBG and idx == 20:
                        # if debugging, only need to see the first few iterations
                        break
                    X, targets = self.get_input(input_data)
                    # measure data loading time
                    data_time.update(time.time() - end)

                    train_loss, _ = self.forward_one_batch(X, targets, True)

                    if train_loss == -1:
                        # continue
                        return None

                    losses.update(train_loss.item(), X.shape[0])

                    # measure elapsed time
                    batch_time.update(time.time() - end)
                    end = time.time()

                    allocated, reserved, total, free = get_gpu_memory_info()
                    pbar.set_postfix({
                        'train_loss': f"{losses.avg:.4f}",
                        'batch_time': f"{batch_time.val:.3f}s",
                        'data_time': f"{data_time.val:.3f}s",
                        'lr': f"{lr:.3e}",
                        'GPU_alloc': f"{allocated:.2f}GB",
                        'GPU_free': f"{free:.2f}GB"
                    })
                    pbar.update(1)

            allocated, reserved, total, free = get_gpu_memory_info()
            logger.info(
                "Epoch {} / {}: ".format(epoch + 1, total_epoch)
                + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                    data_time.avg, batch_time.avg)
                + "average train loss: {:.4f}, ".format(losses.avg)
                + "GPU: allocated={:.2f}GB, free={:.2f}GB, total={:.2f}GB".format(
                    allocated, free, total))
             # update lr, scheduler.step() must be called after optimizer.step() according to the docs: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate  # noqa
            self.scheduler.step()

            # Enable eval mode
            self.model.eval()

            train_name = "train_" + train_loader.dataset.name
            self.evaluator.update_iteration(epoch)
            train_loss_eval = self.eval_classifier(train_eval_loader if train_eval_loader is not None else train_loader, "train", False)
            train_metrics = {}
            try:
                train_metrics = self.evaluator.results[f"epoch_{epoch}"]["classification"][train_name]
            except KeyError:
                logger.warning(f"Could not get train metrics for epoch {epoch}")

            # eval at each epoch for single gpu training
            val_loss = self.eval_classifier(val_loader, "val", epoch == total_epoch - 1)

            # check the patience
            t_name = "val_" + val_loader.dataset.name
            try:
                curr_metrics = self.evaluator.results[f"epoch_{epoch}"]["classification"][t_name]
            except KeyError as exc:
                raise RuntimeError('Validation metrics missing; run cannot select a valid checkpoint') from exc

            # extract top-level metrics generically
            row = {
                'epoch': epoch + 1,
                'lr': lr,
                'train_loss': float(losses.avg),
                'val_loss': float(val_loss) if val_loss is not None else None,
            }
            for k, v in train_metrics.items():
                if isinstance(v, (float, int)):
                    row[f"train_{k}"] = float(v)
                else:
                    # nested (e.g., arrays), store as string
                    row[f"train_{k}"] = str(v)
            # flatten metrics (e.g., top1/top5 or mAP/mAR)
            for k, v in curr_metrics.items():
                if isinstance(v, (float, int)):
                    row[f"val_{k}"] = float(v)
                else:
                    # nested (e.g., arrays), store as string
                    row[f"val_{k}"] = str(v)

            # The metric is fixed in config before training, never chosen from test.
            metric_name = self.cfg.SOLVER.SELECTION_METRIC
            if metric_name not in curr_metrics:
                raise ValueError(f'Configured validation metric missing: {metric_name}')
            primary = float(curr_metrics[metric_name])
            if not np.isfinite(primary):
                raise ValueError('Validation selection metric is not finite')

            if primary > best_metric:
                best_metric = primary
                best_epoch = epoch + 1
                logger.info(f'Best epoch {best_epoch}: best val metric: {best_metric*100:.3f}')
                patience = 0
                self.selection = {'epoch': best_epoch, 'metric': metric_name, 'value': primary,
                                  'split': 'val', 'class_mapping': train_loader.dataset.class_mapping,
                                  'protocol_fingerprint': train_loader.dataset.protocol_fingerprint}
                # Always save the selected model, including a first zero-accuracy
                # epoch. SAVE_CKPT only controls the optional last checkpoint.
                self.checkpointer.save('best_model', implementation_version=IMPLEMENTATION_VERSION,
                                       selection=self.selection, config=self.cfg.dump(),
                                       initialization=getattr(self.model, 'pretrained_info', None),
                                       runtime_protocol=self.runtime_protocol, model_contract=self.model_contract)
                # mark best in csv
                row['is_best'] = True
            else:
                patience += 1
                row['is_best'] = False
            self._append_metrics_row(row)
            if patience >= self.cfg.SOLVER.PATIENCE:
                logger.info("No improvement. Breaking out of loop.")
                break

        # save the last checkpoints
        if self.cfg.MODEL.SAVE_CKPT:
            Checkpointer(
                self.model,
                save_dir=self.cfg.OUTPUT_DIR,
                save_to_disk=True
            ).save("last_model", implementation_version=IMPLEMENTATION_VERSION,
                   selection={'split': 'last', 'epoch': epoch + 1},
                   best_validation_selection=self.selection, config=self.cfg.dump(),
                   model_contract=self.model_contract)

        test_accuracy = None
        if test_loader is not None:
            results = self.run_final_test_evaluation(test_loader)
            if results and 'top1_accuracy' in results:
                test_accuracy = results['top1_accuracy']
                logger.info(f"Final test accuracy: {test_accuracy:.2f}%")

        return test_accuracy

    def run_final_test_evaluation(self, test_loader, checkpoint_path=None):
        """Evaluate the validation-selected checkpoint on the test split."""
        logger.info("\n" + "=" * 60)
        logger.info("Evaluating best_model.pth on the test split.")
        logger.info("=" * 60)

        if self._test_evaluated:
            raise RuntimeError('Final test inference has already run for this training session')
        if self.cfg.NUM_GPUS != 1:
            raise ValueError('Final test requires a single complete loader; distributed aggregation is not implemented')
        # 1. Strictly restore the validation-selected model.
        best_model_path = checkpoint_path or os.path.join(self.cfg.OUTPUT_DIR, "best_model.pth")
        if not os.path.exists(best_model_path):
            raise FileNotFoundError(best_model_path)
        checkpoint = load_model_checkpoint(self.model, best_model_path, require_current=True)
        selection = checkpoint.get('selection', {})
        if selection.get('split') != 'val':
            raise ValueError('Final test requires a validation-selected checkpoint')
        if selection.get('protocol_fingerprint') != test_loader.dataset.protocol_fingerprint:
            raise ValueError('Test indices do not match the selected checkpoint dataset protocol')
        if selection.get('class_mapping') != test_loader.dataset.class_mapping:
            raise ValueError('Test class mapping differs from the selected checkpoint')
        if checkpoint.get('model_contract') != self.model_contract:
            raise ValueError('Test model configuration differs from the selected checkpoint')
        self._test_evaluated = True
        logger.info(f"Restored validation-selected epoch {selection['epoch']}")

        self.model.eval()
        all_probs = []
        all_labels = []
        all_ids = []
        diagnostic_batches = []

        logger.info("Running test inference...")
        with torch.no_grad():
            for batch in test_loader:
                images = batch['image'].to(self.device)
                labels = batch['label']

                outputs = self.model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                if not torch.isfinite(outputs).all():
                    raise FloatingPointError('Non-finite logits during final test')
                probs = torch.softmax(outputs.float(), dim=1).cpu().numpy()
                all_probs.append(probs)
                diagnostic_batches.append(collect_mechanism_stats(self.model, len(labels)))
                all_labels.extend(labels.numpy().tolist())
                all_ids.extend(batch.get('id', [str(i) for i in range(len(all_ids), len(all_ids) + len(labels))]))

        all_probs = np.concatenate(all_probs, axis=0)
        all_labels = np.array(all_labels)
        if len(set(all_ids)) != len(all_ids):
            raise ValueError('Final test contains repeated image IDs')
        if hasattr(test_loader.dataset, '__len__') and len(all_labels) != len(test_loader.dataset):
            raise ValueError('Final test predictions do not cover the complete dataset')
        np.savez_compressed(os.path.join(self.cfg.OUTPUT_DIR, 'final_test_predictions.npz'),
                            probabilities=all_probs, labels=all_labels, image_ids=np.asarray(all_ids))
        diagnostics = {name: np.concatenate([batch[name] for batch in diagnostic_batches])
                       for name in diagnostic_batches[0]}
        rows = {'image_id': all_ids, 'label': all_labels, 'prediction': all_probs.argmax(1),
                **diagnostics, **{f'probability_class_{i}': all_probs[:, i] for i in range(all_probs.shape[1])}}
        pd.DataFrame(rows).to_csv(Path(self.cfg.OUTPUT_DIR, 'final_test_predictions.csv'), index=False)
        with open(os.path.join(self.cfg.OUTPUT_DIR, 'final_test_protocol.json'), 'w', encoding='utf-8') as stream:
            json.dump({'implementation_version': IMPLEMENTATION_VERSION, 'selection': selection,
                       'test_samples': len(all_labels), 'test_inference_passes': 1}, stream,
                      ensure_ascii=False, indent=2)

        class_names = None

        if hasattr(test_loader.dataset, 'classes'):
            class_names = test_loader.dataset.classes
        elif hasattr(test_loader.dataset, '_class_ids'):
            _class_ids = test_loader.dataset._class_ids
            if isinstance(_class_ids, dict):
                class_names = list(_class_ids.keys())
            elif isinstance(_class_ids, list):
                class_names = _class_ids
        elif hasattr(test_loader.dataset, 'class_names'):
            class_names = test_loader.dataset.class_names

        if class_names is None and hasattr(test_loader.dataset, 'data_dir'):
            category2index_path = os.path.join(test_loader.dataset.data_dir, 'category2index.json')
            if os.path.exists(category2index_path):
                try:
                    with open(category2index_path, 'r', encoding='utf-8') as f:
                        category2index = json.load(f)
                    sorted_categories = sorted(category2index.items(), key=lambda x: x[1])
                    class_names = [name for name, _ in sorted_categories]
                    logger.info(f"Class names loaded from {category2index_path}: {class_names}")
                except Exception as e:
                    logger.warning(f"Failed to read category2index.json: {e}")

        if class_names is None:
            num_classes = self.cfg.DATA.NUMBER_CLASSES
            if self.cfg.DATA.NAME == 'barley' and num_classes == 6:
                class_names = ['CY_KW', 'CY_PP', 'CY_PS', 'CY_SM', 'CY_SY', 'CY_ZC']
                logger.info(f"Default barley class names: {class_names}")
            else:
                class_names = [f"class_{i}" for i in range(num_classes)]
                logger.info(f"Default class names: {class_names}")

        comp_evaluator = ComprehensiveEvaluator(
            num_classes=self.cfg.DATA.NUMBER_CLASSES,
            class_names=class_names,
            device=str(self.device)
        )

        results = comp_evaluator.compute_all_metrics(
            y_pred_probs=all_probs,
            y_true=all_labels,
            model=self.model,
            input_size=(3, self.cfg.DATA.CROPSIZE, self.cfg.DATA.CROPSIZE),
            batch_size=1,
            num_warmup=10,
            num_runs=100,
        )

        comp_evaluator.print_summary()

        logger.info("\nPlotting confusion matrices...")

        confusion_matrix_path_count = os.path.join(self.cfg.OUTPUT_DIR, "confusion_matrix_count.png")
        logger.info("Plotting confusion matrix counts...")
        comp_evaluator.plot_confusion_matrix(
            save_path=confusion_matrix_path_count,
            figsize=(8, 7),
            dpi=300,
            cmap='Blues',
            normalize=False,
            show_stats=False,
        )

        confusion_matrix_path_percent = os.path.join(self.cfg.OUTPUT_DIR, "confusion_matrix_percent.png")
        logger.info("Plotting confusion matrix percentages...")
        comp_evaluator.plot_confusion_matrix(
            save_path=confusion_matrix_path_percent,
            figsize=(8, 7),
            dpi=300,
            cmap='Blues',
            normalize=True,
            show_stats=True,
        )

        model_name = self._get_model_name()

        results_dir = os.path.dirname(GLOBAL_RESULTS_CSV)
        if results_dir:
            os.makedirs(results_dir, exist_ok=True)

        comp_evaluator.save_to_csv(
            csv_path=GLOBAL_RESULTS_CSV,
            model_name=model_name,
            config_name=self.cfg.OUTPUT_DIR,
            additional_info={
                'dataset': self.cfg.DATA.NAME,
                'num_test_samples': len(all_labels),
                'num_classes': self.cfg.DATA.NUMBER_CLASSES,
                'input_size': self.cfg.DATA.CROPSIZE,
                'batch_size': self.cfg.DATA.BATCH_SIZE,
                'learning_rate': self.cfg.SOLVER.BASE_LR,
                'total_epochs': self.cfg.SOLVER.TOTAL_EPOCH,
            }
        )

        logger.info(f"Test results saved to: {GLOBAL_RESULTS_CSV}")
        if os.path.exists(confusion_matrix_path_count):
            logger.info(f"Confusion matrix counts saved to: {confusion_matrix_path_count}")
        if os.path.exists(confusion_matrix_path_percent):
            logger.info(f"Confusion matrix percentages saved to: {confusion_matrix_path_percent}")

        return results

    def _get_model_name(self) -> str:
        names = {"cadlgi_rep": "CADLGI-Rep", "cadlgi_swin": "CADLGI-Swin",
                 "pure_repvit": "RepViT-M1.0", "swin": "Swin-T"}
        name = names.get(self.cfg.MODEL.TYPE, self.cfg.MODEL.TYPE)
        if self.cfg.MODEL.TYPE.startswith("cadlgi_"):
            settings = self.cfg.MODEL.CADLGI
            name += f" (CADPA={settings.USE_CADPA}, ALGF={settings.USE_ALGF}, version={settings.ALGF_VERSION})"
        return name

    @torch.no_grad()
    def eval_classifier(self, data_loader, prefix, save=False):
        """evaluate classifier"""
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')

        log_interval = self.cfg.SOLVER.LOG_EVERY_N
        test_name = prefix + "_" + data_loader.dataset.name
        total = len(data_loader)

        # initialize features and target
        total_logits = []
        total_targets = []

        for idx, input_data in enumerate(data_loader):
            end = time.time()
            X, targets = self.get_input(input_data)
            # measure data loading time
            data_time.update(time.time() - end)

            if self.cfg.DBG:
                logger.info("during eval: {}".format(X.shape))
            loss, outputs = self.forward_one_batch(X, targets, False)
            if loss == -1:
                return 0.0
            losses.update(loss, X.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)

            if (idx + 1) % log_interval == 0:
                logger.info(
                    "\tTest {}/{}. loss: {:.3f}, {:.4f} s / batch. (data: {:.2e})".format(  # noqa
                        idx + 1,
                        total,
                        losses.val,
                        batch_time.val,
                        data_time.val
                    ) + "max mem: {:.5f} GB ".format(gpu_mem_usage())
                )

            # targets: ensure 1D class indices
            if targets.dim() > 1:
                safe_targets = torch.argmax(targets, dim=-1)
            else:
                safe_targets = targets
            total_targets.extend(safe_targets.cpu().tolist())
            total_logits.append(outputs)
        logger.info(
            f"Inference ({prefix}):"
            + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                data_time.avg, batch_time.avg)
            + "average loss: {:.4f}".format(losses.avg))
        if self.model.side is not None:
            logger.info(
                "--> side tuning alpha = {:.4f}".format(self.model.side_alpha))
        # total_testimages x num_classes
        joint_logits = torch.cat(total_logits, dim=0).cpu().numpy()
        self.evaluator.classify(
            joint_logits, total_targets,
            test_name, self.cfg.DATA.MULTILABEL,
        )
        return float(losses.avg)
