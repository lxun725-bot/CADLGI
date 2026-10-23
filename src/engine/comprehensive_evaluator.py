#!/usr/bin/env python3
"""Classification metrics, model diagnostics, and optional plots."""

import os
import time
import json
import csv
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    top_k_accuracy_score
)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Plotting requires matplotlib and seaborn. Install: pip install matplotlib seaborn")


class ComprehensiveEvaluator:
    """Compute classification metrics and export reports."""

    def __init__(
        self,
        num_classes: int = 6,
        class_names: Optional[List[str]] = None,
        device: str = 'cuda'
    ):
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.device = device
        self.results = OrderedDict()

    def compute_all_metrics(
        self,
        y_pred_probs: np.ndarray,
        y_true: np.ndarray,
        model: Optional[nn.Module] = None,
        input_size: Tuple[int, int, int] = (3, 224, 224),
        batch_size: int = 1,
        num_warmup: int = 10,
        num_runs: int = 100,
    ) -> Dict[str, Any]:
        """Compute metrics from probabilities and reference labels."""
        results = OrderedDict()

        acc_metrics = self._compute_accuracy_metrics(y_pred_probs, y_true)
        results.update(acc_metrics)

        # 2. Precision, Recall, F1
        prf_metrics = self._compute_prf_metrics(y_pred_probs, y_true)
        results.update(prf_metrics)

        # 3. AUC-ROC
        auc_metrics = self._compute_auc_metrics(y_pred_probs, y_true)
        results.update(auc_metrics)

        # 4. Confusion Matrix
        cm = self._compute_confusion_matrix(y_pred_probs, y_true)
        results['confusion_matrix'] = cm.tolist()

        # 5. Per-class metrics
        per_class = self._compute_per_class_metrics(y_pred_probs, y_true)
        results['per_class_accuracy'] = per_class

        if model is not None:
            model_metrics = self._compute_model_metrics(
                model, input_size, batch_size, num_warmup, num_runs
            )
            results.update(model_metrics)

        self.results = results
        return results

    def _compute_accuracy_metrics(
        self,
        y_pred_probs: np.ndarray,
        y_true: np.ndarray
    ) -> Dict[str, float]:
        """Compute top-1 and top-k accuracy."""
        y_pred = np.argmax(y_pred_probs, axis=1)

        # Top-1 Accuracy
        top1_acc = accuracy_score(y_true, y_pred)

        k = min(5, self.num_classes)
        try:
            topk_acc = top_k_accuracy_score(y_true, y_pred_probs, k=k, labels=range(self.num_classes))
        except:
            topk_acc = self._top_k_accuracy(y_pred_probs, y_true, k)

        return {
            'top1_accuracy': round(top1_acc * 100, 4),
            f'top{k}_accuracy': round(topk_acc * 100, 4),
            'error_rate': round((1 - top1_acc) * 100, 4),
        }

    def _top_k_accuracy(self, y_probs: np.ndarray, y_true: np.ndarray, k: int) -> float:
        """Compute top-k accuracy from class probabilities."""
        top_k_preds = np.argsort(y_probs, axis=1)[:, -k:]
        correct = 0
        for i, true_label in enumerate(y_true):
            if true_label in top_k_preds[i]:
                correct += 1
        return correct / len(y_true)

    def _compute_prf_metrics(
        self,
        y_pred_probs: np.ndarray,
        y_true: np.ndarray
    ) -> Dict[str, float]:
        """Compute precision, recall, and F1 scores."""
        y_pred = np.argmax(y_pred_probs, axis=1)

        return {
            'precision_macro': round(precision_score(y_true, y_pred, average='macro', zero_division=0) * 100, 4),
            'precision_micro': round(precision_score(y_true, y_pred, average='micro', zero_division=0) * 100, 4),
            'precision_weighted': round(precision_score(y_true, y_pred, average='weighted', zero_division=0) * 100, 4),
            'recall_macro': round(recall_score(y_true, y_pred, average='macro', zero_division=0) * 100, 4),
            'recall_micro': round(recall_score(y_true, y_pred, average='micro', zero_division=0) * 100, 4),
            'recall_weighted': round(recall_score(y_true, y_pred, average='weighted', zero_division=0) * 100, 4),
            'f1_macro': round(f1_score(y_true, y_pred, average='macro', zero_division=0) * 100, 4),
            'f1_micro': round(f1_score(y_true, y_pred, average='micro', zero_division=0) * 100, 4),
            'f1_weighted': round(f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100, 4),
        }

    def _compute_auc_metrics(
        self,
        y_pred_probs: np.ndarray,
        y_true: np.ndarray
    ) -> Dict[str, float]:
        """Compute multiclass ROC AUC."""
        try:
            # One-vs-Rest AUC
            auc_ovr = roc_auc_score(y_true, y_pred_probs, multi_class='ovr', average='macro')
            auc_ovo = roc_auc_score(y_true, y_pred_probs, multi_class='ovo', average='macro')
        except Exception as e:
            print(f"AUC calculation failed: {e}")
            auc_ovr = -1
            auc_ovo = -1

        return {
            'auc_roc_ovr': round(auc_ovr * 100, 4) if auc_ovr > 0 else -1,
            'auc_roc_ovo': round(auc_ovo * 100, 4) if auc_ovo > 0 else -1,
        }

    def _compute_confusion_matrix(
        self,
        y_pred_probs: np.ndarray,
        y_true: np.ndarray
    ) -> np.ndarray:
        """Store the confusion matrix."""
        y_pred = np.argmax(y_pred_probs, axis=1)
        return confusion_matrix(y_true, y_pred)

    def _compute_per_class_metrics(
        self,
        y_pred_probs: np.ndarray,
        y_true: np.ndarray
    ) -> Dict[str, float]:
        """Compute per-class classification metrics."""
        y_pred = np.argmax(y_pred_probs, axis=1)

        per_class = {}
        for i in range(self.num_classes):
            mask = y_true == i
            if mask.sum() > 0:
                class_acc = (y_pred[mask] == i).mean() * 100
                per_class[self.class_names[i]] = round(class_acc, 4)
            else:
                per_class[self.class_names[i]] = 0.0

        return per_class

    def _compute_model_metrics(
        self,
        model: nn.Module,
        input_size: Tuple[int, int, int],
        batch_size: int = 1,
        num_warmup: int = 10,
        num_runs: int = 100,
    ) -> Dict[str, Any]:
        """Record model size, supported operation counts, and timing."""
        model.eval()
        device = next(model.parameters()).device

        metrics = {}

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        metrics['total_params'] = total_params
        metrics['trainable_params'] = trainable_params
        metrics['total_params_M'] = round(total_params / 1e6, 4)
        metrics['trainable_params_M'] = round(trainable_params / 1e6, 4)

        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        model_size_mb = (param_size + buffer_size) / (1024 ** 2)
        metrics['model_size_MB'] = round(model_size_mb, 4)

        # Use one documented counter for every model. Unsupported operators are
        # retained in the report; do not label THOP MACs as FLOPs or divide by 2.
        try:
            from ..utils.model_profile import count_model_operations
            import json
            with torch.no_grad():
                counted = count_model_operations(model, torch.randn(1, *input_size).to(device))
            for key in ('fvcore_by_operator', 'fvcore_unsupported_ops'):
                counted[key] = json.dumps(counted[key], sort_keys=True)
            metrics.update(counted)
        except Exception as exc:
            metrics['fvcore_count_error'] = str(exc)

        inference_metrics = self._measure_inference_speed(
            model, input_size, batch_size, num_warmup, num_runs, device
        )
        metrics.update(inference_metrics)

        return metrics

    def _measure_inference_speed(
        self,
        model: nn.Module,
        input_size: Tuple[int, int, int],
        batch_size: int,
        num_warmup: int,
        num_runs: int,
        device: torch.device,
    ) -> Dict[str, float]:
        """Measure inference timing with randomly generated input tensors."""
        model.eval()
        dummy_input = torch.randn(batch_size, *input_size).to(device)

        # Warmup
        with torch.no_grad():
            try:
                for _ in range(num_warmup):
                    output = model(dummy_input)
                    if isinstance(output, tuple):
                        output = output[0]
            except Exception as e:
                print(f"Inference warmup failed: {e}")
                return {
                    'inference_time_ms': -1,
                    'batch_inference_time_ms': -1,
                    'throughput_images_per_sec': -1,
                }

        if device.type == 'cuda':
            torch.cuda.synchronize()

        start_time = time.time()
        with torch.no_grad():
            try:
                for _ in range(num_runs):
                    output = model(dummy_input)
                    if isinstance(output, tuple):
                        output = output[0]
            except Exception as e:
                print(f"Inference timing failed: {e}")
                return {
                    'inference_time_ms': -1,
                    'batch_inference_time_ms': -1,
                    'throughput_images_per_sec': -1,
                }

        if device.type == 'cuda':
            torch.cuda.synchronize()

        end_time = time.time()

        total_time = end_time - start_time
        avg_time_per_batch = total_time / num_runs * 1000  # ms
        avg_time_per_image = avg_time_per_batch / batch_size  # ms
        throughput = (num_runs * batch_size) / total_time  # images/sec

        return {
            'inference_time_ms': round(avg_time_per_image, 4),
            'batch_inference_time_ms': round(avg_time_per_batch, 4),
            'throughput_images_per_sec': round(throughput, 2),
        }

    def save_to_csv(
        self,
        csv_path: str,
        model_name: str,
        config_name: str = "",
        additional_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append one result row to a CSV file."""
        row = OrderedDict()
        row['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row['model_name'] = model_name
        row['config_name'] = config_name

        if additional_info:
            row.update(additional_info)

        for key, value in self.results.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    row[f"{key}_{sub_key}"] = sub_value
            elif isinstance(value, list):
                row[key] = str(value)
            else:
                row[key] = value

        file_exists = os.path.exists(csv_path)

        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())

            if not file_exists:
                writer.writeheader()

            writer.writerow(row)

        print(f"Results saved to: {csv_path}")

    def print_summary(self) -> str:
        """Print the recorded metrics."""
        summary = []
        summary.append("\n" + "=" * 60)
        summary.append("Classification results")
        summary.append("=" * 60)

        summary.append("\nAccuracy metrics")
        for key in ['top1_accuracy', 'top5_accuracy', 'error_rate']:
            if key in self.results:
                summary.append(f"  {key}: {self.results[key]:.2f}%")

        summary.append("\n【Precision/Recall/F1】")
        for key in ['precision_macro', 'recall_macro', 'f1_macro', 'f1_weighted']:
            if key in self.results:
                summary.append(f"  {key}: {self.results[key]:.2f}%")

        # AUC
        summary.append("\n【AUC-ROC】")
        for key in ['auc_roc_ovr', 'auc_roc_ovo']:
            if key in self.results:
                val = self.results[key]
                summary.append(f"  {key}: {val:.2f}%" if val > 0 else f"  {key}: N/A")

        summary.append("\nModel metrics")
        if 'total_params_M' in self.results:
            summary.append(f"  Parameters: {self.results['total_params_M']:.2f}M")
        if 'model_size_MB' in self.results:
            summary.append(f"  Model size: {self.results['model_size_MB']:.2f}MB")
        if 'FLOPs_G' in self.results and self.results['FLOPs_G'] > 0:
            summary.append(f"  FLOPs: {self.results['FLOPs_G']:.2f}G")
        if 'inference_time_ms' in self.results:
            summary.append(f"  Inference time: {self.results['inference_time_ms']:.2f}ms/image")
        if 'throughput_images_per_sec' in self.results:
            summary.append(f"  Throughput: {self.results['throughput_images_per_sec']:.1f} images/sec")

        # Per-class accuracy
        if 'per_class_accuracy' in self.results:
            summary.append("\nPer-class accuracy")
            for class_name, acc in self.results['per_class_accuracy'].items():
                summary.append(f"  {class_name}: {acc:.2f}%")

        summary.append("\n" + "=" * 60)

        result_str = "\n".join(summary)
        print(result_str)
        return result_str

    def plot_confusion_matrix(
        self,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 7),
        dpi: int = 300,
        cmap: str = 'Blues',
        normalize: bool = False,
        show_stats: bool = True,
    ) -> Optional[str]:
        """Plot and optionally save the confusion matrix."""
        if not HAS_PLOTTING:
            print("Plotting requires matplotlib and seaborn.")
            return None

        if 'confusion_matrix' not in self.results:
            print("Compute the confusion matrix before plotting.")
            return None

        cm = np.array(self.results['confusion_matrix'])

        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['axes.unicode_minus'] = False

        try:
            if isinstance(cmap, str):
                try:
                    cmap_obj = plt.cm.get_cmap(cmap)
                except AttributeError:
                    cmap_obj = getattr(plt.cm, cmap)
            else:
                cmap_obj = cmap
        except Exception:
            cmap_obj = plt.cm.Blues

        bg_color = '#ffffff'
        text_color = '#212121'

        if normalize:
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_to_plot = cm_normalized * 100
            fmt = '.1f'
            label = 'Accuracy (%)'
            vmax = 100.0
        else:
            cm_to_plot = cm
            fmt = 'd'
            label = 'Count'
            vmax = cm.max()

        fig, ax = plt.subplots(figsize=figsize, facecolor=bg_color, dpi=dpi)
        ax.set_facecolor(bg_color)

        im = ax.imshow(cm_to_plot, interpolation='nearest', cmap=cmap_obj,
                       aspect='auto', vmin=0, vmax=vmax)

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(label, fontsize=12, fontweight='bold', color=text_color, labelpad=10)
        cbar.ax.tick_params(labelsize=10, colors=text_color)

        if self.class_names is None or len(self.class_names) != len(cm):
            if self.class_names is None:
                self.class_names = [f"Class_{i}" for i in range(len(cm))]
            elif len(self.class_names) != len(cm):
                if len(self.class_names) < len(cm):
                    self.class_names.extend([f"Class_{i}" for i in range(len(self.class_names), len(cm))])
                else:
                    self.class_names = self.class_names[:len(cm)]

        ax.set_xticks(np.arange(len(self.class_names)))
        ax.set_yticks(np.arange(len(self.class_names)))
        ax.set_xticklabels(self.class_names, fontsize=12, fontweight='medium', color=text_color)
        ax.set_yticklabels(self.class_names, fontsize=12, fontweight='medium', color=text_color)

        ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold', color=text_color, labelpad=12)
        ax.set_ylabel('True Label', fontsize=14, fontweight='bold', color=text_color, labelpad=12)

        if normalize:
            title = 'Confusion Matrix (Normalized)'
        else:
            title = 'Confusion Matrix'
        ax.set_title(title, fontsize=16, fontweight='bold', color=text_color, pad=15)

        thresh = vmax / 2.0
        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                value = cm_to_plot[i, j]
                text_color_cell = 'white' if value > thresh else text_color
                ax.text(j, i, f'{value:{fmt}}',
                        ha='center', va='center',
                        color=text_color_cell, fontsize=11, fontweight='bold')

        ax.set_xticks(np.arange(len(self.class_names)+1)-0.5, minor=True)
        ax.set_yticks(np.arange(len(self.class_names)+1)-0.5, minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=1.5)
        ax.tick_params(which='minor', size=0)

        for spine in ax.spines.values():
            spine.set_color('#bdbdbd')
            spine.set_linewidth(1.2)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=dpi, facecolor=bg_color,
                       edgecolor='none', bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")

            if show_stats:
                self._print_confusion_matrix_stats(cm)

            plt.close()
            return save_path
        else:
            if show_stats:
                self._print_confusion_matrix_stats(cm)
            plt.close()
            return None

    def _print_confusion_matrix_stats(self, cm: np.ndarray) -> None:
        """Print per-class confusion statistics."""
        total_samples = cm.sum()
        correct = np.trace(cm)
        accuracy = correct / total_samples * 100

        print("\n" + "="*60)
        print("Confusion matrix statistics")
        print("="*60)
        print(f"\nSamples: {total_samples}")
        print(f"Correct predictions: {correct}")
        print(f"Overall accuracy: {accuracy:.2f}%")
        print(f"\nPer-class accuracy:")
        for i, cls_name in enumerate(self.class_names):
            class_correct = cm[i, i]
            class_total = cm[i].sum()
            if class_total > 0:
                class_acc = class_correct / class_total * 100
                print(f"  {cls_name}: {class_correct}/{class_total} = {class_acc:.2f}%")
            else:
                print(f"  {cls_name}: 0/0 = N/A")


def evaluate_model_on_testset(
    model: nn.Module,
    test_loader,
    num_classes: int = 6,
    class_names: Optional[List[str]] = None,
    device: str = 'cuda',
    csv_path: Optional[str] = None,
    model_name: str = "Unknown",
    config_name: str = "",
) -> Dict[str, Any]:
    """Evaluate a model on the supplied test loader."""
    model.eval()
    model.to(device)

    all_probs = []
    all_labels = []

    print(f"\nEvaluating {model_name}...")

    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            labels = batch['label']

            # Forward
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.extend(labels.numpy().tolist())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.array(all_labels)

    evaluator = ComprehensiveEvaluator(
        num_classes=num_classes,
        class_names=class_names,
        device=device
    )

    results = evaluator.compute_all_metrics(
        y_pred_probs=all_probs,
        y_true=all_labels,
        model=model,
        input_size=(3, 224, 224),
        batch_size=1,
        num_warmup=10,
        num_runs=100,
    )

    evaluator.print_summary()

    if csv_path:
        evaluator.save_to_csv(
            csv_path=csv_path,
            model_name=model_name,
            config_name=config_name,
            additional_info={
                'num_test_samples': len(all_labels),
                'num_classes': num_classes,
            }
        )

    return results
