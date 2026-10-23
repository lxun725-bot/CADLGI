#!/usr/bin/env python3
import numpy as np

from collections import defaultdict
from typing import List, Union

from .eval import singlelabel
from ..utils import logging
logger = logging.get_logger("cadlgi")


class Evaluator():
    """
    An evaluator with below logics:

    1. find which eval module to use.
    2. store the eval results, pretty print it in log file as well.
    """

    def __init__(
        self,
    ) -> None:
        self.results = defaultdict(dict)
        self.iteration = -1
        self.threshold_end = 0.5
        self.compute_f1 = False

    def update_iteration(self, iteration: int) -> None:
        """update iteration info"""
        self.iteration = iteration

    def update_result(self, metric: str, value: Union[float, dict]) -> None:
        if self.iteration > -1:
            key_name = "epoch_" + str(self.iteration)
        else:
            key_name = "final"
        if isinstance(value, float):
            self.results[key_name].update({metric: value})
        else:
            if metric in self.results[key_name]:
                self.results[key_name][metric].update(value)
            else:
                self.results[key_name].update({metric: value})

    def classify(self, probs, targets, test_data, multilabel=False):
        """
        Evaluate classification result.
        Args:
            probs: np.ndarray for num_data x num_class, predicted probabilities
            targets: np.ndarray for multilabel, list of integers for single label
            test_labels:  map test image ids to a list of class labels
        """
        if not targets:
            raise ValueError(
                "When evaluating classification, need at least give targets")

        if multilabel:
            raise ValueError("Only single-label barley classification is retained")
        else:
            self._eval_singlelabel(probs, targets, test_data)

    def _eval_singlelabel(
        self,
        scores: np.ndarray,
        targets: List[int],
        eval_type: str
    ) -> None:
        """
        if number of labels > 2:
            top1 and topk (5 by default) accuracy
        if number of labels == 2:
            top1 and rocauc
        """
        # normalize scores to (N, C)
        sc = np.asarray(scores)
        if sc.ndim == 3:
            # average extra dim
            sc = sc.mean(axis=-1)
        elif sc.ndim > 3:
            sc = sc.reshape(sc.shape[0], sc.shape[1], -1).mean(axis=-1)

        # ensure targets are 1D class indices
        clean_targets = []
        for t in targets:
            try:
                arr = np.asarray(t)
                if arr.ndim == 0:
                    clean_targets.append(int(arr))
                else:
                    clean_targets.append(int(arr.reshape(-1).argmax()))
            except Exception:
                clean_targets.append(int(t))

        acc_dict = singlelabel.compute_acc_auc(sc, clean_targets, self.compute_f1)

        log_results = {
            k: np.around(v * 100, decimals=2) for k, v in acc_dict.items()
        }
        save_results = acc_dict

        self.log_and_update(log_results, save_results, eval_type)

    def log_and_update(self, log_results, save_results, eval_type):
        log_str = ""
        for k, result in log_results.items():
            if not isinstance(result, np.ndarray):
                log_str += f"{k}: {result:.2f}\t"
            else:
                log_str += f"{k}: {list(result)}\t"
        logger.info(f"Classification results with {eval_type}: {log_str}")
        # save everything
        self.update_result("classification", {eval_type: save_results})
