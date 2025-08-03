import logging
logging.basicConfig(level='ERROR')
import numpy as np
from tqdm import tqdm
import json
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
import matplotlib
import random
import os


def sweep(score, x):
    """
    Compute a ROC curve and then return the FPR, TPR, AUC, and ACC.
    """
    fpr, tpr, _ = roc_curve(x, -score)
    acc = np.max(1-(fpr+(1-tpr))/2)
    return fpr, tpr, auc(fpr, tpr), acc


def auc_acc_low(prediction, answers, sweep_fn=sweep, low=0.05 ):
    fpr, tpr, auc, acc = sweep_fn(np.array(prediction), np.array(answers, dtype=bool))

    low = tpr[np.where(fpr<low)[0][-1]]

    return auc, acc, low


def evaluate(preds, labels, part):
    """
    pred: predictions with every metrics
    """

    auc_results = dict()
    acc_results = dict()
    auc_low_results = dict()

    for _part, _part_pred in preds.items():
        if _part not in auc_results:
            auc_results[_part] = dict()
            acc_results[_part] = dict()
            auc_low_results[_part] = dict()
        for _metric, _metric_val in _part_pred.items():
            if isinstance(_metric_val, list):
                auc_val, acc_val, auc_low_val = auc_acc_low(prediction=_metric_val, answers=labels)
                auc_results[_part][_metric] = auc_val
                acc_results[_part][_metric] = acc_val
                auc_low_results[_part][_metric] = auc_low_val

            else:
                auc_results[_part][_metric] = dict()
                acc_results[_part][_metric] = dict()
                auc_low_results[_part][_metric] = dict()
                for _sub_metric, _sub_metric_val in _metric_val.items():
                    auc_val, acc_val, auc_low_val = auc_acc_low(prediction=_sub_metric_val, answers=labels)
                    auc_results[_part][_metric][_sub_metric] = auc_val
                    acc_results[_part][_metric][_sub_metric] = acc_val
                    auc_low_results[_part][_metric][_sub_metric] = auc_low_val

    return auc_results, acc_results, auc_low_results