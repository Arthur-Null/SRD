from typing import Any, Optional, Union, Iterable
from numbers import Number
import time
import datetime
from collections import namedtuple
import copy

import numpy as np
import torch


# Environment Classes
Obs = namedtuple(
    "Obs",
    (
        "date",
        "t",
        "target",
        "position",
        "sell",
        "p",  # vwap price at t-1
        "v",  # market volume at t-1
    ),
)


def to_torch(
    x: Any,
    dtype: Optional[torch.dtype] = None,
    device: Union[str, int, torch.device] = "cpu",
) -> Optional[Union[torch.Tensor, Iterable]]:
    """Return an object without np.ndarray."""
    if isinstance(x, np.ndarray) and issubclass(x.dtype.type, (np.bool_, np.number)):
        x = torch.from_numpy(x).to(device)  # type: ignore
        if dtype is not None:
            x = x.type(dtype)
        return x
    elif isinstance(x, torch.Tensor):  # second often case
        if dtype is not None:
            x = x.type(dtype)
        return x.to(device)  # type: ignore
    elif isinstance(x, (np.number, np.bool_, Number)):
        return to_torch(np.asanyarray(x), dtype, device)
    elif isinstance(x, dict):
        return {k: to_torch(v, dtype, device) for k, v in x.items()}
    elif isinstance(x, (list, tuple)):
        return (to_torch(i, dtype, device) for i in x)
    else:  # fallback
        raise TypeError(f"object {x} cannot be converted to torch.")


# Evaluation Metrics


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.sum = 0.0
        self.avg = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = 1.0 * self.sum / self.count

    def performance(self, care="avg"):
        return getattr(self, care)

    def status(self):
        return str(self.performance())


class GlobalMeter(object):
    def __init__(self, f=lambda x, y: 0):
        self.reset()
        self.f = f

    def reset(self):
        self.ys = []  # np.array([], dtype=np.int) # ground truths
        self.preds = []  # np.array([], dtype=np.float) # predictions

    def update(self, ys, preds):
        if isinstance(ys, torch.Tensor):
            ys = ys.detach().squeeze(-1).cpu().numpy()
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().squeeze(-1).cpu().numpy()
        assert isinstance(ys, np.ndarray) and isinstance(preds, np.ndarray), "Please input as type of ndarray."
        self.ys.append(ys)
        self.preds.append(preds)

    def concat(self):
        if isinstance(self.ys, list) and isinstance(self.preds, list):
            self.ys = np.concatenate(self.ys, axis=0)
            self.preds = np.concatenate(self.preds, axis=0)
        else:
            return

    def get_ys(self):
        # deprecated
        return np.concatenate(self.ys, axis=0)

    def get_preds(self):
        # deprecated
        return np.concatenate(self.preds, axis=0)

    def performance(self):
        return self.f(self.ys, self.preds)

    def status(self):
        return str(self.performance())


class GlobalTracker(GlobalMeter):
    def __init__(self, metrics, metric_fn):
        self.reset()
        self.metrics = metrics
        self.metric_fn = metric_fn
        self.ss = {}

    def performance(self, metric="all"):
        stat = {}
        if isinstance(metric, str):
            assert (metric == "all") or (metric in self.metrics), "Not support %s metric." % metric
            if metric == "all":
                for m in self.metrics:
                    res = self.metric_fn[m](self.ys, self.preds)
                    if hasattr(res, "item"):
                        res = res.item()
                    stat[m] = res
                    self.ss[m] = stat[m]
            else:
                res = self.metric_fn[metric](self.ys, self.preds)
                if hasattr(res, "item"):
                    res = res.item()
                stat[metric] = res
                self.ss[metric] = stat[metric]
        else:
            raise NotImplementedError("TODO")
        return stat

    def snapshot(self, metric="all"):
        stat = {}
        if isinstance(metric, str):
            assert (metric == "all") or (metric in self.metrics), "Not support %s metric." % metric
            if metric == "all":
                for m in self.metrics:
                    try:
                        stat[m] = self.metric_fn[m]
                    except Exception:
                        raise KeyError("Run performance first")
            else:
                try:
                    stat[metric] = self.ss[metric]
                except Exception:
                    raise KeyError("Run performance first")
        else:
            raise NotImplementedError("TODO")
        return stat


def __deepcopy__(self, memo={}):
    cls = self.__class__
    copyobj = cls.__new__(cls)
    memo[id(self)] = copyobj
    for attr, value in self.__dict__.items():
        try:
            setattr(copyobj, attr, copy.deepcopy(value, memo))
        except Exception:
            pass
    return copyobj
