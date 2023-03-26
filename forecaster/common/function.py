import re

import numpy as np
import torch
from sklearn.metrics import (
    r2_score,
)


EPS = 1e-5


def printt(s=None):
    if s is None:
        print()
    else:
        print(str(s), end="\t")



# loss and metric functions


class K(object):
    """backend kernel"""

    @staticmethod
    def sum(x, axis=0, keepdims=True):
        if isinstance(x, np.ndarray):
            return x.sum(axis=axis, keepdims=keepdims)
        if isinstance(x, torch.Tensor):
            return x.sum(dim=axis, keepdim=keepdims)
        raise NotImplementedError("unsupported data type %s" % type(x))

    @staticmethod
    def clip(x, min_val, max_val):
        if isinstance(x, np.ndarray):
            return np.clip(x, min_val, max_val)
        if isinstance(x, torch.Tensor):
            return torch.clamp(x, min_val, max_val)
        raise NotImplementedError("unsupported data type %s" % type(x))

    @staticmethod
    def mean(x, axis=0, keepdims=True):
        # print(x.max())
        if isinstance(x, np.ndarray):
            return x.mean(axis=axis, keepdims=keepdims)
        if isinstance(x, torch.Tensor):
            return x.mean(dim=axis, keepdim=keepdims)
        raise NotImplementedError("unsupported data type %s" % type(x))

    @staticmethod
    def seq_mean(x, keepdims=True):
        if isinstance(x, torch.Tensor):
            return x.mean()
        if isinstance(x, np.ndarray):
            return x.mean()
        raise NotImplementedError("unsupported data type %s" % type(x))

    @staticmethod
    def std(x, axis=0, keepdims=True):
        if isinstance(x, np.ndarray):
            return x.std(axis=axis, keepdims=keepdims)
        if isinstance(x, torch.Tensor):
            return x.std(dim=axis, unbiased=False, keepdim=keepdims)
        raise NotImplementedError("unsupported data type %s" % type(x))

    @staticmethod
    def median(x, axis=0, keepdims=True):
        # NOTE: numpy will average when size is even,
        # but tensorflow and pytorch don't average
        if isinstance(x, np.ndarray):
            return np.median(x, axis=axis, keepdims=keepdims)
        if isinstance(x, torch.Tensor):
            return torch.median(x, dim=axis, keepdim=keepdims)[0]
        raise NotImplementedError("unsupported data type %s" % type(x))

    @staticmethod
    def shape(x):
        if isinstance(x, np.ndarray):
            return x.shape
        if isinstance(x, torch.Tensor):
            return list(x.shape)
        raise NotImplementedError("unsupported data type %s" % type(x))

    @staticmethod
    def cast(x, dtype="float"):
        if isinstance(x, np.ndarray):
            return x.astype(dtype)
        if isinstance(x, torch.Tensor):
            return x.type(getattr(torch, dtype))
        raise NotImplementedError("unsupported data type %s" % type(x))

    @staticmethod
    def maximum(x, y):
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            return np.minimum(x, y)
        if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
            return torch.max(x, y)
        elif isinstance(x, torch.Tensor):
            return torch.clamp(x, max=y)
        elif isinstance(y, torch.Tensor):
            return torch.clamp(y, max=x)
        raise NotImplementedError("unsupported data type %s" % type(x))


    @staticmethod
    def r2_score(y, p):
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        if isinstance(p, torch.Tensor):
            p = p.detach().cpu().numpy()
        if isinstance(y, np.ndarray) and isinstance(p, np.ndarray):
            y = y.reshape(-1)
            p = p.reshape(len(y), -1)
            if p.shape[-1] == 1:
                p = p.squeeze(-1)
            return r2_score(y, p)
        if isinstance(y, list) and isinstance(p, list):
            y, p = np.array(y), np.array(p)
            return K.r2_score(y, p)
        raise NotImplementedError("unsupported data type %s or %s" % (type(y), type(p)))


# Add Static Methods
def generic_ops(method):
    def wrapper(x, *args):
        if isinstance(x, np.ndarray):
            return getattr(np, method)(x, *args)
        if isinstance(x, torch.Tensor):
            return getattr(torch, method)(x, *args)
        raise NotImplementedError("unsupported data type %s" % type(x))

    return wrapper


for method in [
    "abs",
    "log",
    "sqrt",
    "exp",
    "log1p",
    "tanh",
    "cosh",
    "squeeze",
    "reshape",
    "zeros_like",
]:
    setattr(K, method, staticmethod(generic_ops(method)))

# Functions


def zscore(x, axis=0):
    mean = K.mean(x, axis=axis)
    std = K.std(x, axis=axis)
    return (x - mean) / (std + EPS)


def robust_zscore(x, axis=0):
    med = K.median(x, axis=axis)
    mad = K.median(K.abs(x - med), axis=axis)
    x = (x - med) / (mad * 1.4826 + EPS)
    return K.clip(x, -3, 3)


def batch_corr(x, y, axis=0, keepdims=True):
    x = zscore(x, axis=axis)
    y = zscore(y, axis=axis)
    return (x * y).mean()


def robust_batch_corr(x, y, axis=0, keepdims=True):
    x = robust_zscore(x, axis=axis)
    y = robust_zscore(y, axis=axis)
    return batch_corr(x, y)



def r2(y, preds):
    return K.r2_score(y, preds)


def sequence_mse(y_true, y_pred):
    loss = (y_true - y_pred) ** 2
    # return torch.mean(loss)

    return K.seq_mean(loss, keepdims=False)


def sequence_mae(y_true, y_pred):
    loss = torch.abs(y_true - y_pred)
    # return torch.mean(loss)
    return K.seq_mean(loss, keepdims=False)


def sequence_mase(y_true, y_pred):
    if isinstance(y_true, np.ndarray):
        loss = (y_true - y_pred) ** 2 + np.abs(y_true - y_pred)
    else:
        loss = (y_true - y_pred) ** 2 + (y_true - y_pred).abs()
    return K.seq_mean(loss, keepdims=False)


def single_mase(y_true, y_pred):
    if isinstance(y_true, np.ndarray):
        loss = (y_true - y_pred) ** 2 + np.abs(y_true - y_pred)
    else:
        loss = (y_true - y_pred) ** 2 + (y_true - y_pred).abs()
    return K.mean(loss, keepdims=False)


def single_mae(y_true, y_pred):
    if isinstance(y_true, np.ndarray):
        loss = np.abs(y_true - y_pred)
        return np.nanmean(loss)
    else:
        loss = (y_true - y_pred).abs()
        return loss.mean()


def rrse(y_true, y_pred):
    if isinstance(y_true, np.ndarray):
        y_bar = y_true.mean(axis=0)
        loss = np.sqrt(((y_pred - y_true) ** 2).sum()) / np.sqrt(((y_true - y_bar) ** 2).sum())
        return np.nanmean(loss)
    else:
        y_bar = y_true.mean(dim=0)
        loss = torch.sqrt(((y_pred - y_true) ** 2).sum()) / torch.sqrt(((y_true - y_bar) ** 2).sum())
        return loss.mean()


def mape(y_true, y_pred, log=False):
    if isinstance(y_true, np.ndarray):
        if log:
            y_true = np.exp(y_true)
            y_pred = np.exp(y_pred)
        loss = np.abs(y_true - y_pred) / y_true
    else:
        if log:
            y_true = y_true.exp()
            y_pred = y_pred.exp()
        loss = (y_true - y_pred).abs()
    return loss.mean()


def mape_log(y_true, y_pred, log=True):
    if isinstance(y_true, np.ndarray):
        if log:
            y_true = np.exp(y_true)
            y_pred = np.exp(y_pred)
        loss = np.abs(y_true - y_pred) / y_true
    else:
        if log:
            y_true = torch.exp(y_true)
            y_pred = torch.exp(y_pred)
        loss = (y_true - y_pred).abs()
    return K.mean(loss, keepdims=False)


def single_mse(y_true, y_pred):
    if isinstance(y_true, np.ndarray):
        loss = (y_true - y_pred) ** 2
        return np.nanmean(loss)
    else:
        mask = ~torch.isnan(y_true)
        y_pred = torch.masked_select(y_pred, mask)
        y_true = torch.masked_select(y_true, mask)
        loss = (y_true - y_pred) ** 2
        # loss = loss.reshape(-1)
        # mask = torch.logical_not(torch.isnan(loss))
        # loss = torch.masked_select(loss, mask)
        # print(len(loss))
        loss = loss.mean()

        return loss



def neg_wrapper(func):
    def wrapper(*args, **kwargs):
        return -1 * func(*args, **kwargs)

    return wrapper


def get_loss_fn(loss_fn):
    # reflection: legacy name
    if loss_fn == "mse":
        return single_mse
    if loss_fn == "single_mse":
        return single_mse
    if loss_fn == "mase":
        return sequence_mase
    if loss_fn == "mae":
        return single_mae
    if loss_fn.startswith("label"):
        return single_mse
    # if loss_fn == 'mape_log':
    #     return partial(mape, log=True)

    # return function by name
    try:
        return eval(loss_fn)  # dangerous eval
    except Exception:
        pass
    # return negative function by name
    try:
        return neg_wrapper(eval(re.sub("^neg_", "", loss_fn)))
    except Exception:
        raise NotImplementedError("loss function %s is not implemented" % loss_fn)


def get_metric_fn(eval_metric):
    # reflection: legacy name
    if eval_metric == "corr":
        return neg_wrapper(robust_batch_corr)  # more stable
    if eval_metric == "mse":
        return single_mse
    if eval_metric == "mae":
        return single_mae
    if eval_metric == "rse" or eval_metric == "rrse":
        return rrse
    # return function by name
    # if eval_metric == 'mape_log':
    #     return partial(mape, log=True)
    try:
        return eval(eval_metric)  # dangerous eval
    except Exception:
        pass
    # return negative function by name
    try:
        return neg_wrapper(eval(re.sub("^neg_", "", eval_metric)))
    except Exception:
        raise NotImplementedError("metric function %s is not implemented" % eval_metric)


def test():
    pass


if __name__ == "__main__":
    test()
