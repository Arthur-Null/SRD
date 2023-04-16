# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import copy
import datetime
import json
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from utilsd import use_cuda
from utilsd.earlystop import EarlyStopStatus

from ..common.function import printt
from ..common.utils import AverageMeter, GlobalTracker, to_torch
from .base import MODELS
from .timeseries import TS


@MODELS.register_module("minmaxsep")
class MinMaxSep(TS):
    def __init__(
        self,
        task: str,
        optimizer: str,
        lr: float,
        weight_decay: float,
        loss_fn: str,
        metrics: List[str],
        observe: str,
        lower_is_better: bool,
        max_epoches: int,
        batch_size: int,
        network: Optional[nn.Module] = None,
        output_dir: Optional[Path] = None,
        checkpoint_dir: Optional[Path] = None,
        early_stop: Optional[int] = None,
        out_ranges: Optional[List[Union[Tuple[int, int], Tuple[int, int, int]]]] = None,
        model_path: Optional[str] = None,
        alpha1: float = 1.0,
        alpha2: float = 1.0,
        out_size: int = 1,
        aggregate: bool = True,
    ):
        """
        The model for general time-series prediction.

        Args:
            task: the prediction task, classification or regression.
            optimizer: which optimizer to use.
            lr: learning rate.
            weight_decay: L2 normlize weight
            loss_fn: loss function.
            metrics: metrics to evaluate model.
            observe: metric for model selection (earlystop).
            lower_is_better: whether a lower observed metric means better result.
            max_epoches: maximum epoch to learn.
            batch_size: batch size.
            early_stop: earlystop rounds.
            out_ranges: a list of final ranges to take as final output. Should have form [(start, end), (start, end, step), ...]
            model_path: the path to existing model parameters for continued training or finetuning
            out_size: the output size for multi-class classification or multi-variant regression task.
            aggregate: whether to aggregate across whole sequence.
        """
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        super().__init__(
            task,
            optimizer,
            lr,
            weight_decay,
            loss_fn,
            metrics,
            observe,
            lower_is_better,
            max_epoches,
            batch_size,
            network,
            output_dir,
            checkpoint_dir,
            early_stop,
            out_ranges,
            model_path,
            out_size,
            aggregate,
        )

    def _build_network(
        self,
        network,
        task: str,
        out_ranges: Optional[List[Union[Tuple[int, int, int], Tuple[int, int]]]] = None,
        out_size: int = 1,
        aggregate: bool = True,
    ) -> None:
        """Initilize the network parameters

        Args:
            task: the prediction task, classification or regression.
            out_ranges: a list of final ranges to take as final output. Should have form [(start, end), (start, end, step), ...]
            out_size: the output size for multi-class classification or multi-variant regression task.
            aggregate: whether to aggregate across whole sequence.
        """

        self.network = network
        self.aggregate = aggregate

        # Output
        if task == "classification":
            self.act_out = nn.Sigmoid()
            out_size = 1
        elif task == "multiclassification":
            self.act_out = nn.LogSoftmax(-1)
        elif task == "regression":
            self.act_out = nn.Identity()
        else:
            raise ValueError(("Task must be 'classification', 'multiclassification', 'regression'"))

        if out_ranges is not None:
            self.out_ranges = []
            for ran in out_ranges:
                if len(ran) == 2:
                    self.out_ranges.append(np.arange(ran[0], ran[1]))
                elif len(ran) == 3:
                    self.out_ranges.append(np.arange(ran[0], ran[1], ran[2]))
                else:
                    raise ValueError(f"Unknown range {ran}")
            self.out_ranges = np.concatenate(self.out_ranges)
        else:
            self.out_ranges = None

        self.fc_out_static = nn.Linear(network.output_size, out_size)
        self.fc_out_dynamic = nn.Linear(network.output_size, out_size)
        self.task = task

    def forward(self, inputs):
        seq_out_static, emb_out_static, seq_out_dyna, emb_out_dyna = self.network(inputs)  # [B, T, H]
        if self.aggregate:
            out_dyna = emb_out_dyna
            out_static = emb_out_static
        else:
            out_dyna = seq_out_dyna
            out_static = seq_out_static
        preds_dyna = self.act_out(self.fc_out_dynamic(out_dyna).squeeze(-1))
        preds_static = self.act_out(self.fc_out_static(out_static).squeeze(-1))
        if self.task == "multiclassification":
            preds = torch.exp(preds_dyna) + torch.exp(preds_static)
            preds = torch.log(preds / 2)
        else:
            preds = (preds_dyna + preds_static) / 2

        return preds_dyna, preds_static, preds

    def fit(
        self,
        trainset: Dataset,
        validset: Optional[Dataset] = None,
        testset: Optional[Dataset] = None,
    ) -> nn.Module:
        """Fit the model to data, if evaluation dataset is offered,
           model selection (early stopping) would be conducted on it.

        Args:
            trainset (Dataset): The training dataset.
            validset (Dataset, optional): The evaluation dataset. Defaults to None.
            testset (Dataset, optional): The test dataset. Defaults to None.

        Returns:
            nn.Module: return the model itself.
        """
        trainset.load()
        if validset is not None:
            validset.load()

        loader = DataLoader(
            trainset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=8,
        )

        self._init_scheduler(len(loader))
        self.best_params = copy.deepcopy(self.state_dict())
        self.best_network_params = copy.deepcopy(self.network.state_dict())
        iterations = 0
        start_epoch, best_res = self._resume()
        best_epoch = best_res.pop("best_epoch", 0)
        best_score = self.early_stop.best
        for epoch in range(start_epoch, self.max_epoches):
            self.train()
            train_loss = AverageMeter()
            train_loss_static = AverageMeter()
            train_loss_dynamic = AverageMeter()
            train_loss_final = AverageMeter()
            train_global_tracker = GlobalTracker(self.metrics, self.metric_fn)
            start_time = time.time()
            minmax_loss = AverageMeter()
            for _, (data, label) in enumerate(loader):
                if use_cuda():
                    data, label = to_torch(data, device="cuda"), to_torch(label, device="cuda")
                pred1, pred2, pred = self(data)
                if self.out_ranges is not None:
                    pred = pred[:, self.out_ranges]
                    pred1 = pred1[:, self.out_ranges]
                    pred2 = pred2[:, self.out_ranges]
                    label = label[:, self.out_ranges]
                loss_dynamic = self.loss_fn(label.squeeze(-1), pred1.squeeze(-1))
                loss_static = self.loss_fn(label.squeeze(-1), pred2.squeeze(-1))
                loss_final = self.loss_fn(label.squeeze(-1), pred.squeeze(-1))
                loss = (
                    loss_dynamic
                    + loss_static
                    + loss_final
                    + self.alpha1 * self.network.minmax_loss[0]
                    + self.alpha2 * self.network.minmax_loss[1]
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
                self.optimizer.step()
                loss = loss.item()
                train_loss.update(loss, np.prod(label.shape))
                minmax_loss.update(self.network.minmax_loss_value.item(), np.prod(label.shape))
                train_loss_static.update(loss_static.item(), np.prod(label.shape))
                train_loss_dynamic.update(loss_dynamic.item(), np.prod(label.shape))
                train_loss_final.update(loss_final.item(), np.prod(label.shape))

                if hasattr(trainset, "scale"):
                    num_v = len(trainset.scale)
                    label = (label.reshape(-1, num_v).detach().cpu().numpy() * trainset.scale).reshape(label.shape)
                    pred = (pred.reshape(-1, num_v).detach().cpu().numpy() * trainset.scale).reshape(pred.shape)
                train_global_tracker.update(label, pred)
                if self.scheduler is not None:
                    self.scheduler.step()
                iterations += 1
                self._post_batch(iterations, epoch, train_loss, train_global_tracker, validset, testset)

            train_time = time.time() - start_time
            loss = train_loss.performance()  # loss
            start_time = time.time()
            train_global_tracker.concat()
            metric_res = train_global_tracker.performance()
            metric_time = time.time() - start_time
            metric_res["loss"] = loss
            metric_res["minmax_loss"] = np.mean(minmax_loss.performance()).item()
            metric_res["dynamic_loss"] = train_loss_dynamic.performance().item()
            metric_res["static_loss"] = train_loss_static.performance().item()
            metric_res["final_loss"] = train_loss_final.performance().item()

            # print log
            printt(f"{epoch}\t'train'\tTime:{train_time:.2f}\tMetricT: {metric_time:.2f}")
            for metric, value in metric_res.items():
                printt(f"{metric}: {value:.4f}")
            print(f"{datetime.datetime.today()}")
            for k, v in metric_res.items():
                self.writer.add_scalar(f"{k}/train", v, epoch)
            self.writer.flush()

            if validset is not None:
                with torch.no_grad():
                    eval_res = self.evaluate(validset, epoch)
                value = eval_res[self.observe]
                es = self.early_stop.step(value)
                if es == EarlyStopStatus.BEST:
                    best_score = value
                    best_epoch = epoch
                    self.best_params = copy.deepcopy(self.state_dict())
                    self.best_network_params = copy.deepcopy(self.network.state_dict())
                    best_res = {"train": metric_res, "valid": eval_res}
                    torch.save(self.best_params, f"{self.checkpoint_dir}/model_best.pkl")
                    torch.save(self.best_network_params, f"{self.checkpoint_dir}/network_best.pkl")
                elif es == EarlyStopStatus.STOP and self._early_stop():
                    break
            else:
                es = self.early_stop.step(metric_res[self.observe])
                if es == EarlyStopStatus.BEST:
                    best_score = metric_res[self.observe]
                    best_epoch = epoch
                    self.best_params = copy.deepcopy(self.state_dict())
                    self.best_network_params = copy.deepcopy(self.network.state_dict())
                    best_res = {"train": metric_res}
                    torch.save(self.best_params, f"{self.checkpoint_dir}/model_best.pkl")
                    torch.save(self.best_network_params, f"{self.checkpoint_dir}/network_best.pkl")
                elif es == EarlyStopStatus.STOP and self._early_stop():
                    break
            self._checkpoint(epoch, {**best_res, "best_epoch": best_epoch})

        # release the space of train and valid dataset
        trainset.freeup()
        if validset is not None:
            validset.freeup()

        # finish training, test, save model and write logs
        self._load_weight(self.best_params)
        if testset is not None:
            testset.load()
            print("Begin evaluate on testset ...")
            with torch.no_grad():
                test_res = self.evaluate(testset)
            for k, v in test_res.items():
                self.writer.add_scalar(f"{k}/test", v, epoch)
            value = test_res[self.observe]
            best_score = value
            best_res["test"] = test_res
            testset.freeup()
        torch.save(self.best_params, f"{self.checkpoint_dir}/model_best.pkl")
        torch.save(self.best_network_params, f"{self.checkpoint_dir}/network_best.pkl")
        with open(f"{self.checkpoint_dir}/res.json", "w") as f:
            json.dump(best_res, f, indent=4, sort_keys=True)
        keys = list(self.hyper_paras.keys())
        for k in keys:
            if type(self.hyper_paras[k]) not in [int, float, str, bool, torch.Tensor]:
                self.hyper_paras.pop(k)
        self.writer.add_hparams(self.hyper_paras, {"result": best_score, "best_epoch": best_epoch})

        return self

    def evaluate(self, validset: Dataset, epoch: Optional[int] = None) -> dict:
        """Evaluate the model on the given dataset.

        Args:
            validset (Dataset): The dataset to be evaluated on.
            epoch (int, optional): If given, would write log to tensorboard and stdout. Defaults to None.

        Returns:
            dict: The results of evaluation.
        """
        loader = DataLoader(
            validset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=8,
        )
        self.eval()
        eval_loss = AverageMeter()
        minmax_loss = AverageMeter()
        eval_loss_static = AverageMeter()
        eval_loss_dynamic = AverageMeter()
        eval_loss_final = AverageMeter()
        eval_global_tracker = GlobalTracker(self.metrics, self.metric_fn)
        start_time = time.time()
        validset.load()
        with torch.no_grad():
            for _, (data, label) in enumerate(loader):
                if use_cuda():
                    data, label = to_torch(data, device="cuda"), to_torch(label, device="cuda")
                pred1, pred2, pred = self(data)
                if self.out_ranges is not None:
                    pred = pred[:, self.out_ranges]
                    pred1 = pred1[:, self.out_ranges]
                    pred2 = pred2[:, self.out_ranges]
                    label = label[:, self.out_ranges]
                loss_dynamic = self.loss_fn(label.squeeze(-1), pred1.squeeze(-1))
                loss_static = self.loss_fn(label.squeeze(-1), pred2.squeeze(-1))
                loss_final = self.loss_fn(label.squeeze(-1), pred.squeeze(-1))
                loss = (
                    loss_dynamic
                    + loss_static
                    + loss_final
                    + self.alpha1 * self.network.minmax_loss[0]
                    + self.alpha2 * self.network.minmax_loss[1]
                )
                loss = loss.item()
                eval_loss.update(loss, np.prod(label.shape))
                eval_loss_static.update(loss_static.item(), np.prod(label.shape))
                eval_loss_dynamic.update(loss_dynamic.item(), np.prod(label.shape))
                eval_loss_final.update(loss_final.item(), np.prod(label.shape))
                if hasattr(validset, "scale"):
                    num_v = len(validset.scale)
                    label = (label.reshape(-1, num_v).detach().cpu().numpy() * validset.scale).reshape(label.shape)
                    pred = (pred.reshape(-1, num_v).detach().cpu().numpy() * validset.scale).reshape(pred.shape)
                eval_global_tracker.update(label, pred)
                minmax_loss.update(self.network.minmax_loss_value.item(), np.prod(label.shape))

        eval_time = time.time() - start_time
        loss = eval_loss.performance()  # loss
        start_time = time.time()
        eval_global_tracker.concat()
        metric_res = eval_global_tracker.performance()
        metric_time = time.time() - start_time
        metric_res["loss"] = loss
        metric_res["minmax_loss"] = minmax_loss.performance()
        metric_res["dynamic_loss"] = eval_loss_dynamic.performance().item()
        metric_res["static_loss"] = eval_loss_static.performance().item()
        metric_res["final_loss"] = eval_loss_final.performance().item()

        if epoch is not None:
            printt(f"{epoch}\t'valid'\tTime:{eval_time:.2f}\tMetricT: {metric_time:.2f}")
            for metric, value in metric_res.items():
                printt(f"{metric}: {value:.4f}")
            print(f"{datetime.datetime.today()}")
            for k, v in metric_res.items():
                self.writer.add_scalar(f"{k}/valid", v, epoch)

        return metric_res

    def predict(self, dataset: Dataset, name: str):
        """Output the prediction on given data.

        Args:
            datasets (Dataset): The dataset to predict on.
            name (str): The results would be saved to {name}_pre.pkl.

        Returns:
            np.ndarray: The model output.
        """
        self.eval()
        preds = []
        dataset.load()
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=8)
        for _, (data, _) in enumerate(loader):
            if use_cuda():
                data = to_torch(data, device="cuda")
            pred = self(data)[2]
            if self.out_ranges is not None:
                pred = pred[:, self.out_ranges]
            pred = pred.squeeze(-1).cpu().detach().numpy()
            preds.append(pred)

        prediction = np.concatenate(preds, axis=0)
        data_length = len(dataset.get_index())
        prediction = prediction.reshape(data_length, -1)

        prediction = pd.DataFrame(data=prediction, index=dataset.get_index())
        prediction.to_pickle(self.checkpoint_dir / (name + "_pre.pkl"))
        return prediction
