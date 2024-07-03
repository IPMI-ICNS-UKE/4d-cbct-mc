from __future__ import annotations

import bisect
import inspect
import logging
import math
import os
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Sequence

import numpy as np
import PIL
import torch
import torch.nn as nn
import yaml
from aim.sdk.repo import Repo, RepoStatus, Run
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from vroc.common_types import Number, PathLike
from vroc.decorators import convert, timing
from vroc.helper import concat_dicts
from vroc.logger import LoggerMixin


class BaseTrainer(ABC, LoggerMixin):
    METRICS = {}

    @convert("run_folder", converter=Path)
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer | None = None,
        train_loader: DataLoader | None = None,
        run_folder: PathLike | None = None,
        aim_repo: PathLike | None = None,
        loss_function: nn.Module | None = None,
        val_loader: DataLoader = None,
        experiment_name: str | None = None,
        device: str = "cuda",
    ):
        run_folder: Path

        self.model = model
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.run_folder = run_folder
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.is_trainable = (
            self.optimizer is not None
            and self.train_loader is not None
            and self.run_folder is not None
        )

        if self.is_trainable:
            self.optimizer.zero_grad()

            self.run_folder.mkdir(parents=True, exist_ok=True)

            if aim_repo:
                self._aim_repo = aim_repo
            else:
                self._aim_repo = self.run_folder

            repo = BaseTrainer._get_aim_repo(self._aim_repo)
            self.aim_run = Run(repo=repo, experiment=experiment_name)
            self._set_run_params()
            self._save_trainer_source_code()

            self._output_folder = (
                self.run_folder
                / f"{datetime.now().isoformat()}_run_{self.aim_run.hash}"
            )
            self._output_folder.mkdir(parents=True, exist_ok=True)

            self._model_folder = self._output_folder / "models"
            self._model_folder.mkdir(parents=True, exist_ok=True)

            self._image_folder = self._output_folder / "images"
            self._image_folder.mkdir(parents=True, exist_ok=True)

            # metric tracking and model saving
            self.val_model_saver = BestModelSaver(
                tracked_metrics=self.METRICS,
                model=self.model,
                optimizer=self.optimizer,
                output_folder=self._model_folder / "validation",
                top_k=10,
            )
            self.train_model_saver = BestModelSaver(
                tracked_metrics={"step": MetricType.LARGER_IS_BETTER},
                model=self.model,
                optimizer=self.optimizer,
                output_folder=self._model_folder / "training",
                top_k=100,
            )

        self.model = model.to(device=self.device)
        self.loss_function = loss_function

        self.scaler = torch.cuda.amp.GradScaler()

        # dict for metric history (for, e.g., calculating running means)
        self._metric_history = {}

        # training step and epoch tracking
        self.i_step = 0
        self.i_epoch = 0

    def save_image(self, image: PIL.Image, name: str) -> Path:
        filename = f"step_{self.i_step:09d}__{name}.png"
        filepath = self._image_folder / filename
        image.save(filepath)

        return filepath

    def _save_trainer_source_code(self) -> str:
        source_code = inspect.getsource(self.__class__)
        self.aim_run["source_code"] = source_code

    def add_run_params(self, params: dict):
        _params = self.aim_run["params"]
        _params.update(params)

        self.aim_run["params"] = params

    def _set_run_params(self):
        params = {
            "model": self.model.__class__.__name__,
            "optimizer": self.optimizer.__class__.__name__,
        }

        try:
            params["model_config"] = self.model.config
        except AttributeError:
            pass

        self.aim_run["params"] = params

    @staticmethod
    def _get_aim_repo(path: PathLike) -> Repo:
        path = str(path)
        repo_status = Repo.check_repo_status(path)
        if repo_status == RepoStatus.MISSING:
            repo = Repo.from_path(path, init=True)
        elif repo_status == RepoStatus.UPDATE_REQUIRED:
            raise RuntimeError("Please upgrade repo")
        else:
            repo = Repo.from_path(path)

        return repo

    def _prefixed_log(self, message, context: str, level: int):
        context = context.upper()
        prefix = f"[Step {self.i_step:<6} | {context:>6}]"
        self.logger.log(level=level, msg=f"{prefix} {message}", stacklevel=3)

    def log_info(self, message, context: str):
        self._prefixed_log(message, context=context, level=logging.INFO)

    def log_debug(self, message, context: str):
        self._prefixed_log(message, context=context, level=logging.DEBUG)

    def _track_metrics(self, metrics: dict, context: dict | None = None):
        for metric_name, metic_value in metrics.items():
            # metric value may be a list containing a value for each sample
            if (
                isinstance(metic_value, list)
                and len(metic_value) > 0
                and isinstance(metic_value[0], (int, float))
            ):
                metric_name = f"mean_batch_{metric_name}"
                metic_value = np.mean(metic_value)

            if isinstance(metic_value, (int, float)):
                subset = context["subset"]
                self._metric_history.setdefault(subset, {})
                history = self._metric_history[subset].setdefault(
                    metric_name, deque(maxlen=100)
                )
                history.append(metic_value)
                self.log_info(
                    f"Running mean (n={len(history)}) of "
                    f"{metric_name}: {np.mean(history):.6f}",
                    context=subset,
                )

            self.aim_run.track(
                metic_value,
                epoch=self.i_epoch,
                step=self.i_step,
                name=metric_name,
                context=context,
            )

    @timing()
    def train_one_epoch(self):
        self.model.train()
        self.optimizer.zero_grad()
        self.log_info("started epoch", context="TRAIN")

        try:
            n_train_batches = len(self.train_loader)
        except TypeError:
            n_train_batches = None

        for i_batch, data in enumerate(self.train_loader, start=1):
            batch_metrics = self._train_on_batch(data)

            batch_metrics_formatted = self._format_metric_dict(batch_metrics)
            self.log_debug(
                f"Batch {i_batch}/{n_train_batches}, "
                f"metrics: {batch_metrics_formatted}",
                context="TRAIN",
            )

            self._track_metrics(batch_metrics, context={"subset": "train"})

            yield batch_metrics

            self.i_step += 1
        self.i_epoch += 1
        self.log_info("finished epoch", context="TRAIN")

    def _format_metric_dict(self, metrics: dict) -> str:
        metrics_formatted = {
            key: f"{value:.4f}" if isinstance(value, float) else str(value)
            for key, value in metrics.items()
        }
        formatted_str = " / ".join(
            f"{key}: {value}" for key, value in metrics_formatted.items()
        )

        return formatted_str

    @timing()
    def validate(self):
        if self.val_loader is None:
            raise RuntimeError("No validation loader given")

        self.log_info("started validation", context="VAL")
        # set to eval mode
        self.model.eval()

        try:
            n_val_batches = len(self.val_loader)
        except TypeError:
            n_val_batches = None

        metrics = []
        for i_batch, data in enumerate(self.val_loader):
            batch_metrics = self._validate_on_batch(data)
            batch_metrics_formatted = self._format_metric_dict(batch_metrics)

            self.log_debug(
                f"Batch {i_batch}/{n_val_batches}, "
                f"metrics: {batch_metrics_formatted}",
                context="VAL",
            )
            metrics.append(batch_metrics)

        metrics = concat_dicts(metrics, extend_lists=True)

        self._track_metrics(metrics, context={"subset": "val"})
        self.val_model_saver.track(metrics, step=self.i_step)
        self.log_info("finished validation", context="VAL")

        # set to train mode again
        self.model.train()

    @timing()
    def run(
        self,
        steps: int = 10_000,
        validation_interval: int = 1000,
        save_interval: int = 1000,
    ):
        self.log_info("started run", context="RUN")
        self.i_step = 0
        self.i_epoch = 0
        training_finished = False
        while True:
            # train one epoch, i.e. one dataset iteration
            for batch_metrics in self.train_one_epoch():
                if (
                    self.i_step > 0
                    and self.val_loader is not None
                    and validation_interval > 0
                    and (
                        self.i_step % validation_interval == 0
                        or self.i_step == steps - 1
                    )
                ):
                    # run validation at given intervals (if validation loader is given)
                    self.validate()
                if save_interval and (
                    self.i_step % save_interval == 0 or self.i_step == steps - 1
                ):
                    # save model without validation
                    self.train_model_saver.track(
                        {"step": self.i_step}, step=self.i_step
                    )

                if self.i_step >= steps:
                    # stop training
                    training_finished = True
                    break

            if training_finished:
                self.log_info("finished training", context="TRAIN")
                break

    @timing()
    def _train_on_batch(self, data: dict) -> dict:
        self.train_on_batch_pre_hook(data=data)
        batch_results = self.train_on_batch(data=data)
        self.train_on_batch_post_hook(data=data, batch_results=batch_results)

        return batch_results

    @timing()
    def _validate_on_batch(self, data: dict) -> dict:
        return self.validate_on_batch(data=data)

    @abstractmethod
    def train_on_batch(self, data: dict) -> dict:
        raise NotImplementedError

    @abstractmethod
    def validate_on_batch(self, data: dict) -> dict:
        raise NotImplementedError

    def train_on_batch_pre_hook(self, data: dict):
        return

    def train_on_batch_post_hook(self, data: dict, batch_results: dict):
        return


class MetricType(Enum):
    SMALLER_IS_BETTER = auto()
    LARGER_IS_BETTER = auto()


class BestModelSaver(LoggerMixin):
    @convert("output_folder", converter=Path)
    def __init__(
        self,
        tracked_metrics: dict[str, MetricType],
        model: nn.Module,
        output_folder: PathLike,
        top_k: int = 1,
        model_name: str | None = None,
        optimizer: Optimizer | None = None,
        move_to_cpu: bool = True,
    ):
        output_folder: Path

        self.model = model
        self.output_folder = output_folder
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.top_k = top_k
        self.model_name = model_name
        self.optimizer = optimizer
        self.move_to_cpu = move_to_cpu

        self._tracked_metrics = tracked_metrics
        self._best = {
            metric_name: {"values": [], "models": []}
            for metric_name in tracked_metrics.keys()
        }

    def _track_metric(
        self, metric_name: str, metric_value: Number, model_filepath: Path
    ):
        best_values = self._best[metric_name]["values"]
        best_models = self._best[metric_name]["models"]

        if self._tracked_metrics[metric_name] == MetricType.LARGER_IS_BETTER:
            factor = -1
        else:
            factor = 1

        insert_index = bisect.bisect_left(
            best_values, factor * metric_value, key=lambda x: factor * x
        )
        best_values.insert(insert_index, metric_value)
        best_models.insert(insert_index, model_filepath)
        # restrict to top k
        self._best[metric_name]["values"] = best_values[: self.top_k]
        self._best[metric_name]["models"] = best_models[: self.top_k]

    def _save_model(self, output_filepath: Path):
        model_state = self.model.state_dict()
        if self.move_to_cpu:
            model_state = {k: v.cpu() for k, v in model_state.items()}
        state = {
            "model": model_state,
            "optimizer": self.optimizer.state_dict() if self.optimizer else None,
        }
        torch.save(state, output_filepath)

        with open(output_filepath.parent / "models.yaml", "wt") as f:
            data = {
                metric_name: {
                    value: model_filepath.name
                    for (value, model_filepath) in zip(
                        self._best[metric_name]["values"],
                        self._best[metric_name]["models"],
                    )
                }
                for metric_name in self._tracked_metrics
            }

            yaml.dump(data, f)

        self.logger.info(f"Saved model to {output_filepath}")

    @property
    def referenced_models(self):
        return set(
            model
            for metric_name in self._tracked_metrics.keys()
            for model in self._best[metric_name]["models"]
        )

    def _model_is_referenced(self, model_filepath: Path) -> bool:
        return model_filepath in self.referenced_models

    def _delete_non_referenced_models(self):
        saved_models = set(self.output_folder.glob("*.pth"))

        for model_filepath in saved_models - self.referenced_models:
            self.logger.info(f"Deleting unreferenced model {model_filepath}")
            os.remove(model_filepath)

    def track(self, metrics: dict[str, Number | Sequence[Number]], step: int):
        if self.model_name:
            model_filepath = (
                self.output_folder / f"{self.model_name.lower()}_step_{step:03d}.pth"
            )
        else:
            model_filepath = self.output_folder / f"step_{step:03d}.pth"

        for metric_name, metric_value in metrics.items():
            if metric_name not in self._tracked_metrics:
                # skip this metric (tracking disabled for this metric)
                continue

            if isinstance(metric_value, (np.ndarray, list, tuple)) and isinstance(
                metric_value[0], (int, float)
            ):
                # multiple metric valued passed, e.g. for each sample in batch,
                # take the mean
                metric_value = float(np.mean(metric_value))
            else:
                # convert to float if value is numpy 0-dim float
                metric_value = float(metric_value)

            self._track_metric(
                metric_name=metric_name,
                metric_value=metric_value,
                model_filepath=model_filepath,
            )

        if model_filepath in self.referenced_models:
            self._save_model(model_filepath)
            self.logger.info(f"Saved model to {model_filepath}")

        self._delete_non_referenced_models()
