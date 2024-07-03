from abc import ABC, abstractmethod
from typing import Callable

import numpy as np


class CosineScheduler:
    def __init__(
        self,
        final_value: float,
        base_value: float,
        total_iterations: int,
        warm_up_iterations: int = 0,
        warm_up_starting_value: float = 0.0,
    ):
        warmup_schedule = np.array([])
        if warm_up_iterations:
            warmup_schedule = np.linspace(
                warm_up_starting_value, base_value, warm_up_iterations
            )
        cos_iters = np.arange(total_iterations - warm_up_iterations)

        schedule = final_value + 0.5 * (base_value - final_value) * (
            1 + np.cos(np.pi * cos_iters / len(cos_iters))
        )

        self.schedule = np.concatenate((warmup_schedule, schedule))
        self.steps = 0

    def __call__(self, step=None) -> float:
        if step:
            self.steps = step

        if self.steps < len(self.schedule):
            value = self.schedule[self.steps]
        else:
            value = self.schedule[-1]

        self.steps += 1

        return value


class SimpleScheduler(ABC):
    def __init__(self, scheduler: Callable):
        self.scheduler = scheduler
        self.current_lr = None

    @abstractmethod
    def step(self, *args, **kwargs) -> float:
        """Needs to implement call function for manipulation."""
        pass


class OptimizerScheduler(SimpleScheduler):
    def __init__(self, scheduler: Callable, optimizer: object = None):
        super().__init__(scheduler=scheduler)
        self.optimizer = optimizer

    @abstractmethod
    def step(self, *args, **kwargs) -> float:
        """Needs to implement call function for manipulation."""
        pass


class LRScheduler(OptimizerScheduler):
    def step(self, *args, **kwargs) -> float:
        lr = self.scheduler()
        self.current_lr = lr
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = lr
        return lr
