from typing import Callable

import torch
import torch.nn as nn


class EmaStudentTeacherUpdate:
    def __init__(self, momentum: float | Callable = 0.9):
        self.momentum = momentum

    def __call__(self, student: nn.Module, teacher: nn.Module, iteration: int = None):
        with torch.no_grad():
            if callable(self.momentum):
                momentum = self.momentum(iteration)
            else:
                momentum = self.momentum
            for student_p, teacher_p in zip(student.parameters(), teacher.parameters()):
                teacher_p.data.mul_(momentum).add_(
                    (1 - momentum) * student_p.detach().data
                )
