from typing import List
import torch
from torch.optim.lr_scheduler import _LRScheduler
import math


class CustomLRScheduler(_LRScheduler):
    """
    Create a customized learning rate scheduler.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_epochs: int,
        lr_max: float,
        lr_min: float,
        last_epoch=-1,
    ) -> None:
        """
        Create a new scheduler.

        Note to students: You can change the arguments to this constructor,
        if you need to add new parameters.

        """
        # ... Your Code Here ...
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.lr_max = lr_max
        self.lr_min = lr_min
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        get learning rate
        """
        # Note to students: You CANNOT change the arguments or return type of
        # this function (because it is called internally by Torch)

        # ... Your Code Here ...
        # Here's our dumb baseline implementation:
        lrs = []
        for base_lr in self.base_lrs:
            curr_lr = 0
            for i in range(self.last_epoch - 1):
                curr_lr += (
                    (
                        (i / (self.last_epoch - i)) ** 0.5
                        + math.cos(math.pi * i / (self.last_epoch - i))
                    )
                    * base_lr
                    / 2
                )
            if self.last_epoch > 0:
                lr_candidate = curr_lr / (
                    self.last_epoch
                    + math.cos(math.pi * self.last_epoch / self.num_epochs)
                    + 1
                )
                if lr_candidate > self.lr_max:
                    lrs.append(self.lr_max)
                elif lr_candidate < self.lr_min:
                    lrs.append(self.lr_min)
                else:
                    lrs.append(lr_candidate)
            else:
                lrs.append(base_lr)
        return lrs
