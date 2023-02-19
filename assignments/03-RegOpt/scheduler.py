from typing import List
import torch
from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    """
    Create a customized learning rate scheduler.
    """

    def __init__(
        self, optimizer: torch.optim.Optimizer, lr_lambda: object, last_epoch=-1
    ) -> None:
        """
        Create a new scheduler.

        Note to students: You can change the arguments to this constructor,
        if you need to add new parameters.

        """
        # ... Your Code Here ...
        self.optimizer = optimizer
        self.lr_lambdas = [lr_lambda] * len(optimizer.param_groups)
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
            for lmbda in self.lr_lambdas:
                for i in range(self.last_epoch - 1):
                    curr_lr += lmbda(i) / (self.last_epoch - i)
                curr_lr += lmbda(self.last_epoch)
            if self.last_epoch > 0 and curr_lr > 0:
                lrs.append(base_lr * curr_lr / self.last_epoch)
            else:
                lrs.append(base_lr)

        return lrs
