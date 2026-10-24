import torch
import math
from torch.optim.lr_scheduler import _LRScheduler


class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iter, decay_iter=1, power=0.9, last_epoch=-1) -> None:
        self.decay_iter = decay_iter
        self.max_iter = max_iter
        self.power = power
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if self.last_epoch % self.decay_iter or self.last_epoch % self.max_iter:
            return self.base_lrs
        else:
            factor = (1 - self.last_epoch / float(self.max_iter)) ** self.power
            return [factor*lr for lr in self.base_lrs]


class WarmupLR(_LRScheduler):
    def __init__(self, optimizer, warmup_iter=500, warmup_ratio=5e-4, warmup='exp', last_epoch=-1) -> None:
        self.warmup_iter = warmup_iter
        self.warmup_ratio = warmup_ratio
        self.warmup = warmup
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        ratio = self.get_lr_ratio()
        return [ratio * lr for lr in self.base_lrs]

    def get_lr_ratio(self):
        return self.get_warmup_ratio() if self.last_epoch < self.warmup_iter else self.get_main_ratio()

    def get_main_ratio(self):
        raise NotImplementedError

    def get_warmup_ratio(self):
        assert self.warmup in ['linear', 'exp']
        alpha = self.last_epoch / self.warmup_iter

        return self.warmup_ratio + (1. - self.warmup_ratio) * alpha if self.warmup == 'linear' else self.warmup_ratio ** (1. - alpha)


class WarmupPolyLR(WarmupLR):
    def __init__(self, optimizer, power, max_iter, warmup_iter=500, warmup_ratio=5e-4, warmup='exp', last_epoch=-1) -> None:
        self.power = power
        self.max_iter = max_iter
        super().__init__(optimizer, warmup_iter, warmup_ratio, warmup, last_epoch)

    def get_main_ratio(self):
        real_iter = self.last_epoch - self.warmup_iter
        real_max_iter = self.max_iter - self.warmup_iter
        alpha = real_iter / real_max_iter

        return (1 - alpha) ** self.power


class WarmupExpLR(WarmupLR):
    def __init__(self, optimizer, gamma, interval=1, warmup_iter=500, warmup_ratio=5e-4, warmup='exp', last_epoch=-1) -> None:
        self.gamma = gamma
        self.interval = interval
        super().__init__(optimizer, warmup_iter, warmup_ratio, warmup, last_epoch)

    def get_main_ratio(self):
        real_iter = self.last_epoch - self.warmup_iter
        return self.gamma ** (real_iter // self.interval)


class WarmupCosineLR(WarmupLR):
    def __init__(self, optimizer, max_iter, eta_ratio=0, warmup_iter=500, warmup_ratio=5e-4, warmup='exp', last_epoch=-1) -> None:
        self.eta_ratio = eta_ratio
        self.max_iter = max_iter
        super().__init__(optimizer, warmup_iter, warmup_ratio, warmup, last_epoch)

    def get_main_ratio(self):
        real_iter = self.last_epoch - self.warmup_iter
        real_max_iter = self.max_iter - self.warmup_iter
        
        return self.eta_ratio + (1 - self.eta_ratio) * (1 + math.cos(math.pi * self.last_epoch / real_max_iter)) / 2



class WarmupPlateauLR:
    """
    Linear warmup per iteration, then ReduceLROnPlateau-style decay per epoch.

    step() is called after every iteration and only drives the warmup (from warmup_ratio * lr to lr).
    step_metric(value) is called after every validation: once the metric (to maximise) has not
    improved by more than `threshold` (absolute) for more than `patience` validations, the LR is
    multiplied by `factor`. Validations that fall inside the warmup are ignored, and the LR never
    goes below `min_lr`. Unlike the poly schedule it does not depend on the number of epochs, so
    it can be combined with early stopping.
    """

    def __init__(self, optimizer, warmup_iter, warmup_ratio, factor=0.5, patience=8, threshold=0.1, min_lr=1e-6) -> None:
        self.optimizer = optimizer
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.warmup_iter = warmup_iter
        self.warmup_ratio = warmup_ratio
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.min_lr = min_lr
        self.last_epoch = 0             # iteration counter (same name as in _LRScheduler)
        self.scale = 1.0                # product of the reductions applied so far
        self.best = -math.inf
        self.num_bad_epochs = 0
        self.num_reductions = 0
        self._set_lr()

    def in_warmup(self) -> bool:
        return self.last_epoch < self.warmup_iter

    def get_lr(self):
        if self.in_warmup():
            ratio = self.warmup_ratio + (1. - self.warmup_ratio) * self.last_epoch / self.warmup_iter
            return [ratio * lr for lr in self.base_lrs]
        return [max(self.scale * lr, self.min_lr) for lr in self.base_lrs]

    def _set_lr(self):
        for group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            group['lr'] = lr

    def step(self):
        self.last_epoch += 1
        self._set_lr()

    def step_metric(self, value) -> bool:
        """Feeds one validation result; returns True when the LR has just been reduced."""
        if self.in_warmup():
            return False
        if value > self.best + self.threshold:
            self.best = value
            self.num_bad_epochs = 0
            return False
        self.num_bad_epochs += 1
        if self.num_bad_epochs <= self.patience or min(self.get_lr()) <= self.min_lr:
            return False
        self.scale *= self.factor
        self.num_bad_epochs = 0
        self.num_reductions += 1
        self._set_lr()
        return True

    def state_dict(self):
        return {k: v for k, v in self.__dict__.items() if k != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        self._set_lr()


__all__ = ['polylr', 'warmuppolylr', 'warmupcosinelr', 'warmupsteplr', 'warmupplateau']


def get_scheduler(scheduler_name: str, optimizer, max_iter: int, power: int, warmup_iter: int, warmup_ratio: float, plateau_cfg: dict = None):
    assert scheduler_name in __all__, f"Unavailable scheduler name >> {scheduler_name}.\nAvailable schedulers: {__all__}"
    if scheduler_name == 'warmupplateau':
        # plateau_cfg: the SCHEDULER section of the config (FACTOR, PATIENCE, THRESHOLD, MIN_LR)
        plateau_cfg = plateau_cfg or {}
        return WarmupPlateauLR(optimizer, warmup_iter, warmup_ratio,
                               factor=float(plateau_cfg.get('FACTOR', 0.5)),
                               patience=int(plateau_cfg.get('PATIENCE', 8)),
                               threshold=float(plateau_cfg.get('THRESHOLD', 0.1)),
                               min_lr=float(plateau_cfg.get('MIN_LR', 1e-6)))
    if scheduler_name == 'warmuppolylr':
        return WarmupPolyLR(optimizer, power, max_iter, warmup_iter, warmup_ratio, warmup='linear')
    elif scheduler_name == 'warmupcosinelr':
        return WarmupCosineLR(optimizer, max_iter, warmup_iter=warmup_iter, warmup_ratio=warmup_ratio)
    return PolyLR(optimizer, max_iter)


if __name__ == '__main__':
    model = torch.nn.Conv2d(3, 16, 3, 1, 1)
    optim = torch.optim.SGD(model.parameters(), lr=1e-3)

    max_iter = 20000
    sched = WarmupPolyLR(optim, power=0.9, max_iter=max_iter, warmup_iter=200, warmup_ratio=0.1, warmup='exp', last_epoch=-1)

    lrs = []

    for _ in range(max_iter):
        lr = sched.get_lr()[0]
        lrs.append(lr)
        optim.step()
        sched.step()

    import matplotlib.pyplot as plt
    import numpy as np 

    plt.plot(np.arange(len(lrs)), np.array(lrs))
    plt.grid()
    plt.show()