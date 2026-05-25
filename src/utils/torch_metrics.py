import torch
from torch import Tensor
from torchmetrics import Metric, MeanAbsoluteError
from torchmetrics.regression import MeanSquaredError, SpearmanCorrCoef, R2Score
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.functional.regression.spearman import (
    _spearman_corrcoef_compute,
    _spearman_corrcoef_update,
)
from torchmetrics.functional.regression.r2 import _r2_score_update

from collections import defaultdict
import numpy as np

class MeanMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, loss: torch.Tensor):
        self.sum += torch.sum(loss)
        self.total += loss.numel()

    def compute(self):
        if self.total == 0:
            return torch.tensor(0.0, device=self.sum.device)
        return self.sum.float() / self.total

def dist_reduce_custom(inp):
    return torch.sum(inp, dim=1)


class MeanMetricByPos(Metric):
    def __init__(self):
        super().__init__()
        self.add_state(
            "sum",
            default=torch.zeros(10, 10),
            dist_reduce_fx=dist_reduce_custom,
        )
        self.add_state(
            "total",
            default=torch.zeros(10, 10),
            dist_reduce_fx=dist_reduce_custom,
        )

    def update(self, loss, freq):
        for pos in loss.keys():
            self.sum[pos] += loss[pos]
            self.total[pos] += freq[pos]

    def compute(self):
        return torch.div(self.sum, self.total)

class MaskedMeanSquaredError(MeanSquaredError):
    def update(self, preds: Tensor, target: Tensor, mask: Tensor) -> None:
        preds = preds.float()
        target = target.float()
        mask = mask.float()

        squared_error = torch.square(preds - target)
        masked_squared_error = squared_error * mask

        self.sum_squared_error += torch.sum(masked_squared_error)
        self.total += torch.sum(mask).to(torch.long)


class MaskedMeanAbsoluteError(MeanAbsoluteError):
    def update(self, preds: Tensor, target: Tensor, mask: Tensor) -> None:
        preds = preds.float()
        target = target.float()
        mask = mask.float()

        absolute_error = torch.abs(preds - target)
        masked_absolute_error = absolute_error * mask

        self.sum_abs_error += torch.sum(masked_absolute_error)
        self.total += torch.sum(mask).to(torch.long)


class MaskedPearsonCorrCoef(Metric):
    full_state_update = False
    higher_is_better = True

    def __init__(self):
        super().__init__()
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor, mask: Tensor) -> None:
        preds = preds[mask == 1].detach().float()
        target = target[mask == 1].detach().float()

        if preds.numel() == 0 or target.numel() == 0:
            return
        if torch.isnan(preds).any() or torch.isnan(target).any():
            return

        self.preds.append(preds)
        self.target.append(target)

    def compute(self) -> Tensor:
        if len(self.preds) == 0 or len(self.target) == 0:
            return torch.tensor(0.0, device=self.device)

        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target)

        if preds.numel() < 2 or target.numel() < 2:
            return torch.tensor(0.0, device=preds.device)

        preds_centered = preds - preds.mean()
        target_centered = target - target.mean()
        numerator = torch.sum(preds_centered * target_centered)
        denominator = torch.sqrt(
            torch.sum(preds_centered ** 2) * torch.sum(target_centered ** 2)
        )

        if denominator <= 0:
            return torch.tensor(0.0, device=preds.device)

        return numerator / denominator


class MaskedSpearmanCorrCoeff(SpearmanCorrCoef):
    def update(self, preds: Tensor, target: Tensor, mask: Tensor) -> None:
        preds = preds[mask == 1]
        target = target[mask == 1]

        if preds.numel() == 0 or target.numel() == 0:
            return
        if torch.isnan(preds).any() or torch.isnan(target).any():
            return

        preds, target = _spearman_corrcoef_update(
            preds, target, num_outputs=self.num_outputs
        )
        self.preds.append(preds)
        self.target.append(target)

    def compute(self) -> Tensor:
        if len(self.preds) == 0 or len(self.target) == 0:
            return torch.tensor(0.0, device=self.device)

        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target)

        if preds.numel() < 2 or target.numel() < 2:
            return torch.tensor(0.0, device=preds.device)

        return _spearman_corrcoef_compute(preds, target, eps=1e-4)


class MaskedR2Score(R2Score):
    def update(self, preds: Tensor, target: Tensor, mask: Tensor) -> None:
        masked_preds = preds[mask == 1]
        masked_target = target[mask == 1]

        if len(masked_preds) < 2:
            return

        sum_squared_error, sum_error, residual, total = _r2_score_update(
            masked_preds, masked_target
        )
        self.sum_squared_error += sum_squared_error
        self.sum_error += sum_error
        self.residual += residual
        self.total += total