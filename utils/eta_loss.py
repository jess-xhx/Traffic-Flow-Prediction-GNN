from __future__ import annotations

import math

import torch
import torch.nn as nn


class ETALoss(nn.Module):
    def __init__(self, use_uncertainty: bool = False, uncertainty_weight: float = 1.0):
        super().__init__()
        self.use_uncertainty = bool(use_uncertainty)
        self.uncertainty_weight = float(uncertainty_weight)

    def forward(self, pred: dict[str, torch.Tensor], target_eta_minutes: torch.Tensor) -> dict[str, torch.Tensor]:
        target = target_eta_minutes.float().view(-1)
        eta_pred = pred["eta_minutes"].float().view(-1)

        abs_err = torch.abs(eta_pred - target)
        mae = abs_err.mean()
        mse = torch.mean((eta_pred - target) ** 2)
        rmse = torch.sqrt(mse.clamp_min(0.0))

        out = {
            "loss": mae,
            "mae_minutes": mae,
            "mse_minutes": mse,
            "rmse_minutes": rmse,
        }

        if self.use_uncertainty and "log_sigma" in pred:
            log_sigma = pred["log_sigma"].float().view(-1).clamp(min=-6.0, max=6.0)
            sigma2 = torch.exp(2.0 * log_sigma).clamp_min(1e-6)
            nll = 0.5 * (((eta_pred - target) ** 2) / sigma2 + 2.0 * log_sigma + math.log(2.0 * math.pi))
            out["nll"] = nll.mean()
            out["loss"] = out["loss"] + self.uncertainty_weight * out["nll"]

        return out


__all__ = ["ETALoss"]
