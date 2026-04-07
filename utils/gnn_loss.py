from __future__ import annotations

import torch
import torch.nn as nn

from .gnn_utils import masked_mae


Tensor = torch.Tensor


class TrafficGNNLoss(nn.Module):
    def __init__(self, lambda_recent_reg: float = 1e-4, lambda_event_reg: float = 1e-4):
        super().__init__()
        self.lambda_recent_reg = lambda_recent_reg
        self.lambda_event_reg = lambda_event_reg

    def base_loss(self, pred_speed_bank: Tensor, y_base_bank: Tensor, mask: Tensor | None = None) -> dict[str, Tensor]:
        loss = masked_mae(pred_speed_bank, y_base_bank, mask)
        return {
            'loss': loss,
            'mae': loss.detach(),
        }

    def recent_loss(
        self,
        pred_speed_bank: Tensor,
        y_future_bank: Tensor,
        delta_recent_bank: Tensor,
        mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        data_loss = masked_mae(pred_speed_bank, y_future_bank, mask)
        reg_loss = delta_recent_bank.pow(2).mean()
        total = data_loss + self.lambda_recent_reg * reg_loss
        return {
            'loss': total,
            'mae': data_loss.detach(),
            'reg': reg_loss.detach(),
        }

    def event_loss(
        self,
        pred_speed,
        target_speed,
        delta_event,
        mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        data_loss = masked_mae(pred_speed, target_speed, mask)
        reg_loss = delta_event.pow(2).mean()
        total = data_loss + self.lambda_event_reg * reg_loss
        return {
            'loss': total,
            'mae': data_loss.detach(),
            'reg': reg_loss.detach(),
        }
