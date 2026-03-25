from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from configs.gnn_config import ModelConfig
from utils.gnn_utils import Batch, Tensor, prepare_batch, scalar_int

try:
    from GNN_1_base import BaseWeeklyBank
    from GNN_2_recent import RecentResidualBank
    from GNN_3_event import EventResidualInjector
except ImportError:  # 兼容 package 导入
    from .GNN_1_base import BaseWeeklyBank
    from .GNN_2_recent import RecentResidualBank
    from .GNN_3_event import EventResidualInjector


class TrafficGNNSystem(nn.Module):
    """
    总模型：
      模块1：BaseWeeklyBank
      模块2：RecentResidualBank
      模块3：EventResidualInjector
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.base = BaseWeeklyBank(
            static_dim=cfg.static_dim,
            profile_dim=cfg.profile_dim,
            static_hidden_dim=cfg.static_hidden_dim,
            calendar_hidden_dim=cfg.calendar_hidden_dim,
            profile_hidden_dim=cfg.profile_hidden_dim,
            bank_hidden_dim=cfg.bank_hidden_dim,
            use_speed_head=True,
        )
        self.recent = RecentResidualBank(
            bank_hidden_dim=cfg.bank_hidden_dim,
            recent_hidden_dim=cfg.recent_hidden_dim,
            calendar_hidden_dim=cfg.calendar_hidden_dim,
            use_speed_head=True,
        )
        self.event = EventResidualInjector(
            hidden_dim=cfg.bank_hidden_dim,
            event_dim=cfg.event_dim,
            use_speed_head=True,
        )

    def forward_base(self, x_static: Tensor, profile_feat: Tensor, edge_index: Tensor) -> dict[str, Tensor]:
        H_base_bank, pred_speed_bank = self.base.build_bank(x_static, profile_feat, edge_index)
        return {
            'H_base_bank': H_base_bank,
            'pred_speed_bank': pred_speed_bank,
        }

    def forward_recent(
        self,
        x_static,
        profile_feat,
        recent_speed_seq,
        edge_index,
        detach_base: bool = False,
        return_full: bool = True,
    ):
        if detach_base:
            with torch.no_grad():
                base_out = self.forward_base(x_static, profile_feat, edge_index)
            H_base_bank = base_out['H_base_bank'].detach()
        else:
            base_out = self.forward_base(x_static, profile_feat, edge_index)
            H_base_bank = base_out['H_base_bank']

        delta_recent_bank, H_adapted_bank, pred_speed_bank = self.recent.build_delta_bank(
            H_base_bank, recent_speed_seq, edge_index,return_full=return_full
        )

        if return_full:
            return {
                'H_base_bank': H_base_bank,
                'delta_recent_bank': delta_recent_bank,
                'H_adapted_bank': H_adapted_bank,
                'pred_speed_bank': pred_speed_bank,
            }

        return {
            'delta_recent_bank': delta_recent_bank,
            'pred_speed_bank': pred_speed_bank,
        }

    def forward_event(
        self,
        x_static: Tensor,
        profile_feat: Tensor,
        recent_speed_seq: Tensor,
        edge_index: Tensor,
        target_weekday: int | Tensor,
        target_slot: int | Tensor,
        event_vector: Tensor,
        detach_pre_event: bool = False,
    ) -> dict[str, Tensor]:
        recent_out = self.forward_recent(
            x_static=x_static,
            profile_feat=profile_feat,
            recent_speed_seq=recent_speed_seq,
            edge_index=edge_index,
            detach_base=detach_pre_event,
        )

        H_adapted_bank = recent_out['H_adapted_bank']
        if detach_pre_event:
            H_adapted_bank = H_adapted_bank.detach()

        weekday = scalar_int(target_weekday)
        slot = scalar_int(target_slot)
        H_adapted_t = H_adapted_bank[weekday, slot]

        delta_event_t, H_final_t, pred_speed_t = self.event.inject(H_adapted_t, event_vector, edge_index)
        return {
            'H_base_bank': recent_out['H_base_bank'],
            'delta_recent_bank': recent_out['delta_recent_bank'],
            'H_adapted_bank': H_adapted_bank,
            'H_adapted_t': H_adapted_t,
            'delta_event_t': delta_event_t,
            'H_final_t': H_final_t,
            'pred_speed_t': pred_speed_t,
        }

    def forward(
        self,
        batch: Batch,
        mode: str = 'joint',
        detach_base: bool = False,
        detach_pre_event: bool = False,
        return_full: bool = True,
    ) -> dict[str, Tensor]:
        b = prepare_batch(batch, next(self.parameters()).device)

        if mode == 'base':
            return self.forward_base(
                x_static=b['x_static'],
                profile_feat=b['profile_feat'],
                edge_index=b['edge_index'],
            )

        if mode == 'recent':
            return self.forward_recent(
                x_static=b['x_static'],
                profile_feat=b['profile_feat'],
                recent_speed_seq=b['recent_speed_seq'],
                edge_index=b['edge_index'],
                detach_base=detach_base,return_full=return_full,
            )

        if mode in {'event', 'joint'}:
            return self.forward_event(
                x_static=b['x_static'],
                profile_feat=b['profile_feat'],
                recent_speed_seq=b['recent_speed_seq'],
                edge_index=b['edge_index'],
                target_weekday=b['target_weekday'],
                target_slot=b['target_slot'],
                event_vector=b['event_vector'],
                detach_pre_event=detach_pre_event,
            )

        raise ValueError(f'未知 mode={mode}')

    @torch.no_grad()
    def build_base_bank(self, x_static: Tensor, profile_feat: Tensor, edge_index: Tensor) -> Tensor:
        self.eval()
        out = self.forward_base(x_static, profile_feat, edge_index)
        return out['H_base_bank']

    @torch.no_grad()
    def build_adapted_bank(
        self,
        x_static: Tensor,
        profile_feat: Tensor,
        recent_speed_seq: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        self.eval()
        out = self.forward_recent(x_static, profile_feat, recent_speed_seq, edge_index)
        return out['H_adapted_bank']

    @torch.no_grad()
    def predict_one_horizon(
        self,
        x_static: Tensor,
        profile_feat: Tensor,
        recent_speed_seq: Tensor,
        edge_index: Tensor,
        target_weekday: int,
        target_slot: int,
        event_vector: Optional[Tensor] = None,
    ) -> Tensor:
        self.eval()
        recent_out = self.forward_recent(x_static, profile_feat, recent_speed_seq, edge_index)
        H_adapted_t = recent_out['H_adapted_bank'][target_weekday, target_slot]

        if event_vector is None:
            return self.recent.speed_head(H_adapted_t).squeeze(-1)

        _, _, pred_speed_t = self.event.inject(H_adapted_t, event_vector, edge_index)
        return pred_speed_t

    @torch.no_grad()
    def predict_multi_horizon(
        self,
        x_static: Tensor,
        profile_feat: Tensor,
        recent_speed_seq: Tensor,
        edge_index: Tensor,
        target_pairs: Sequence[Tuple[int, int]],
        event_vectors: Optional[Sequence[Optional[Tensor]]] = None,
    ) -> Tensor:
        self.eval()
        recent_out = self.forward_recent(x_static, profile_feat, recent_speed_seq, edge_index)
        H_adapted_bank = recent_out['H_adapted_bank']

        preds: list[Tensor] = []
        if event_vectors is None:
            event_vectors = [None] * len(target_pairs)

        for (weekday, slot), event_vector in zip(target_pairs, event_vectors):
            H_adapted_t = H_adapted_bank[weekday, slot]
            if event_vector is None:
                pred_speed_t = self.recent.speed_head(H_adapted_t).squeeze(-1)
            else:
                _, _, pred_speed_t = self.event.inject(H_adapted_t, event_vector, edge_index)
            preds.append(pred_speed_t)

        return torch.stack(preds, dim=0)


__all__ = ['TrafficGNNSystem']
