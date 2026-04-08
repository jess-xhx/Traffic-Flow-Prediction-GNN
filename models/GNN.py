from __future__ import annotations

import torch
import torch.nn as nn

from configs.gnn_config import ModelConfig
from utils.gnn_utils import Batch, Tensor, prepare_batch, scalar_int

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
            time_chunk_size=cfg.base_time_chunk_size,
            temporal_kernel_size=cfg.temporal_kernel_size,
            temporal_dropout=cfg.temporal_dropout,
            temporal_dilations=cfg.temporal_dilations,
            temporal_node_chunk_size=cfg.temporal_node_chunk_size,
        )

        self.recent = RecentResidualBank(
            bank_hidden_dim=cfg.bank_hidden_dim,
            recent_hidden_dim=cfg.recent_hidden_dim,
            calendar_hidden_dim=cfg.calendar_hidden_dim,
            use_speed_head=True,
            time_chunk_size=cfg.recent_time_chunk_size,
        )

        self.event = EventResidualInjector(
            hidden_dim=cfg.bank_hidden_dim,
            event_dim=cfg.event_dim,
            use_speed_head=True,
            future_chunk_size=cfg.event_future_chunk_size,
        )

        # 按配置决定是否启用 checkpoint
        self.base.set_checkpoint_enabled(cfg.enable_base_checkpoint)
        self.recent.set_checkpoint_enabled(cfg.enable_recent_checkpoint)

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
            H_base_bank, recent_speed_seq, edge_index
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
        event_weekday: int | Tensor,
        event_slot: int | Tensor,
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

        weekday = scalar_int(event_weekday)
        slot = scalar_int(event_slot)

        event_out = self.event.inject_week(
            H_adapted_bank=H_adapted_bank,
            event_weekday=weekday,
            event_slot=slot,
            event_vector=event_vector,
            edge_index=edge_index,
        )

        return {
            'H_base_bank': recent_out['H_base_bank'],
            'delta_recent_bank': recent_out['delta_recent_bank'],
            'H_adapted_bank': H_adapted_bank,
            'delta_seed': event_out['delta_seed'],
            'diffused_event': event_out['diffused_event'],
            'delta_event_bank': event_out['delta_event_bank'],
            'H_final_bank': event_out['H_final_bank'],
            'pred_speed_bank': event_out['pred_speed_bank'],
            'event_bank_mask': event_out['event_bank_mask'],
        }

    def forward_recent_from_base(
        self,
        H_base_bank: Tensor,
        recent_speed_seq: Tensor,
        edge_index: Tensor,
        return_full: bool = True,
    ) -> dict[str, Tensor]:
        delta_recent_bank, H_adapted_bank, pred_speed_bank = self.recent.build_delta_bank(
            H_base_bank, recent_speed_seq, edge_index
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
    
    def forward_event_from_adapted(
        self,
        H_base_bank: Tensor,
        delta_recent_bank: Tensor,
        H_adapted_bank: Tensor,
        edge_index: Tensor,
        event_weekday: int | Tensor,
        event_slot: int | Tensor,
        event_vector: Tensor,
    ) -> dict[str, Tensor]:
        weekday = scalar_int(event_weekday)
        slot = scalar_int(event_slot)

        event_out = self.event.inject_week(
            H_adapted_bank=H_adapted_bank,
            event_weekday=weekday,
            event_slot=slot,
            event_vector=event_vector,
            edge_index=edge_index,
        )

        return {
            'H_base_bank': H_base_bank,
            'delta_recent_bank': delta_recent_bank,
            'H_adapted_bank': H_adapted_bank,
            'delta_seed': event_out['delta_seed'],
            'diffused_event': event_out['diffused_event'],
            'delta_event_bank': event_out['delta_event_bank'],
            'H_final_bank': event_out['H_final_bank'],
            'pred_speed_bank': event_out['pred_speed_bank'],
            'event_bank_mask': event_out['event_bank_mask'],
        }

    def forward_joint_shared(
        self,
        x_static: Tensor,
        profile_feat: Tensor,
        recent_speed_seq: Tensor,
        edge_index: Tensor,
        event_weekday: int | Tensor,
        event_slot: int | Tensor,
        event_vector: Tensor,
    ) -> dict[str, Tensor]:
        base_out = self.forward_base(x_static, profile_feat, edge_index)
        H_base_bank = base_out['H_base_bank']

        recent_out = self.forward_recent_from_base(
            H_base_bank=H_base_bank,
            recent_speed_seq=recent_speed_seq,
            edge_index=edge_index,
            return_full=True,
        )

        event_out = self.forward_event_from_adapted(
            H_base_bank=H_base_bank,
            delta_recent_bank=recent_out['delta_recent_bank'],
            H_adapted_bank=recent_out['H_adapted_bank'],
            edge_index=edge_index,
            event_weekday=event_weekday,
            event_slot=event_slot,
            event_vector=event_vector,
        )

        return {
            'H_base_bank': H_base_bank,
            'base_pred_speed_bank': base_out['pred_speed_bank'],
            'delta_recent_bank': recent_out['delta_recent_bank'],
            'H_adapted_bank': recent_out['H_adapted_bank'],
            'recent_pred_speed_bank': recent_out['pred_speed_bank'],
            'delta_seed': event_out['delta_seed'],
            'diffused_event': event_out['diffused_event'],
            'delta_event_bank': event_out['delta_event_bank'],
            'H_final_bank': event_out['H_final_bank'],
            'event_pred_speed_bank': event_out['pred_speed_bank'],
            'event_bank_mask': event_out['event_bank_mask'],
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
                detach_base=detach_base,
                return_full=return_full,
            )

        if mode == 'joint_shared':
            return self.forward_joint_shared(
                x_static=b['x_static'],
                profile_feat=b['profile_feat'],
                recent_speed_seq=b['recent_speed_seq'],
                edge_index=b['edge_index'],
                event_weekday=b['event_weekday'],
                event_slot=b['event_slot'],
                event_vector=b['event_vector'],
            )

        if mode in {'event', 'joint'}:
            return self.forward_event(
                x_static=b['x_static'],
                profile_feat=b['profile_feat'],
                recent_speed_seq=b['recent_speed_seq'],
                edge_index=b['edge_index'],
                event_weekday=b['event_weekday'],
                event_slot=b['event_slot'],
                event_vector=b['event_vector'],
                detach_pre_event=detach_pre_event,
            )

        raise ValueError(f'未知 mode={mode}')

    @torch.no_grad()
    def build_event_bank(
        self,
        x_static: Tensor,
        profile_feat: Tensor,
        recent_speed_seq: Tensor,
        edge_index: Tensor,
        event_weekday: int,
        event_slot: int,
        event_vector: Tensor,
    ) -> Tensor:
        self.eval()
        recent_out = self.forward_recent(x_static, profile_feat, recent_speed_seq, edge_index)
        H_adapted_bank = recent_out['H_adapted_bank']

        event_out = self.event.inject_week(
            H_adapted_bank=H_adapted_bank,
            event_weekday=event_weekday,
            event_slot=event_slot,
            event_vector=event_vector,
            edge_index=edge_index,
        )
        return event_out['H_final_bank']

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


__all__ = ['TrafficGNNSystem']
