from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from configs.eta_config import SameCityETAConfig
from models.GNN import TrafficGNNSystem


SLOTS_PER_DAY = 288
SLOT_SECONDS = 300.0


def _ensure_batch_2d(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1:
        return x.unsqueeze(0)
    return x


def _ensure_batch_1d_long(x: torch.Tensor | int) -> torch.Tensor:
    if isinstance(x, int):
        return torch.tensor([x], dtype=torch.long)
    if x.dim() == 0:
        return x.view(1).long()
    if x.dim() == 2 and x.size(1) == 1:
        return x[:, 0].long()
    return x.long()


class TripTokenEncoder(nn.Module):
    def __init__(self, cfg, bank_hidden_dim: int, d_model: int):
        super().__init__()
        self.cfg = cfg
        self.embeddings = nn.ModuleList([
            nn.Embedding(cfg.cat_bucket_size, cfg.cat_emb_dim)
            for _ in range(cfg.num_categories)
        ])
        cat_total_dim = cfg.num_categories * cfg.cat_emb_dim
        self.net = nn.Sequential(
            nn.LayerNorm(cfg.numeric_dim + cat_total_dim + bank_hidden_dim),
            nn.Linear(cfg.numeric_dim + cat_total_dim + bank_hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, d_model),
        )

    def forward(self, trip_num_feat: torch.Tensor, trip_cat_feat: torch.Tensor, bank_context: torch.Tensor) -> torch.Tensor:
        trip_num_feat = _ensure_batch_2d(trip_num_feat)
        trip_cat_feat = _ensure_batch_2d(trip_cat_feat).long()
        if bank_context.dim() == 1:
            bank_context = bank_context.unsqueeze(0)
        cat_embs = [emb(trip_cat_feat[:, idx]) for idx, emb in enumerate(self.embeddings)]
        x = torch.cat([trip_num_feat, torch.cat(cat_embs, dim=-1), bank_context], dim=-1)
        return self.net(x)


class RouteTimeEncoder(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.weekday_emb = nn.Embedding(7, max(4, out_dim // 4))
        self.slot_emb = nn.Embedding(SLOTS_PER_DAY, max(8, out_dim // 2))
        in_dim = self.weekday_emb.embedding_dim + self.slot_emb.embedding_dim + 5
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, weekday_ids: torch.Tensor, slot_ids: torch.Tensor, delta_seconds: torch.Tensor) -> torch.Tensor:
        weekday_emb = self.weekday_emb(weekday_ids.long())
        slot_ids = slot_ids.long().clamp(min=0, max=SLOTS_PER_DAY - 1)
        slot_emb = self.slot_emb(slot_ids)
        slot_float = slot_ids.float()
        weekday_float = weekday_ids.float()
        day_angle = 2.0 * math.pi * slot_float / float(SLOTS_PER_DAY)
        week_angle = 2.0 * math.pi * weekday_float / 7.0
        cyc = torch.stack(
            [
                torch.sin(day_angle),
                torch.cos(day_angle),
                torch.sin(week_angle),
                torch.cos(week_angle),
                delta_seconds.float() / 3600.0,
            ],
            dim=-1,
        )
        feat = torch.cat([weekday_emb, slot_emb, cyc], dim=-1)
        return self.proj(feat)


class EntryTimeScheduler(nn.Module):
    def __init__(self, min_speed_kmh: float = 5.0):
        super().__init__()
        self.min_speed_kmh = float(min_speed_kmh)

    def forward(
        self,
        current_weekday: torch.Tensor,
        current_slot: torch.Tensor,
        route_edge_lengths_m: torch.Tensor,
        route_speeds_kmh: torch.Tensor,
        current_edge_remaining_ratio: torch.Tensor | None = None,
        route_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        current_weekday = _ensure_batch_1d_long(current_weekday).to(route_edge_lengths_m.device)
        current_slot = _ensure_batch_1d_long(current_slot).to(route_edge_lengths_m.device)
        if route_edge_lengths_m.dim() == 1:
            route_edge_lengths_m = route_edge_lengths_m.unsqueeze(0)
        if route_speeds_kmh.dim() == 1:
            route_speeds_kmh = route_speeds_kmh.unsqueeze(0)
        if route_mask is None:
            route_mask = torch.ones_like(route_edge_lengths_m, dtype=torch.bool)
        if current_edge_remaining_ratio is None:
            current_edge_remaining_ratio = torch.ones(route_edge_lengths_m.size(0), device=route_edge_lengths_m.device)
        current_edge_remaining_ratio = current_edge_remaining_ratio.float().view(-1, 1)

        safe_speed = torch.clamp(route_speeds_kmh.float(), min=self.min_speed_kmh)
        speed_mps = safe_speed * (1000.0 / 3600.0)
        adj_lengths = route_edge_lengths_m.float().clone()
        adj_lengths[:, :1] = adj_lengths[:, :1] * current_edge_remaining_ratio
        travel_seconds = torch.where(route_mask, adj_lengths / speed_mps.clamp_min(0.5), torch.zeros_like(adj_lengths))

        enter_delta_seconds = torch.zeros_like(travel_seconds)
        if travel_seconds.size(1) > 1:
            enter_delta_seconds[:, 1:] = torch.cumsum(travel_seconds[:, :-1], dim=1)

        flat_slots = current_weekday.view(-1, 1) * SLOTS_PER_DAY + current_slot.view(-1, 1)
        enter_total_slots = flat_slots + torch.floor(enter_delta_seconds / SLOT_SECONDS).long()
        weekday_ids = torch.remainder(enter_total_slots // SLOTS_PER_DAY, 7)
        slot_ids = torch.remainder(enter_total_slots, SLOTS_PER_DAY)
        return {
            "weekday_ids": weekday_ids,
            "slot_ids": slot_ids,
            "enter_delta_seconds": enter_delta_seconds,
            "travel_seconds": travel_seconds,
        }


class RouteTokenEncoder(nn.Module):
    def __init__(self, cfg, d_model: int):
        super().__init__()
        self.cfg = cfg
        self.time_encoder = RouteTimeEncoder(cfg.time_emb_dim)
        base_numeric_dim = 5 + cfg.extra_numeric_dim
        in_dim = cfg.static_dim + cfg.bank_hidden_dim + 1 + cfg.time_emb_dim + base_numeric_dim + cfg.turn_feat_dim
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, d_model),
        )

    def forward(
        self,
        edge_static_feat: torch.Tensor,
        edge_dyn_feat: torch.Tensor,
        edge_speed_kmh: torch.Tensor,
        edge_lengths_m: torch.Tensor,
        weekday_ids: torch.Tensor,
        slot_ids: torch.Tensor,
        delta_seconds: torch.Tensor,
        route_remaining_ratio: torch.Tensor,
        route_cumulative_ratio: torch.Tensor,
        route_turn_feat: torch.Tensor | None = None,
        route_extra_feat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        edge_speed_kmh = edge_speed_kmh.unsqueeze(-1)
        edge_lengths_km = edge_lengths_m.unsqueeze(-1) / 1000.0
        rel_hours = delta_seconds.unsqueeze(-1) / 3600.0
        numeric_core = torch.cat(
            [
                edge_speed_kmh,
                edge_lengths_km,
                route_remaining_ratio.unsqueeze(-1),
                route_cumulative_ratio.unsqueeze(-1),
                rel_hours,
            ],
            dim=-1,
        )
        if route_extra_feat is None:
            route_extra_feat = torch.zeros((*numeric_core.shape[:-1], self.cfg.extra_numeric_dim), device=numeric_core.device)
        if route_turn_feat is None:
            route_turn_feat = torch.zeros((*numeric_core.shape[:-1], self.cfg.turn_feat_dim), device=numeric_core.device)
        time_feat = self.time_encoder(weekday_ids, slot_ids, delta_seconds)
        feat = torch.cat(
            [edge_static_feat, edge_dyn_feat, edge_speed_kmh, numeric_core, route_extra_feat, route_turn_feat, time_feat],
            dim=-1,
        )
        return self.net(feat)


class ETASequenceEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.pos_emb = nn.Embedding(cfg.max_route_len + 1, cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, seq: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = seq.shape
        pos_ids = torch.arange(seq_len, device=seq.device).unsqueeze(0).expand(bsz, seq_len)
        x = seq + self.pos_emb(pos_ids)
        out = self.encoder(x, src_key_padding_mask=~valid_mask.bool())
        return self.norm(out)


class ETAHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.predict_uncertainty = bool(cfg.predict_uncertainty)
        out_dim = 2 if self.predict_uncertainty else 1
        self.net = nn.Sequential(
            nn.LayerNorm(cfg.input_dim),
            nn.Linear(cfg.input_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, out_dim),
        )

    def forward(self, route_repr: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.net(route_repr)
        out = {"eta_minutes": raw[:, 0]}
        if self.predict_uncertainty:
            out["log_sigma"] = raw[:, 1]
        return out


class FinalHybridETAModel(nn.Module):
    """
    工程可跑版最终模型：
      trip_token + route edge tokens -> Transformer -> ETA
    复用已训练好的 GNN 作为动态 bank 提供器。

    bank_mode:
      - "base": 只用 BaseWeeklyBank，最容易直接跑
      - "recent": 需要 default_recent_speed_seq 或 batch["recent_speed_seq"]
      - "joint": 需要 recent + event（没有事件时不建议开）
    """
    def __init__(
        self,
        cfg: SameCityETAConfig,
        gnn_backbone: TrafficGNNSystem,
        x_static: torch.Tensor,
        profile_feat: torch.Tensor,
        edge_index: torch.Tensor,
        bank_mode: str = "base",
        default_recent_speed_seq: torch.Tensor | None = None,
        default_event_vector: torch.Tensor | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.gnn_backbone = gnn_backbone
        self.bank_mode = str(bank_mode).lower()
        self.static_base_bank_cache_enabled = self.bank_mode == "base" and bool(cfg.freeze_gnn_backbone)
        self._static_bank_ready = False
        self._cache_usage_announced = False
        if cfg.freeze_gnn_backbone:
            for p in self.gnn_backbone.parameters():
                p.requires_grad = False
            self.gnn_backbone.eval()

        self.register_buffer("x_static_buf", x_static.float())
        self.register_buffer("profile_feat_buf", profile_feat.float())
        self.register_buffer("edge_index_buf", edge_index.long())
        self.register_buffer("cached_bank_buf", torch.empty(0), persistent=False)
        self.register_buffer("cached_pred_speed_bank_buf", torch.empty(0), persistent=False)
        if default_recent_speed_seq is not None:
            self.register_buffer("default_recent_speed_seq_buf", default_recent_speed_seq.float())
        else:
            self.default_recent_speed_seq_buf = None
        if default_event_vector is not None:
            self.register_buffer("default_event_vector_buf", default_event_vector.float())
        else:
            self.default_event_vector_buf = None

        d_model = cfg.encoder.d_model
        self.trip_token_encoder = TripTokenEncoder(cfg.trip_token, cfg.route_token.bank_hidden_dim, d_model)
        self.route_token_encoder = RouteTokenEncoder(cfg.route_token, d_model)
        self.entry_time_scheduler = EntryTimeScheduler(min_speed_kmh=cfg.route_token.min_speed_kmh)
        self.sequence_encoder = ETASequenceEncoder(cfg.encoder)
        self.eta_head = ETAHead(cfg.head)

    def warmup_static_bank_cache(self) -> None:
        if not self.static_base_bank_cache_enabled or self._static_bank_ready:
            return
        with torch.no_grad():
            out = self.gnn_backbone.forward_base(self.x_static_buf, self.profile_feat_buf, self.edge_index_buf)
        self.cached_bank_buf = out["H_base_bank"].detach()
        self.cached_pred_speed_bank_buf = out["pred_speed_bank"].detach()
        self._static_bank_ready = True
        print("Warmed up static base bank cache.")

    def _run_gnn_backbone(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        recent_speed_seq = batch.get("recent_speed_seq", self.default_recent_speed_seq_buf)
        event_vector = batch.get("event_vector", self.default_event_vector_buf)

        grad_enabled = any(p.requires_grad for p in self.gnn_backbone.parameters())
        context = torch.enable_grad() if grad_enabled else torch.no_grad()

        with context:
            if self.bank_mode == "base":
                if self.static_base_bank_cache_enabled and self._static_bank_ready:
                    if not self._cache_usage_announced:
                        print("Using cached base bank.")
                        self._cache_usage_announced = True
                    return {
                        "bank": self.cached_bank_buf,
                        "pred_speed_bank": self.cached_pred_speed_bank_buf,
                        "source": "base_cached",
                    }
                out = self.gnn_backbone.forward_base(self.x_static_buf, self.profile_feat_buf, self.edge_index_buf)
                return {"bank": out["H_base_bank"], "pred_speed_bank": out["pred_speed_bank"], "source": "base"}
            if self.bank_mode == "recent":
                if recent_speed_seq is None:
                    raise KeyError("bank_mode=recent 时需要 recent_speed_seq")
                out = self.gnn_backbone.forward_recent(
                    x_static=self.x_static_buf,
                    profile_feat=self.profile_feat_buf,
                    recent_speed_seq=recent_speed_seq.to(self.x_static_buf.device),
                    edge_index=self.edge_index_buf,
                    detach_base=False,
                    return_full=True,
                )
                return {"bank": out["H_adapted_bank"], "pred_speed_bank": out["pred_speed_bank"], "source": "recent"}
            if self.bank_mode == "joint":
                if recent_speed_seq is None or event_vector is None:
                    raise KeyError("bank_mode=joint 时需要 recent_speed_seq 和 event_vector")
                event_weekday = batch.get("event_weekday", batch["current_weekday"])
                event_slot = batch.get("event_slot", batch["current_slot"])
                out = self.gnn_backbone.forward_joint_shared(
                    x_static=self.x_static_buf,
                    profile_feat=self.profile_feat_buf,
                    recent_speed_seq=recent_speed_seq.to(self.x_static_buf.device),
                    edge_index=self.edge_index_buf,
                    event_weekday=event_weekday.to(self.x_static_buf.device),
                    event_slot=event_slot.to(self.x_static_buf.device),
                    event_vector=event_vector.to(self.x_static_buf.device),
                )
                return {
                    "bank": out["H_final_bank"],
                    "pred_speed_bank": out["event_pred_speed_bank"],
                    "source": "joint",
                }
        raise ValueError(f"未知 bank_mode={self.bank_mode}")

    @staticmethod
    def _neighbors_of_edge(edge_index: torch.Tensor, edge_id: int) -> torch.Tensor:
        src, dst = edge_index.long()
        neigh = dst[src == edge_id]
        if neigh.numel() == 0:
            neigh = src[dst == edge_id]
        return neigh.unique()

    def _current_bank_context(
        self,
        bank: torch.Tensor,
        current_weekday: torch.Tensor,
        current_slot: torch.Tensor,
        current_edge_id: torch.Tensor | None = None,
        current_edge_candidate_ids: torch.Tensor | None = None,
        current_edge_candidate_probs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        current_weekday = _ensure_batch_1d_long(current_weekday).to(bank.device)
        current_slot = _ensure_batch_1d_long(current_slot).to(bank.device)
        out = []
        for b in range(current_weekday.numel()):
            h_now = bank[current_weekday[b], current_slot[b]]
            if current_edge_candidate_ids is not None and current_edge_candidate_probs is not None:
                edge_ids = current_edge_candidate_ids[b].long().to(bank.device)
                probs = current_edge_candidate_probs[b].float().to(bank.device)
                probs = probs / probs.sum().clamp_min(1e-6)
                center_state = (h_now[edge_ids] * probs.unsqueeze(-1)).sum(dim=0)
                center_edge = edge_ids[torch.argmax(probs)]
            else:
                if current_edge_id is None:
                    raise KeyError("需要 current_edge_id 或 top-k candidates")
                center_edge = _ensure_batch_1d_long(current_edge_id)[b].to(bank.device)
                center_state = h_now[center_edge]
            neighbors = self._neighbors_of_edge(self.edge_index_buf, int(center_edge.item()))
            neigh_state = center_state if neighbors.numel() == 0 else h_now[neighbors].mean(dim=0)
            out.append(0.5 * center_state + 0.5 * neigh_state)
        return torch.stack(out, dim=0)

    @staticmethod
    def _gather_bank_values(bank: torch.Tensor, edge_ids: torch.Tensor, weekday_ids: torch.Tensor, slot_ids: torch.Tensor) -> torch.Tensor:
        if edge_ids.dim() == 1:
            edge_ids = edge_ids.unsqueeze(0)
        bsz = edge_ids.size(0)
        gathered = []
        for b in range(bsz):
            gathered.append(bank[weekday_ids[b], slot_ids[b], edge_ids[b]])
        return torch.stack(gathered, dim=0)

    def _prepare_route_bank_queries(self, bank: torch.Tensor, pred_speed_bank: torch.Tensor, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        route_edge_ids = batch["route_edge_ids"].long().to(bank.device)
        route_edge_lengths_m = batch["route_edge_lengths_m"].float().to(bank.device)
        route_mask = batch["route_mask"].bool().to(bank.device)
        current_weekday = _ensure_batch_1d_long(batch["current_weekday"]).to(bank.device)
        current_slot = _ensure_batch_1d_long(batch["current_slot"]).to(bank.device)
        current_edge_remaining_ratio = batch["current_edge_remaining_ratio"].float().to(bank.device).view(-1)

        cur_week = current_weekday.view(-1, 1).expand_as(route_edge_ids)
        cur_slot = current_slot.view(-1, 1).expand_as(route_edge_ids)
        speed_pass1 = self._gather_bank_values(pred_speed_bank, route_edge_ids, cur_week, cur_slot)
        sched1 = self.entry_time_scheduler(
            current_weekday=current_weekday,
            current_slot=current_slot,
            route_edge_lengths_m=route_edge_lengths_m,
            route_speeds_kmh=speed_pass1,
            current_edge_remaining_ratio=current_edge_remaining_ratio,
            route_mask=route_mask,
        )
        speed_pass2 = self._gather_bank_values(pred_speed_bank, route_edge_ids, sched1["weekday_ids"], sched1["slot_ids"])
        sched2 = self.entry_time_scheduler(
            current_weekday=current_weekday,
            current_slot=current_slot,
            route_edge_lengths_m=route_edge_lengths_m,
            route_speeds_kmh=speed_pass2,
            current_edge_remaining_ratio=current_edge_remaining_ratio,
            route_mask=route_mask,
        )
        dyn_feat = self._gather_bank_values(bank, route_edge_ids, sched2["weekday_ids"], sched2["slot_ids"])
        dyn_speed = self._gather_bank_values(pred_speed_bank, route_edge_ids, sched2["weekday_ids"], sched2["slot_ids"])
        return {
            "route_edge_ids": route_edge_ids,
            "route_edge_lengths_m": route_edge_lengths_m,
            "route_mask": route_mask,
            "weekday_ids": sched2["weekday_ids"],
            "slot_ids": sched2["slot_ids"],
            "enter_delta_seconds": sched2["enter_delta_seconds"],
            "travel_seconds": sched2["travel_seconds"],
            "edge_dyn_feat": dyn_feat,
            "edge_speed_kmh": dyn_speed,
        }

    @staticmethod
    def _route_position_ratios(route_edge_lengths_m: torch.Tensor, route_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        masked_len = route_edge_lengths_m * route_mask.float()
        total = masked_len.sum(dim=1, keepdim=True).clamp_min(1e-6)
        cum_after = torch.cumsum(masked_len, dim=1)
        cum_before = cum_after - masked_len
        remaining = (total - cum_before) / total
        cumulative = cum_before / total
        return remaining.clamp(0.0, 1.0), cumulative.clamp(0.0, 1.0)

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        gnn_out = self._run_gnn_backbone(batch)
        bank = gnn_out["bank"]
        pred_speed_bank = gnn_out["pred_speed_bank"]

        current_bank_context = self._current_bank_context(
            bank=bank,
            current_weekday=batch["current_weekday"],
            current_slot=batch["current_slot"],
            current_edge_id=batch.get("current_edge_id"),
            current_edge_candidate_ids=batch.get("current_edge_candidate_ids"),
            current_edge_candidate_probs=batch.get("current_edge_candidate_probs"),
        )

        trip_token = self.trip_token_encoder(
            trip_num_feat=batch["trip_num_feat"].to(bank.device),
            trip_cat_feat=batch["trip_cat_feat"].to(bank.device),
            bank_context=current_bank_context,
        )

        route_bank = self._prepare_route_bank_queries(bank, pred_speed_bank, batch)
        route_edge_ids = route_bank["route_edge_ids"]
        edge_static_feat = self.x_static_buf[route_edge_ids]
        route_remaining_ratio, route_cumulative_ratio = self._route_position_ratios(
            route_bank["route_edge_lengths_m"],
            route_bank["route_mask"],
        )
        route_tokens = self.route_token_encoder(
            edge_static_feat=edge_static_feat,
            edge_dyn_feat=route_bank["edge_dyn_feat"],
            edge_speed_kmh=route_bank["edge_speed_kmh"],
            edge_lengths_m=route_bank["route_edge_lengths_m"],
            weekday_ids=route_bank["weekday_ids"],
            slot_ids=route_bank["slot_ids"],
            delta_seconds=route_bank["enter_delta_seconds"],
            route_remaining_ratio=route_remaining_ratio,
            route_cumulative_ratio=route_cumulative_ratio,
            route_turn_feat=batch.get("route_turn_feat", None),
            route_extra_feat=batch.get("route_extra_feat", None),
        )

        seq = torch.cat([trip_token.unsqueeze(1), route_tokens], dim=1)
        valid_mask = torch.cat(
            [
                torch.ones((seq.size(0), 1), dtype=torch.bool, device=seq.device),
                batch["route_mask"].to(seq.device).bool(),
            ],
            dim=1,
        )
        encoded = self.sequence_encoder(seq, valid_mask)
        route_repr = encoded[:, 0]
        out = self.eta_head(route_repr)
        out["bank_source"] = torch.tensor(0, device=seq.device)
        return out


__all__ = [
    "FinalHybridETAModel",
]
