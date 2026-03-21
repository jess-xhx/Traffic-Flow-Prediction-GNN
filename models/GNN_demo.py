from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from GNN_1_base import BaseWeeklyBank
from GNN_2_recent import RecentResidualBank
from GNN_3_event import EventResidualInjector


Tensor = torch.Tensor
Batch = Mapping[str, Any]


# =========================
# 配置
# =========================


@dataclass
class ModelConfig:
    static_dim: int
    profile_dim: int
    event_dim: int
    static_hidden_dim: int = 32
    calendar_hidden_dim: int = 32
    profile_hidden_dim: int = 32
    bank_hidden_dim: int = 64
    recent_hidden_dim: int = 32


@dataclass
class StageTrainConfig:
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: Optional[float] = 5.0
    log_every: int = 1


@dataclass
class JointTrainConfig:
    epochs: int = 5
    lr_base: float = 1e-4
    lr_recent: float = 3e-4
    lr_event: float = 5e-4
    weight_decay: float = 1e-5
    alpha_base: float = 0.2
    beta_recent: float = 0.3
    gamma_event: float = 1.0
    lambda_recent_reg: float = 1e-4
    lambda_event_reg: float = 1e-4
    grad_clip: Optional[float] = 5.0
    log_every: int = 1


# =========================
# 基础工具
# =========================


def _to_device(obj: Any, device: torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device(v, device) for v in obj)
    return obj


def _maybe_squeeze_graph_batch(x: Any) -> Any:
    """
    DataLoader 常见情况: batch_size=1 时会在最前面多一维。
    这里仅在第一维为 1 时去掉它，避免改坏真实张量形状。
    """
    if not torch.is_tensor(x):
        return x
    if x.dim() > 0 and x.size(0) == 1:
        return x.squeeze(0)
    return x


def _prepare_batch(batch: Batch, device: torch.device) -> Dict[str, Any]:
    batch = _to_device(dict(batch), device)
    for k, v in list(batch.items()):
        if torch.is_tensor(v):
            batch[k] = _maybe_squeeze_graph_batch(v)
    return batch


def _scalar_int(x: Union[int, Tensor]) -> int:
    if isinstance(x, int):
        return x
    x = _maybe_squeeze_graph_batch(x)
    if x.numel() != 1:
        raise ValueError(f"期望标量，但拿到 shape={tuple(x.shape)}")
    return int(x.item())


def _ensure_bank_layout(y: Tensor, num_nodes: int) -> Tensor:
    """
    把 bank 监督目标统一成 [7, 288, N]。
    支持:
      - [7, 288, N]
      - [N, 7, 288]
      - [1, 7, 288, N]
      - [1, N, 7, 288]
    """
    y = _maybe_squeeze_graph_batch(y)
    if y.dim() != 3:
        raise ValueError(f"bank 监督张量应为 3 维，实际为 {tuple(y.shape)}")

    if y.shape[0] == 7 and y.shape[1] == 288:
        return y
    if y.shape[1] == 7 and y.shape[2] == 288 and y.shape[0] == num_nodes:
        return y.permute(1, 2, 0).contiguous()

    raise ValueError(
        f"无法识别 bank 目标形状 {tuple(y.shape)}。支持 [7,288,N] 或 [N,7,288]。"
    )


def _ensure_node_vector_layout(y: Tensor, num_nodes: int) -> Tensor:
    """
    把单时刻速度监督统一成 [N]。
    支持 [N], [1, N], [N, 1], [1, N, 1]。
    """
    y = _maybe_squeeze_graph_batch(y)
    if y.dim() == 1 and y.shape[0] == num_nodes:
        return y
    if y.dim() == 2:
        if y.shape == (num_nodes, 1):
            return y.squeeze(-1)
        if y.shape[0] == 1 and y.shape[1] == num_nodes:
            return y.squeeze(0)
    raise ValueError(f"无法识别单时刻监督形状 {tuple(y.shape)}，期望 [N] / [N,1] / [1,N]。")


def masked_mae(pred: Tensor, target: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    if mask is None:
        return torch.mean(torch.abs(pred - target))
    mask = mask.to(pred.dtype)
    loss = torch.abs(pred - target) * mask
    denom = mask.sum().clamp_min(1.0)
    return loss.sum() / denom


def freeze_module(module: nn.Module) -> None:
    module.eval()
    for p in module.parameters():
        p.requires_grad = False


def unfreeze_module(module: nn.Module) -> None:
    module.train()
    for p in module.parameters():
        p.requires_grad = True


# =========================
# 总模型
# =========================


class TrafficGNNSystem(nn.Module):
    """
    总模型封装：
      模块1: BaseWeeklyBank
      模块2: RecentResidualBank
      模块3: EventResidualInjector

    约定 batch 字段：
      x_static         [N, F_s]
      profile_feat     [N, 7, 288, F_p]
      edge_index       [2, E]
      recent_speed_seq [N, K, 1]                 (阶段2/3/联合需要)
      event_vector     [N, F_e]                  (阶段3/联合需要)
      target_weekday   标量 int / Tensor         (阶段3/联合需要)
      target_slot      标量 int / Tensor         (阶段3/联合需要)
      y_base_bank      [7, 288, N] or [N, 7, 288]
      y_future_bank    [7, 288, N] or [N, 7, 288]
      y_target_speed   [N] / [N,1] / [1,N]

    说明：
      - 模块3只对“单个目标时刻”做即时事件修正。
      - 多 horizon 预测时，可循环多次调用 predict_one_horizon / predict_multi_horizon。
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

    # ---------- 核心前向 ----------

    def forward_base(self, x_static: Tensor, profile_feat: Tensor, edge_index: Tensor) -> Dict[str, Tensor]:
        H_base_bank, pred_speed_bank = self.base.build_bank(x_static, profile_feat, edge_index)
        return {
            "H_base_bank": H_base_bank,
            "pred_speed_bank": pred_speed_bank,
        }

    def forward_recent(
        self,
        x_static: Tensor,
        profile_feat: Tensor,
        recent_speed_seq: Tensor,
        edge_index: Tensor,
        detach_base: bool = False,
    ) -> Dict[str, Tensor]:
        base_out = self.forward_base(x_static, profile_feat, edge_index)
        H_base_bank = base_out["H_base_bank"]
        if detach_base:
            H_base_bank = H_base_bank.detach()

        delta_recent_bank, H_adapted_bank, pred_speed_bank = self.recent.build_delta_bank(
            H_base_bank, recent_speed_seq, edge_index
        )
        return {
            "H_base_bank": H_base_bank,
            "delta_recent_bank": delta_recent_bank,
            "H_adapted_bank": H_adapted_bank,
            "pred_speed_bank": pred_speed_bank,
        }

    def forward_event(
        self,
        x_static: Tensor,
        profile_feat: Tensor,
        recent_speed_seq: Tensor,
        edge_index: Tensor,
        target_weekday: Union[int, Tensor],
        target_slot: Union[int, Tensor],
        event_vector: Tensor,
        detach_pre_event: bool = False,
    ) -> Dict[str, Tensor]:
        recent_out = self.forward_recent(
            x_static=x_static,
            profile_feat=profile_feat,
            recent_speed_seq=recent_speed_seq,
            edge_index=edge_index,
            detach_base=detach_pre_event,
        )

        H_adapted_bank = recent_out["H_adapted_bank"]
        if detach_pre_event:
            H_adapted_bank = H_adapted_bank.detach()

        weekday = _scalar_int(target_weekday)
        slot = _scalar_int(target_slot)
        H_adapted_t = H_adapted_bank[weekday, slot]  # [N, D]

        delta_event_t, H_final_t, pred_speed_t = self.event.inject(H_adapted_t, event_vector, edge_index)
        return {
            "H_base_bank": recent_out["H_base_bank"],
            "delta_recent_bank": recent_out["delta_recent_bank"],
            "H_adapted_bank": H_adapted_bank,
            "H_adapted_t": H_adapted_t,
            "delta_event_t": delta_event_t,
            "H_final_t": H_final_t,
            "pred_speed_t": pred_speed_t,
        }

    def forward(self, batch: Batch, mode: str = "joint", detach_base: bool = False, detach_pre_event: bool = False) -> Dict[str, Tensor]:
        b = _prepare_batch(batch, next(self.parameters()).device)

        if mode == "base":
            return self.forward_base(
                x_static=b["x_static"],
                profile_feat=b["profile_feat"],
                edge_index=b["edge_index"],
            )

        if mode == "recent":
            return self.forward_recent(
                x_static=b["x_static"],
                profile_feat=b["profile_feat"],
                recent_speed_seq=b["recent_speed_seq"],
                edge_index=b["edge_index"],
                detach_base=detach_base,
            )

        if mode in {"event", "joint"}:
            return self.forward_event(
                x_static=b["x_static"],
                profile_feat=b["profile_feat"],
                recent_speed_seq=b["recent_speed_seq"],
                edge_index=b["edge_index"],
                target_weekday=b["target_weekday"],
                target_slot=b["target_slot"],
                event_vector=b["event_vector"],
                detach_pre_event=detach_pre_event,
            )

        raise ValueError(f"未知 mode={mode}")

    # ---------- 推理接口 ----------

    @torch.no_grad()
    def build_base_bank(self, x_static: Tensor, profile_feat: Tensor, edge_index: Tensor) -> Tensor:
        self.eval()
        out = self.forward_base(x_static, profile_feat, edge_index)
        return out["H_base_bank"]

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
        return out["H_adapted_bank"]

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
        H_adapted_t = recent_out["H_adapted_bank"][target_weekday, target_slot]

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
        """
        返回 [H, N]。
        event_vectors 允许为 None；若给出，则长度应与 target_pairs 相同。
        """
        self.eval()
        recent_out = self.forward_recent(x_static, profile_feat, recent_speed_seq, edge_index)
        H_adapted_bank = recent_out["H_adapted_bank"]

        preds: List[Tensor] = []
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


# =========================
# 损失函数
# =========================


class TrafficGNNLoss(nn.Module):
    def __init__(self, lambda_recent_reg: float = 1e-4, lambda_event_reg: float = 1e-4):
        super().__init__()
        self.lambda_recent_reg = lambda_recent_reg
        self.lambda_event_reg = lambda_event_reg

    def base_loss(self, pred_speed_bank: Tensor, y_base_bank: Tensor, mask: Optional[Tensor] = None) -> Dict[str, Tensor]:
        loss = masked_mae(pred_speed_bank, y_base_bank, mask)
        return {
            "loss": loss,
            "mae": loss.detach(),
        }

    def recent_loss(
        self,
        pred_speed_bank: Tensor,
        y_future_bank: Tensor,
        delta_recent_bank: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        data_loss = masked_mae(pred_speed_bank, y_future_bank, mask)
        reg_loss = delta_recent_bank.pow(2).mean()
        total = data_loss + self.lambda_recent_reg * reg_loss
        return {
            "loss": total,
            "mae": data_loss.detach(),
            "reg": reg_loss.detach(),
        }

    def event_loss(
        self,
        pred_speed_t: Tensor,
        y_target_speed: Tensor,
        delta_event_t: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        data_loss = masked_mae(pred_speed_t, y_target_speed, mask)
        reg_loss = delta_event_t.pow(2).mean()
        total = data_loss + self.lambda_event_reg * reg_loss
        return {
            "loss": total,
            "mae": data_loss.detach(),
            "reg": reg_loss.detach(),
        }


# =========================
# 训练器
# =========================


class TrafficGNNTrainer:
    def __init__(
        self,
        model: TrafficGNNSystem,
        device: Union[str, torch.device] = "cpu",
        criterion: Optional[TrafficGNNLoss] = None,
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.criterion = criterion or TrafficGNNLoss()

    # ---------- 单 epoch ----------

    def _run_base_epoch(
        self,
        dataloader: DataLoader,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Dict[str, float]:
        training = optimizer is not None
        self.model.train(training)

        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0

        for raw_batch in dataloader:
            batch = _prepare_batch(raw_batch, self.device)
            out = self.model.forward(batch, mode="base")

            num_nodes = batch["x_static"].shape[0]
            y_base_bank = _ensure_bank_layout(batch["y_base_bank"], num_nodes)
            base_mask = batch.get("base_mask")
            if base_mask is not None:
                base_mask = _ensure_bank_layout(base_mask, num_nodes)

            loss_dict = self.criterion.base_loss(out["pred_speed_bank"], y_base_bank, base_mask)
            loss = loss_dict["loss"]

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += float(loss.detach().item())
            total_mae += float(loss_dict["mae"].item())
            num_batches += 1

        denom = max(num_batches, 1)
        return {
            "loss": total_loss / denom,
            "mae": total_mae / denom,
        }

    def _run_recent_epoch(
        self,
        dataloader: DataLoader,
        optimizer: Optional[torch.optim.Optimizer] = None,
        detach_base: bool = True,
        grad_clip: Optional[float] = None,
    ) -> Dict[str, float]:
        training = optimizer is not None
        self.model.train(training)

        total_loss = 0.0
        total_mae = 0.0
        total_reg = 0.0
        num_batches = 0

        for raw_batch in dataloader:
            batch = _prepare_batch(raw_batch, self.device)
            out = self.model.forward(batch, mode="recent", detach_base=detach_base)

            num_nodes = batch["x_static"].shape[0]
            y_future_bank = _ensure_bank_layout(batch["y_future_bank"], num_nodes)
            future_mask = batch.get("future_mask")
            if future_mask is not None:
                future_mask = _ensure_bank_layout(future_mask, num_nodes)

            loss_dict = self.criterion.recent_loss(
                out["pred_speed_bank"], y_future_bank, out["delta_recent_bank"], future_mask
            )
            loss = loss_dict["loss"]

            if training:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.recent.parameters(), grad_clip)
                optimizer.step()

            total_loss += float(loss.detach().item())
            total_mae += float(loss_dict["mae"].item())
            total_reg += float(loss_dict["reg"].item())
            num_batches += 1

        denom = max(num_batches, 1)
        return {
            "loss": total_loss / denom,
            "mae": total_mae / denom,
            "reg": total_reg / denom,
        }

    def _run_event_epoch(
        self,
        dataloader: DataLoader,
        optimizer: Optional[torch.optim.Optimizer] = None,
        detach_pre_event: bool = True,
        grad_clip: Optional[float] = None,
    ) -> Dict[str, float]:
        training = optimizer is not None
        self.model.train(training)

        total_loss = 0.0
        total_mae = 0.0
        total_reg = 0.0
        num_batches = 0

        for raw_batch in dataloader:
            batch = _prepare_batch(raw_batch, self.device)
            out = self.model.forward(batch, mode="event", detach_pre_event=detach_pre_event)

            num_nodes = batch["x_static"].shape[0]
            y_target_speed = _ensure_node_vector_layout(batch["y_target_speed"], num_nodes)
            event_mask = batch.get("event_mask")
            if event_mask is not None:
                event_mask = _ensure_node_vector_layout(event_mask, num_nodes)

            loss_dict = self.criterion.event_loss(
                out["pred_speed_t"], y_target_speed, out["delta_event_t"], event_mask
            )
            loss = loss_dict["loss"]

            if training:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.event.parameters(), grad_clip)
                optimizer.step()

            total_loss += float(loss.detach().item())
            total_mae += float(loss_dict["mae"].item())
            total_reg += float(loss_dict["reg"].item())
            num_batches += 1

        denom = max(num_batches, 1)
        return {
            "loss": total_loss / denom,
            "mae": total_mae / denom,
            "reg": total_reg / denom,
        }

    def _run_joint_epoch(
        self,
        dataloader: DataLoader,
        optimizer: Optional[torch.optim.Optimizer] = None,
        joint_cfg: Optional[JointTrainConfig] = None,
    ) -> Dict[str, float]:
        if joint_cfg is None:
            raise ValueError("joint_cfg 不能为空")

        training = optimizer is not None
        self.model.train(training)

        total_loss = 0.0
        total_base = 0.0
        total_recent = 0.0
        total_event = 0.0
        num_batches = 0

        for raw_batch in dataloader:
            batch = _prepare_batch(raw_batch, self.device)

            # 1) base loss
            base_out = self.model.forward(batch, mode="base")
            num_nodes = batch["x_static"].shape[0]
            y_base_bank = _ensure_bank_layout(batch["y_base_bank"], num_nodes)
            base_mask = batch.get("base_mask")
            if base_mask is not None:
                base_mask = _ensure_bank_layout(base_mask, num_nodes)
            base_loss_dict = self.criterion.base_loss(base_out["pred_speed_bank"], y_base_bank, base_mask)

            # 2) recent loss
            recent_out = self.model.forward(batch, mode="recent", detach_base=False)
            y_future_bank = _ensure_bank_layout(batch["y_future_bank"], num_nodes)
            future_mask = batch.get("future_mask")
            if future_mask is not None:
                future_mask = _ensure_bank_layout(future_mask, num_nodes)
            recent_loss_dict = self.criterion.recent_loss(
                recent_out["pred_speed_bank"], y_future_bank, recent_out["delta_recent_bank"], future_mask
            )

            # 3) event loss
            event_out = self.model.forward(batch, mode="event", detach_pre_event=False)
            y_target_speed = _ensure_node_vector_layout(batch["y_target_speed"], num_nodes)
            event_mask = batch.get("event_mask")
            if event_mask is not None:
                event_mask = _ensure_node_vector_layout(event_mask, num_nodes)
            event_loss_dict = self.criterion.event_loss(
                event_out["pred_speed_t"], y_target_speed, event_out["delta_event_t"], event_mask
            )

            loss = (
                joint_cfg.alpha_base * base_loss_dict["loss"]
                + joint_cfg.beta_recent * recent_loss_dict["loss"]
                + joint_cfg.gamma_event * event_loss_dict["loss"]
            )

            if training:
                optimizer.zero_grad()
                loss.backward()
                if joint_cfg.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), joint_cfg.grad_clip)
                optimizer.step()

            total_loss += float(loss.detach().item())
            total_base += float(base_loss_dict["loss"].detach().item())
            total_recent += float(recent_loss_dict["loss"].detach().item())
            total_event += float(event_loss_dict["loss"].detach().item())
            num_batches += 1

        denom = max(num_batches, 1)
        return {
            "loss": total_loss / denom,
            "base_loss": total_base / denom,
            "recent_loss": total_recent / denom,
            "event_loss": total_event / denom,
        }

    # ---------- 对外训练接口 ----------

    def train_stage1(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        cfg: Optional[StageTrainConfig] = None,
    ) -> List[Dict[str, float]]:
        cfg = cfg or StageTrainConfig()

        unfreeze_module(self.model.base)
        freeze_module(self.model.recent)
        freeze_module(self.model.event)

        optimizer = torch.optim.Adam(self.model.base.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        history: List[Dict[str, float]] = []

        for epoch in range(1, cfg.epochs + 1):
            train_metrics = self._run_base_epoch(train_loader, optimizer)
            log = {"epoch": epoch, "stage": 1, **{f"train_{k}": v for k, v in train_metrics.items()}}
            if val_loader is not None:
                with torch.no_grad():
                    val_metrics = self._run_base_epoch(val_loader, optimizer=None)
                log.update({f"val_{k}": v for k, v in val_metrics.items()})
            history.append(log)
            if epoch % cfg.log_every == 0:
                print(log)
        return history

    def train_stage2(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        cfg: Optional[StageTrainConfig] = None,
    ) -> List[Dict[str, float]]:
        cfg = cfg or StageTrainConfig()

        freeze_module(self.model.base)
        unfreeze_module(self.model.recent)
        freeze_module(self.model.event)

        optimizer = torch.optim.Adam(self.model.recent.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        history: List[Dict[str, float]] = []

        for epoch in range(1, cfg.epochs + 1):
            train_metrics = self._run_recent_epoch(
                train_loader,
                optimizer=optimizer,
                detach_base=True,
                grad_clip=cfg.grad_clip,
            )
            log = {"epoch": epoch, "stage": 2, **{f"train_{k}": v for k, v in train_metrics.items()}}
            if val_loader is not None:
                with torch.no_grad():
                    val_metrics = self._run_recent_epoch(
                        val_loader,
                        optimizer=None,
                        detach_base=True,
                        grad_clip=None,
                    )
                log.update({f"val_{k}": v for k, v in val_metrics.items()})
            history.append(log)
            if epoch % cfg.log_every == 0:
                print(log)
        return history

    def train_stage3(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        cfg: Optional[StageTrainConfig] = None,
    ) -> List[Dict[str, float]]:
        cfg = cfg or StageTrainConfig()

        freeze_module(self.model.base)
        freeze_module(self.model.recent)
        unfreeze_module(self.model.event)

        optimizer = torch.optim.Adam(self.model.event.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        history: List[Dict[str, float]] = []

        for epoch in range(1, cfg.epochs + 1):
            train_metrics = self._run_event_epoch(
                train_loader,
                optimizer=optimizer,
                detach_pre_event=True,
                grad_clip=cfg.grad_clip,
            )
            log = {"epoch": epoch, "stage": 3, **{f"train_{k}": v for k, v in train_metrics.items()}}
            if val_loader is not None:
                with torch.no_grad():
                    val_metrics = self._run_event_epoch(
                        val_loader,
                        optimizer=None,
                        detach_pre_event=True,
                        grad_clip=None,
                    )
                log.update({f"val_{k}": v for k, v in val_metrics.items()})
            history.append(log)
            if epoch % cfg.log_every == 0:
                print(log)
        return history

    def joint_finetune(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        cfg: Optional[JointTrainConfig] = None,
    ) -> List[Dict[str, float]]:
        cfg = cfg or JointTrainConfig()

        unfreeze_module(self.model.base)
        unfreeze_module(self.model.recent)
        unfreeze_module(self.model.event)

        optimizer = torch.optim.Adam(
            [
                {"params": self.model.base.parameters(), "lr": cfg.lr_base},
                {"params": self.model.recent.parameters(), "lr": cfg.lr_recent},
                {"params": self.model.event.parameters(), "lr": cfg.lr_event},
            ],
            weight_decay=cfg.weight_decay,
        )

        history: List[Dict[str, float]] = []
        for epoch in range(1, cfg.epochs + 1):
            train_metrics = self._run_joint_epoch(train_loader, optimizer=optimizer, joint_cfg=cfg)
            log = {"epoch": epoch, "stage": "joint", **{f"train_{k}": v for k, v in train_metrics.items()}}
            if val_loader is not None:
                with torch.no_grad():
                    val_metrics = self._run_joint_epoch(val_loader, optimizer=None, joint_cfg=cfg)
                log.update({f"val_{k}": v for k, v in val_metrics.items()})
            history.append(log)
            if epoch % cfg.log_every == 0:
                print(log)
        return history

    def fit(
        self,
        stage1_train_loader: DataLoader,
        stage2_train_loader: DataLoader,
        stage3_train_loader: DataLoader,
        joint_train_loader: DataLoader,
        stage1_val_loader: Optional[DataLoader] = None,
        stage2_val_loader: Optional[DataLoader] = None,
        stage3_val_loader: Optional[DataLoader] = None,
        joint_val_loader: Optional[DataLoader] = None,
        stage1_cfg: Optional[StageTrainConfig] = None,
        stage2_cfg: Optional[StageTrainConfig] = None,
        stage3_cfg: Optional[StageTrainConfig] = None,
        joint_cfg: Optional[JointTrainConfig] = None,
    ) -> Dict[str, List[Dict[str, float]]]:
        return {
            "stage1": self.train_stage1(stage1_train_loader, stage1_val_loader, stage1_cfg),
            "stage2": self.train_stage2(stage2_train_loader, stage2_val_loader, stage2_cfg),
            "stage3": self.train_stage3(stage3_train_loader, stage3_val_loader, stage3_cfg),
            "joint": self.joint_finetune(joint_train_loader, joint_val_loader, joint_cfg),
        }

    # ---------- 保存 / 加载 ----------

    def save_checkpoint(self, path: Union[str, Path], extra: Optional[Dict[str, Any]] = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state_dict": self.model.state_dict(),
            "model_config": self.model.cfg.__dict__,
            "criterion": {
                "lambda_recent_reg": self.criterion.lambda_recent_reg,
                "lambda_event_reg": self.criterion.lambda_event_reg,
            },
        }
        if extra is not None:
            payload["extra"] = extra
        torch.save(payload, path)

    @staticmethod
    def load_checkpoint(path: Union[str, Path], device: Union[str, torch.device] = "cpu") -> "TrafficGNNTrainer":
        ckpt = torch.load(path, map_location=device)
        model_cfg = ModelConfig(**ckpt["model_config"])
        model = TrafficGNNSystem(model_cfg)
        model.load_state_dict(ckpt["model_state_dict"])
        criterion_cfg = ckpt.get("criterion", {})
        criterion = TrafficGNNLoss(
            lambda_recent_reg=criterion_cfg.get("lambda_recent_reg", 1e-4),
            lambda_event_reg=criterion_cfg.get("lambda_event_reg", 1e-4),
        )
        return TrafficGNNTrainer(model=model, device=device, criterion=criterion)


# =========================
# 使用示例（伪代码）
# =========================


if __name__ == "__main__":
    """
    这里只给最小使用示例，真实训练时请自己准备 DataLoader。

    每个 batch 推荐是一个 dict，至少包含：
      stage1:
        x_static, profile_feat, edge_index, y_base_bank
      stage2:
        x_static, profile_feat, edge_index, recent_speed_seq, y_future_bank
      stage3:
        x_static, profile_feat, edge_index, recent_speed_seq,
        target_weekday, target_slot, event_vector, y_target_speed
      joint:
        以上字段全都有
    """
    cfg = ModelConfig(
        static_dim=16,
        profile_dim=8,
        event_dim=6,
        bank_hidden_dim=64,
    )
    model = TrafficGNNSystem(cfg)
    trainer = TrafficGNNTrainer(model, device="cpu")

    print("TrafficGNNSystem 已创建。请传入 DataLoader 调用 train_stage1/train_stage2/train_stage3/joint_finetune。")
