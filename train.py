from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader

from configs.gnn_config import JointTrainConfig, ModelConfig, StageTrainConfig
from GNN import TrafficGNNSystem
from utils.gnn_loss import TrafficGNNLoss
from utils.gnn_utils import (
    ensure_bank_layout,
    ensure_node_vector_layout,
    freeze_module,
    prepare_batch,
    unfreeze_module,
)


class TrafficGNNTrainer:
    def __init__(
        self,
        model: TrafficGNNSystem,
        device: str | torch.device = 'cpu',
        criterion: Optional[TrafficGNNLoss] = None,
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.criterion = criterion or TrafficGNNLoss()

    def _run_base_epoch(
        self,
        dataloader: DataLoader,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> dict[str, float]:
        training = optimizer is not None
        self.model.train(training)

        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0

        for raw_batch in dataloader:
            batch = prepare_batch(raw_batch, self.device)
            out = self.model.forward(batch, mode='base')

            num_nodes = batch['x_static'].shape[0]
            y_base_bank = ensure_bank_layout(batch['y_base_bank'], num_nodes)
            base_mask = batch.get('base_mask')
            if base_mask is not None:
                base_mask = ensure_bank_layout(base_mask, num_nodes)

            loss_dict = self.criterion.base_loss(out['pred_speed_bank'], y_base_bank, base_mask)
            loss = loss_dict['loss']

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += float(loss.detach().item())
            total_mae += float(loss_dict['mae'].item())
            num_batches += 1

        denom = max(num_batches, 1)
        return {
            'loss': total_loss / denom,
            'mae': total_mae / denom,
        }

    def _run_recent_epoch(
        self,
        dataloader: DataLoader,
        optimizer: Optional[torch.optim.Optimizer] = None,
        detach_base: bool = True,
        grad_clip: float | None = None,
    ) -> dict[str, float]:
        training = optimizer is not None
        self.model.train(training)

        total_loss = 0.0
        total_mae = 0.0
        total_reg = 0.0
        num_batches = 0

        for raw_batch in dataloader:
            batch = prepare_batch(raw_batch, self.device)
            out = self.model.forward(batch, mode='recent', detach_base=detach_base)

            num_nodes = batch['x_static'].shape[0]
            y_future_bank = ensure_bank_layout(batch['y_future_bank'], num_nodes)
            future_mask = batch.get('future_mask')
            if future_mask is not None:
                future_mask = ensure_bank_layout(future_mask, num_nodes)

            loss_dict = self.criterion.recent_loss(
                out['pred_speed_bank'], y_future_bank, out['delta_recent_bank'], future_mask
            )
            loss = loss_dict['loss']

            if training:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.recent.parameters(), grad_clip)
                optimizer.step()

            total_loss += float(loss.detach().item())
            total_mae += float(loss_dict['mae'].item())
            total_reg += float(loss_dict['reg'].item())
            num_batches += 1

        denom = max(num_batches, 1)
        return {
            'loss': total_loss / denom,
            'mae': total_mae / denom,
            'reg': total_reg / denom,
        }

    def _run_event_epoch(
        self,
        dataloader: DataLoader,
        optimizer: Optional[torch.optim.Optimizer] = None,
        detach_pre_event: bool = True,
        grad_clip: float | None = None,
    ) -> dict[str, float]:
        training = optimizer is not None
        self.model.train(training)

        total_loss = 0.0
        total_mae = 0.0
        total_reg = 0.0
        num_batches = 0

        for raw_batch in dataloader:
            batch = prepare_batch(raw_batch, self.device)
            out = self.model.forward(batch, mode='event', detach_pre_event=detach_pre_event)

            num_nodes = batch['x_static'].shape[0]
            y_target_speed = ensure_node_vector_layout(batch['y_target_speed'], num_nodes)
            event_mask = batch.get('event_mask')
            if event_mask is not None:
                event_mask = ensure_node_vector_layout(event_mask, num_nodes)

            loss_dict = self.criterion.event_loss(
                out['pred_speed_t'], y_target_speed, out['delta_event_t'], event_mask
            )
            loss = loss_dict['loss']

            if training:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.event.parameters(), grad_clip)
                optimizer.step()

            total_loss += float(loss.detach().item())
            total_mae += float(loss_dict['mae'].item())
            total_reg += float(loss_dict['reg'].item())
            num_batches += 1

        denom = max(num_batches, 1)
        return {
            'loss': total_loss / denom,
            'mae': total_mae / denom,
            'reg': total_reg / denom,
        }

    def _run_joint_epoch(
        self,
        dataloader: DataLoader,
        optimizer: Optional[torch.optim.Optimizer] = None,
        joint_cfg: Optional[JointTrainConfig] = None,
    ) -> dict[str, float]:
        if joint_cfg is None:
            raise ValueError('joint_cfg 不能为空')

        training = optimizer is not None
        self.model.train(training)

        total_loss = 0.0
        total_base = 0.0
        total_recent = 0.0
        total_event = 0.0
        num_batches = 0

        for raw_batch in dataloader:
            batch = prepare_batch(raw_batch, self.device)
            num_nodes = batch['x_static'].shape[0]

            base_out = self.model.forward(batch, mode='base')
            y_base_bank = ensure_bank_layout(batch['y_base_bank'], num_nodes)
            base_mask = batch.get('base_mask')
            if base_mask is not None:
                base_mask = ensure_bank_layout(base_mask, num_nodes)
            base_loss_dict = self.criterion.base_loss(base_out['pred_speed_bank'], y_base_bank, base_mask)

            recent_out = self.model.forward(batch, mode='recent', detach_base=False)
            y_future_bank = ensure_bank_layout(batch['y_future_bank'], num_nodes)
            future_mask = batch.get('future_mask')
            if future_mask is not None:
                future_mask = ensure_bank_layout(future_mask, num_nodes)
            recent_loss_dict = self.criterion.recent_loss(
                recent_out['pred_speed_bank'], y_future_bank, recent_out['delta_recent_bank'], future_mask
            )

            event_out = self.model.forward(batch, mode='event', detach_pre_event=False)
            y_target_speed = ensure_node_vector_layout(batch['y_target_speed'], num_nodes)
            event_mask = batch.get('event_mask')
            if event_mask is not None:
                event_mask = ensure_node_vector_layout(event_mask, num_nodes)
            event_loss_dict = self.criterion.event_loss(
                event_out['pred_speed_t'], y_target_speed, event_out['delta_event_t'], event_mask
            )

            loss = (
                joint_cfg.alpha_base * base_loss_dict['loss']
                + joint_cfg.beta_recent * recent_loss_dict['loss']
                + joint_cfg.gamma_event * event_loss_dict['loss']
            )

            if training:
                optimizer.zero_grad()
                loss.backward()
                if joint_cfg.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), joint_cfg.grad_clip)
                optimizer.step()

            total_loss += float(loss.detach().item())
            total_base += float(base_loss_dict['loss'].detach().item())
            total_recent += float(recent_loss_dict['loss'].detach().item())
            total_event += float(event_loss_dict['loss'].detach().item())
            num_batches += 1

        denom = max(num_batches, 1)
        return {
            'loss': total_loss / denom,
            'base_loss': total_base / denom,
            'recent_loss': total_recent / denom,
            'event_loss': total_event / denom,
        }

    def train_stage1(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        cfg: Optional[StageTrainConfig] = None,
    ) -> list[dict[str, float]]:
        cfg = cfg or StageTrainConfig()

        unfreeze_module(self.model.base)
        freeze_module(self.model.recent)
        freeze_module(self.model.event)

        optimizer = torch.optim.Adam(self.model.base.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        history: list[dict[str, float]] = []

        for epoch in range(1, cfg.epochs + 1):
            train_metrics = self._run_base_epoch(train_loader, optimizer)
            log = {'epoch': epoch, 'stage': 1, **{f'train_{k}': v for k, v in train_metrics.items()}}
            if val_loader is not None:
                with torch.no_grad():
                    val_metrics = self._run_base_epoch(val_loader, optimizer=None)
                log.update({f'val_{k}': v for k, v in val_metrics.items()})
            history.append(log)
            if epoch % cfg.log_every == 0:
                print(log)
        return history

    def train_stage2(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        cfg: Optional[StageTrainConfig] = None,
    ) -> list[dict[str, float]]:
        cfg = cfg or StageTrainConfig()

        freeze_module(self.model.base)
        unfreeze_module(self.model.recent)
        freeze_module(self.model.event)

        optimizer = torch.optim.Adam(self.model.recent.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        history: list[dict[str, float]] = []

        for epoch in range(1, cfg.epochs + 1):
            train_metrics = self._run_recent_epoch(
                train_loader,
                optimizer=optimizer,
                detach_base=True,
                grad_clip=cfg.grad_clip,
            )
            log = {'epoch': epoch, 'stage': 2, **{f'train_{k}': v for k, v in train_metrics.items()}}
            if val_loader is not None:
                with torch.no_grad():
                    val_metrics = self._run_recent_epoch(
                        val_loader,
                        optimizer=None,
                        detach_base=True,
                        grad_clip=None,
                    )
                log.update({f'val_{k}': v for k, v in val_metrics.items()})
            history.append(log)
            if epoch % cfg.log_every == 0:
                print(log)
        return history

    def train_stage3(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        cfg: Optional[StageTrainConfig] = None,
    ) -> list[dict[str, float]]:
        cfg = cfg or StageTrainConfig()

        freeze_module(self.model.base)
        freeze_module(self.model.recent)
        unfreeze_module(self.model.event)

        optimizer = torch.optim.Adam(self.model.event.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        history: list[dict[str, float]] = []

        for epoch in range(1, cfg.epochs + 1):
            train_metrics = self._run_event_epoch(
                train_loader,
                optimizer=optimizer,
                detach_pre_event=True,
                grad_clip=cfg.grad_clip,
            )
            log = {'epoch': epoch, 'stage': 3, **{f'train_{k}': v for k, v in train_metrics.items()}}
            if val_loader is not None:
                with torch.no_grad():
                    val_metrics = self._run_event_epoch(
                        val_loader,
                        optimizer=None,
                        detach_pre_event=True,
                        grad_clip=None,
                    )
                log.update({f'val_{k}': v for k, v in val_metrics.items()})
            history.append(log)
            if epoch % cfg.log_every == 0:
                print(log)
        return history

    def joint_finetune(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        cfg: Optional[JointTrainConfig] = None,
    ) -> list[dict[str, float]]:
        cfg = cfg or JointTrainConfig()
        self.criterion.lambda_recent_reg = cfg.lambda_recent_reg
        self.criterion.lambda_event_reg = cfg.lambda_event_reg

        unfreeze_module(self.model.base)
        unfreeze_module(self.model.recent)
        unfreeze_module(self.model.event)

        optimizer = torch.optim.Adam(
            [
                {'params': self.model.base.parameters(), 'lr': cfg.lr_base},
                {'params': self.model.recent.parameters(), 'lr': cfg.lr_recent},
                {'params': self.model.event.parameters(), 'lr': cfg.lr_event},
            ],
            weight_decay=cfg.weight_decay,
        )

        history: list[dict[str, float]] = []
        for epoch in range(1, cfg.epochs + 1):
            train_metrics = self._run_joint_epoch(train_loader, optimizer=optimizer, joint_cfg=cfg)
            log = {'epoch': epoch, 'stage': 'joint', **{f'train_{k}': v for k, v in train_metrics.items()}}
            if val_loader is not None:
                with torch.no_grad():
                    val_metrics = self._run_joint_epoch(val_loader, optimizer=None, joint_cfg=cfg)
                log.update({f'val_{k}': v for k, v in val_metrics.items()})
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
    ) -> dict[str, list[dict[str, float]]]:
        return {
            'stage1': self.train_stage1(stage1_train_loader, stage1_val_loader, stage1_cfg),
            'stage2': self.train_stage2(stage2_train_loader, stage2_val_loader, stage2_cfg),
            'stage3': self.train_stage3(stage3_train_loader, stage3_val_loader, stage3_cfg),
            'joint': self.joint_finetune(joint_train_loader, joint_val_loader, joint_cfg),
        }

    def save_checkpoint(self, path: str | Path, extra: Optional[dict[str, Any]] = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'model_state_dict': self.model.state_dict(),
            'model_config': self.model.cfg.__dict__,
            'criterion': {
                'lambda_recent_reg': self.criterion.lambda_recent_reg,
                'lambda_event_reg': self.criterion.lambda_event_reg,
            },
        }
        if extra is not None:
            payload['extra'] = extra
        torch.save(payload, path)

    @staticmethod
    def load_checkpoint(path: str | Path, device: str | torch.device = 'cpu') -> 'TrafficGNNTrainer':
        ckpt = torch.load(path, map_location=device)
        model_cfg = ModelConfig(**ckpt['model_config'])
        model = TrafficGNNSystem(model_cfg)
        model.load_state_dict(ckpt['model_state_dict'])
        criterion_cfg = ckpt.get('criterion', {})
        criterion = TrafficGNNLoss(
            lambda_recent_reg=criterion_cfg.get('lambda_recent_reg', 1e-4),
            lambda_event_reg=criterion_cfg.get('lambda_event_reg', 1e-4),
        )
        return TrafficGNNTrainer(model=model, device=device, criterion=criterion)


if __name__ == '__main__':
    cfg = ModelConfig(
        static_dim=16,
        profile_dim=8,
        event_dim=6,
        bank_hidden_dim=64,
    )
    model = TrafficGNNSystem(cfg)
    trainer = TrafficGNNTrainer(model, device='cpu')
    print('已完成拆分：GNN.py 只保留总模型；configs/ 放配置；utils/ 放工具；train.py 放训练代码。')
