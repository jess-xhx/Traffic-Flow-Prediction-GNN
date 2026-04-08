from __future__ import annotations

import gc

import argparse
import csv
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from configs.gnn_config import (
    ExperimentConfig,
    JointTrainConfig,
    ModelConfig,
    ResolvedPaths,
    StageTrainConfig,
    dataclass_to_dict,
    dump_experiment_config,
    load_experiment_config,
)
from utils.gnn_loss import TrafficGNNLoss
from utils.gnn_utils import (
    ensure_bank_layout,
    ensure_dir,
    freeze_module,
    prepare_batch,
    save_json,
    set_seed,
    unfreeze_module,
)

from Dataset.gnn_dataset import load_datasets_and_dataloaders
from models.GNN import TrafficGNNSystem


def is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_initialized() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def maybe_barrier() -> None:
    if is_dist_initialized():
        dist.barrier()


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def resolve_device(device_str: str) -> str:
    if device_str == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return device_str


def init_distributed_if_needed(requested_device: str | torch.device) -> dict[str, Any]:
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))

    requested_device = str(requested_device)
    use_cuda = requested_device.startswith('cuda') and torch.cuda.is_available()

    if world_size <= 1:
        device = requested_device if requested_device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        return {
            'distributed': False,
            'backend': None,
            'world_size': 1,
            'rank': 0,
            'local_rank': 0,
            'device': device,
        }

    if use_cuda:
        torch.cuda.set_device(local_rank)
        backend = 'nccl'
        device = f'cuda:{local_rank}'
    else:
        backend = 'gloo'
        device = 'cpu'

    if not is_dist_initialized():
        dist.init_process_group(backend=backend, init_method='env://', rank=rank, world_size=world_size)

    return {
        'distributed': True,
        'backend': backend,
        'world_size': world_size,
        'rank': rank,
        'local_rank': local_rank,
        'device': device,
    }


class TrafficGNNTrainer:
    def __init__(
        self,
        model: TrafficGNNSystem,
        device: str | torch.device = 'cpu',
        criterion: Optional[TrafficGNNLoss] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.criterion = criterion or TrafficGNNLoss()
        self.logger = logger or logging.getLogger(__name__)
        self._is_main_process = True
        self.use_amp = (self.device.type == 'cuda')
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

    def _select_batch_keys(self, raw_batch: dict[str, Any], keys: list[str]) -> dict[str, Any]:
        selected = {}
        for k in keys:
            if k in raw_batch:
                selected[k] = raw_batch[k]
        return selected

    def _prepare_stage_batch(self, raw_batch: dict[str, Any], keys: list[str]) -> dict[str, Any]:
        selected = self._select_batch_keys(raw_batch, keys)
        return prepare_batch(selected, self.device) 

    def clear_cuda_memory(self, tag: str = '') -> None:
        if self.is_main_process and tag:
            self.logger.info('清理显存: %s', tag)

        gc.collect()

        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize(self.device)
            except Exception:
                pass
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass

    @property
    def core_model(self) -> TrafficGNNSystem:
        return unwrap_model(self.model)

    @property
    def is_main_process(self) -> bool:
        return self._is_main_process

    def set_main_process(self, is_main: bool) -> None:
        self._is_main_process = is_main

    def enable_ddp(self, find_unused_parameters: bool = False) -> None:
        if not is_dist_initialized() or isinstance(self.model, DDP):
            return
        if self.device.type == 'cuda':
            self.model = DDP(
                self.model,
                device_ids=[self.device.index],
                output_device=self.device.index,
                find_unused_parameters=find_unused_parameters,
            )
        else:
            self.model = DDP(self.model, find_unused_parameters=find_unused_parameters)

    def _reduce_scalar_dict(self, totals: dict[str, float], count: int) -> tuple[dict[str, float], int]:
        if not is_dist_initialized():
            return totals, count
        keys = list(totals.keys())
        values = [totals[k] for k in keys] + [float(count)]
        buf = torch.tensor(values, device=self.device, dtype=torch.float64)
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        reduced = {k: float(buf[i].item()) for i, k in enumerate(keys)}
        reduced_count = int(buf[-1].item())
        return reduced, reduced_count

    def _maybe_set_epoch(self, dataloader: Optional[DataLoader], epoch: int) -> None:
        if dataloader is None:
            return
        sampler = getattr(dataloader, 'sampler', None)
        if sampler is not None and hasattr(sampler, 'set_epoch'):
            sampler.set_epoch(epoch)

    def _pbar(self, dataloader: DataLoader, phase_name: str):
        return tqdm(
            dataloader,
            desc=phase_name,
            leave=True,
            disable=not self.is_main_process,
            dynamic_ncols=True,
            mininterval=1.0,
        )

    def _run_base_epoch(
        self,
        dataloader: DataLoader,
        optimizer: Optional[torch.optim.Optimizer] = None,
        max_steps_per_epoch: int | None = None,
    ) -> dict[str, float]:
        training = optimizer is not None
        self.model.train(training)

        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0

        pbar = self._pbar(dataloader, 'Train-Stage1' if training else 'Val-Stage1')
        for step, raw_batch in enumerate(pbar):
            if max_steps_per_epoch is not None and step >= max_steps_per_epoch:
                break
            batch = self._prepare_stage_batch(
            raw_batch,
            ['x_static', 'profile_feat', 'edge_index', 'y_base_bank', 'base_mask']
            )

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                out = self.model.forward(batch, mode='base')
                num_nodes = batch['x_static'].shape[0]
                y_base_bank = ensure_bank_layout(batch['y_base_bank'], num_nodes)
                base_mask = batch.get('base_mask')
                if base_mask is not None:
                    base_mask = ensure_bank_layout(base_mask, num_nodes)
                loss_dict = self.criterion.base_loss(out['pred_speed_bank'], y_base_bank, base_mask)
                loss = loss_dict['loss']

            if training:
                optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()

            loss_value = float(loss.detach().item())
            mae_value = float(loss_dict['mae'].item())
            total_loss += loss_value
            total_mae += mae_value
            num_batches += 1

            if self.is_main_process:
                pbar.set_postfix({'loss': f'{loss_value:.4f}', 'mae': f'{mae_value:.4f}', 'avg_loss': f'{total_loss / num_batches:.4f}'})

        totals, num_batches = self._reduce_scalar_dict({'loss': total_loss, 'mae': total_mae}, num_batches)
        denom = max(num_batches, 1)
        return {'loss': totals['loss'] / denom, 'mae': totals['mae'] / denom}

    def _run_recent_epoch(
        self,
        dataloader: DataLoader,
        optimizer: Optional[torch.optim.Optimizer] = None,
        detach_base: bool = True,
        grad_clip: float | None = None,
        max_steps_per_epoch: int | None = None,
    ) -> dict[str, float]:
        training = optimizer is not None
        self.model.train(training)

        total_loss = 0.0
        total_mae = 0.0
        total_reg = 0.0
        num_batches = 0

        pbar = self._pbar(dataloader, 'Train-Stage2' if training else 'Val-Stage2')
        for step, raw_batch in enumerate(pbar):
            if max_steps_per_epoch is not None and step >= max_steps_per_epoch:
                break
            batch = self._prepare_stage_batch(
                raw_batch,
                [
                    'x_static', 'profile_feat', 'edge_index',
                    'recent_speed_seq',
                    'y_future_bank', 'future_mask'
                ]
            )

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                out = self.model.forward(batch, mode='recent', detach_base=detach_base, return_full=False)
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
                optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                if grad_clip is not None:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.core_model.recent.parameters(), grad_clip)
                self.scaler.step(optimizer)
                self.scaler.update()

            loss_value = float(loss.detach().item())
            mae_value = float(loss_dict['mae'].item())
            reg_value = float(loss_dict['reg'].item())
            total_loss += loss_value
            total_mae += mae_value
            total_reg += reg_value
            num_batches += 1

            if self.is_main_process:
                pbar.set_postfix({
                    'loss': f'{loss_value:.4f}',
                    'mae': f'{mae_value:.4f}',
                    'reg': f'{reg_value:.4f}',
                    'avg_loss': f'{total_loss / num_batches:.4f}',
                })

        totals, num_batches = self._reduce_scalar_dict({'loss': total_loss, 'mae': total_mae, 'reg': total_reg}, num_batches)
        denom = max(num_batches, 1)
        return {'loss': totals['loss'] / denom, 'mae': totals['mae'] / denom, 'reg': totals['reg'] / denom}

    def _run_event_epoch(
        self,
        dataloader: DataLoader,
        optimizer: Optional[torch.optim.Optimizer] = None,
        detach_pre_event: bool = True,
        grad_clip: float | None = None,
        max_steps_per_epoch: int | None = None,
    ) -> dict[str, float]:
        training = optimizer is not None
        self.model.train(training)

        total_loss = 0.0
        total_mae = 0.0
        total_reg = 0.0
        num_batches = 0

        pbar = self._pbar(dataloader, 'Train-Stage3' if training else 'Val-Stage3')
        for step, raw_batch in enumerate(pbar):
            if max_steps_per_epoch is not None and step >= max_steps_per_epoch:
                break

            stage3_keys = [
                'x_static', 'profile_feat', 'edge_index',
                'recent_speed_seq',
                'event_weekday', 'event_slot',
                'event_vector',
                'y_event_bank',
                'event_mask',
            ]
            batch = self._prepare_stage_batch(raw_batch, stage3_keys)

            if 'y_event_bank' not in batch:
                raise KeyError('Stage3 已彻底关闭单点监督，样本必须提供 y_event_bank。')

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                out = self.model.forward(batch, mode='event', detach_pre_event=detach_pre_event)
                num_nodes = batch['x_static'].shape[0]

                y_event_bank = ensure_bank_layout(batch['y_event_bank'], num_nodes)

                event_mask = batch.get('event_mask')
                if event_mask is not None:
                    event_mask = ensure_bank_layout(event_mask, num_nodes)
                else:
                    event_mask = out['event_bank_mask']

                loss_dict = self.criterion.event_loss(
                    out['pred_speed_bank'],
                    y_event_bank,
                    out['delta_event_bank'],
                    event_mask,
                )
                loss = loss_dict['loss']

            if training:
                optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                if grad_clip is not None:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.core_model.event.parameters(), grad_clip)
                self.scaler.step(optimizer)
                self.scaler.update()

            loss_value = float(loss.detach().item())
            mae_value = float(loss_dict['mae'].item())
            reg_value = float(loss_dict['reg'].item())
            total_loss += loss_value
            total_mae += mae_value
            total_reg += reg_value
            num_batches += 1

            if self.is_main_process:
                pbar.set_postfix({
                    'loss': f'{loss_value:.4f}',
                    'mae': f'{mae_value:.4f}',
                    'reg': f'{reg_value:.4f}',
                    'avg_loss': f'{total_loss / num_batches:.4f}',
                })

        totals, num_batches = self._reduce_scalar_dict(
            {'loss': total_loss, 'mae': total_mae, 'reg': total_reg},
            num_batches,
        )
        denom = max(num_batches, 1)
        return {
            'loss': totals['loss'] / denom,
            'mae': totals['mae'] / denom,
            'reg': totals['reg'] / denom,
        }

    def _run_joint_epoch(
        self,
        dataloader: DataLoader,
        optimizer: Optional[torch.optim.Optimizer] = None,
        joint_cfg: Optional[JointTrainConfig] = None,
        max_steps_per_epoch: int | None = None,
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

        pbar = self._pbar(dataloader, 'Train-Joint' if training else 'Val-Joint')
        for step, raw_batch in enumerate(pbar):
            if max_steps_per_epoch is not None and step >= max_steps_per_epoch:
                break

            batch = self._prepare_stage_batch(
                raw_batch,
                [
                    'x_static', 'profile_feat', 'edge_index',
                    'y_base_bank', 'base_mask',
                    'recent_speed_seq',
                    'y_future_bank', 'future_mask',
                    'event_weekday', 'event_slot',
                    'event_vector',
                    'y_event_bank',
                    'event_mask',
                ]
            )

            if 'y_event_bank' not in batch:
                raise KeyError('Joint 阶段样本必须提供 y_event_bank。')

            num_nodes = batch['x_static'].shape[0]

            y_base_bank = ensure_bank_layout(batch['y_base_bank'], num_nodes)
            y_future_bank = ensure_bank_layout(batch['y_future_bank'], num_nodes)
            y_event_bank = ensure_bank_layout(batch['y_event_bank'], num_nodes)

            base_mask = batch.get('base_mask')
            if base_mask is not None:
                base_mask = ensure_bank_layout(base_mask, num_nodes)

            future_mask = batch.get('future_mask')
            if future_mask is not None:
                future_mask = ensure_bank_layout(future_mask, num_nodes)

            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                # 关键：一次共享 forward，整条链路 base -> recent -> event
                out = self.model.forward(batch, mode='joint_shared')

                # 1) base loss
                base_loss_dict = self.criterion.base_loss(
                    out['base_pred_speed_bank'],
                    y_base_bank,
                    base_mask,
                )

                # 2) recent loss
                recent_loss_dict = self.criterion.recent_loss(
                    out['recent_pred_speed_bank'],
                    y_future_bank,
                    out['delta_recent_bank'],
                    future_mask,
                )

                # 3) event loss
                event_mask = batch.get('event_mask')
                if event_mask is not None:
                    event_mask = ensure_bank_layout(event_mask, num_nodes)
                else:
                    event_mask = out['event_bank_mask']

                event_loss_dict = self.criterion.event_loss(
                    out['event_pred_speed_bank'],
                    y_event_bank,
                    out['delta_event_bank'],
                    event_mask,
                )

                # 总 loss：一次加权，一次 backward
                total_joint_loss = (
                    joint_cfg.alpha_base * base_loss_dict['loss']
                    + joint_cfg.beta_recent * recent_loss_dict['loss']
                    + joint_cfg.gamma_event * event_loss_dict['loss']
                )

            if training:
                self.scaler.scale(total_joint_loss).backward()

                if joint_cfg.grad_clip is not None:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), joint_cfg.grad_clip)

                self.scaler.step(optimizer)
                self.scaler.update()

            base_value = float(base_loss_dict['loss'].detach().item())
            recent_value = float(recent_loss_dict['loss'].detach().item())
            event_value = float(event_loss_dict['loss'].detach().item())
            loss_value = float(total_joint_loss.detach().item())

            total_loss += loss_value
            total_base += base_value
            total_recent += recent_value
            total_event += event_value
            num_batches += 1

            if self.is_main_process:
                pbar.set_postfix({
                    'loss': f'{loss_value:.4f}',
                    'base': f'{base_value:.4f}',
                    'recent': f'{recent_value:.4f}',
                    'event': f'{event_value:.4f}',
                    'avg_loss': f'{total_loss / num_batches:.4f}',
                })

            del out
            del base_loss_dict, recent_loss_dict, event_loss_dict
            del y_base_bank, y_future_bank, y_event_bank

            if base_mask is not None:
                del base_mask
            if future_mask is not None:
                del future_mask
            if event_mask is not None:
                del event_mask

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        totals, num_batches = self._reduce_scalar_dict(
            {
                'loss': total_loss,
                'base_loss': total_base,
                'recent_loss': total_recent,
                'event_loss': total_event,
            },
            num_batches,
        )
        denom = max(num_batches, 1)
        return {
            'loss': totals['loss'] / denom,
            'base_loss': totals['base_loss'] / denom,
            'recent_loss': totals['recent_loss'] / denom,
            'event_loss': totals['event_loss'] / denom,
        }

    def save_checkpoint(self, path: str | Path, extra: Optional[dict[str, Any]] = None) -> None:
        if not self.is_main_process:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'model_state_dict': self.core_model.state_dict(),
            'model_config': self.core_model.cfg.__dict__,
            'criterion': {
                'lambda_recent_reg': self.criterion.lambda_recent_reg,
                'lambda_event_reg': self.criterion.lambda_event_reg,
            },
        }
        if extra is not None:
            payload['extra'] = extra
        torch.save(payload, path)

    @staticmethod
    def load_checkpoint(
        path: str | Path,
        device: str | torch.device = 'cpu',
        logger: Optional[logging.Logger] = None,
    ) -> 'TrafficGNNTrainer':
        ckpt = torch.load(path, map_location=device)
        model_cfg = ModelConfig(**ckpt['model_config'])
        model = TrafficGNNSystem(model_cfg)
        model.load_state_dict(ckpt['model_state_dict'])
        criterion_cfg = ckpt.get('criterion', {})
        criterion = TrafficGNNLoss(
            lambda_recent_reg=criterion_cfg.get('lambda_recent_reg', 1e-4),
            lambda_event_reg=criterion_cfg.get('lambda_event_reg', 1e-4),
        )
        return TrafficGNNTrainer(model=model, device=device, criterion=criterion, logger=logger)

    def _is_better(self, current: float, best: float | None, mode: str) -> bool:
        if best is None:
            return True
        if mode == 'min':
            return current < best
        if mode == 'max':
            return current > best
        raise ValueError(f'不支持的 monitor_mode={mode}')

    def _metric_from_log(self, log: dict[str, float], monitor: str) -> float:
        if monitor in log:
            return float(log[monitor])
        fallback = f'val_{monitor}'
        if fallback in log:
            return float(log[fallback])
        raise KeyError(f'日志中找不到监控指标 {monitor}')

    def _append_history_csv(self, path: Path, row: dict[str, Any]) -> None:
        if not self.is_main_process:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = path.exists()
        with path.open('a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def _finalize_epoch(
        self,
        stage_name: str,
        epoch_log: dict[str, Any],
        history: list[dict[str, Any]],
        history_csv_path: Path,
        checkpoint_dir: Path,
        monitor: str,
        monitor_mode: str,
        save_every_epoch: bool,
        best_metric: float | None,
    ) -> float | None:
        history.append(epoch_log)
        self._append_history_csv(history_csv_path, epoch_log)
        if self.is_main_process:
            display_log = {}
            for k, v in epoch_log.items():
                if isinstance(v, float):
                    display_log[k] = round(v, 4)
                else:
                    display_log[k] = v
            self.logger.info('%s', display_log)

        metric_value = self._metric_from_log(epoch_log, monitor)
        if self._is_better(metric_value, best_metric, monitor_mode):
            best_metric = metric_value
            self.save_checkpoint(
                checkpoint_dir / f'{stage_name}_best.pt',
                extra={'stage': stage_name, 'epoch_log': epoch_log, 'history': history},
            )

        if save_every_epoch:
            self.save_checkpoint(
                checkpoint_dir / f'{stage_name}_epoch_{epoch_log["epoch"]:03d}.pt',
                extra={'stage': stage_name, 'epoch_log': epoch_log},
            )
        self.save_checkpoint(
            checkpoint_dir / f'{stage_name}_last.pt',
            extra={'stage': stage_name, 'epoch_log': epoch_log, 'history': history},
        )
        return best_metric

    def train_stage1(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        cfg: Optional[StageTrainConfig] = None,
        checkpoint_dir: str | Path = './checkpoints',
        history_dir: str | Path = './history',
        monitor: str = 'val_loss',
        monitor_mode: str = 'min',
        save_every_epoch: bool = True,
    ) -> list[dict[str, float]]:
        cfg = cfg or StageTrainConfig()
        unfreeze_module(self.core_model.base)
        freeze_module(self.core_model.recent)
        freeze_module(self.core_model.event)

        optimizer = torch.optim.Adam(self.core_model.base.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        history: list[dict[str, float]] = []
        best_metric: float | None = None
        checkpoint_dir = Path(checkpoint_dir)
        history_csv_path = Path(history_dir) / 'stage1_history.csv'

        for epoch in range(1, cfg.epochs + 1):
            self._maybe_set_epoch(train_loader, epoch)
            self._maybe_set_epoch(val_loader, epoch)
            train_metrics = self._run_base_epoch(
                train_loader,
                optimizer,
                max_steps_per_epoch=cfg.max_steps_per_epoch,
            )
            log = {'epoch': epoch, 'stage': 1, **{f'train_{k}': v for k, v in train_metrics.items()}}
            if val_loader is not None:
                with torch.no_grad():
                    val_metrics = self._run_base_epoch(
                        val_loader,
                         optimizer=None,
                         max_steps_per_epoch=cfg.max_steps_per_epoch,
                         )
                log.update({f'val_{k}': v for k, v in val_metrics.items()})
            best_metric = self._finalize_epoch(
                'stage1', log, history, history_csv_path, checkpoint_dir, monitor, monitor_mode, save_every_epoch, best_metric
            )

        del optimizer
        self.clear_cuda_memory('stage1_return')
        return history

    def train_stage2(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        cfg: Optional[StageTrainConfig] = None,
        checkpoint_dir: str | Path = './checkpoints',
        history_dir: str | Path = './history',
        monitor: str = 'val_loss',
        monitor_mode: str = 'min',
        save_every_epoch: bool = True,
    ) -> list[dict[str, float]]:
        cfg = cfg or StageTrainConfig()
        freeze_module(self.core_model.base)
        unfreeze_module(self.core_model.recent)
        freeze_module(self.core_model.event)

        optimizer = torch.optim.Adam(self.core_model.recent.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        history: list[dict[str, float]] = []
        best_metric: float | None = None
        checkpoint_dir = Path(checkpoint_dir)
        history_csv_path = Path(history_dir) / 'stage2_history.csv'

        for epoch in range(1, cfg.epochs + 1):
            self._maybe_set_epoch(train_loader, epoch)
            self._maybe_set_epoch(val_loader, epoch)
            train_metrics = self._run_recent_epoch(
                train_loader, 
                optimizer=optimizer, 
                detach_base=True, 
                grad_clip=cfg.grad_clip,
                max_steps_per_epoch=cfg.max_steps_per_epoch,
            )
            log = {'epoch': epoch, 'stage': 2, **{f'train_{k}': v for k, v in train_metrics.items()}}
            if val_loader is not None:
                with torch.no_grad():
                    val_metrics = self._run_recent_epoch(
                        val_loader, 
                        optimizer=None, 
                        detach_base=True, 
                        grad_clip=None,
                        max_steps_per_epoch=cfg.max_steps_per_epoch,
                        )
                log.update({f'val_{k}': v for k, v in val_metrics.items()})
            best_metric = self._finalize_epoch(
                'stage2', log, history, history_csv_path, checkpoint_dir, monitor, monitor_mode, save_every_epoch, best_metric
            )
        
        del optimizer
        self.clear_cuda_memory('stage2_return')
        return history

    def train_stage3(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        cfg: Optional[StageTrainConfig] = None,
        checkpoint_dir: str | Path = './checkpoints',
        history_dir: str | Path = './history',
        monitor: str = 'val_loss',
        monitor_mode: str = 'min',
        save_every_epoch: bool = True,
    ) -> list[dict[str, float]]:
        cfg = cfg or StageTrainConfig()
        freeze_module(self.core_model.base)
        freeze_module(self.core_model.recent)
        unfreeze_module(self.core_model.event)

        optimizer = torch.optim.Adam(self.core_model.event.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        history: list[dict[str, float]] = []
        best_metric: float | None = None
        checkpoint_dir = Path(checkpoint_dir)
        history_csv_path = Path(history_dir) / 'stage3_history.csv'

        for epoch in range(1, cfg.epochs + 1):
            self._maybe_set_epoch(train_loader, epoch)
            self._maybe_set_epoch(val_loader, epoch)
            train_metrics = self._run_event_epoch(
                train_loader, 
                optimizer=optimizer, 
                detach_pre_event=True, 
                grad_clip=cfg.grad_clip,
                max_steps_per_epoch=cfg.max_steps_per_epoch,
            )
            log = {'epoch': epoch, 'stage': 3, **{f'train_{k}': v for k, v in train_metrics.items()}}
            if val_loader is not None:
                with torch.no_grad():
                    val_metrics = self._run_event_epoch(
                        val_loader, 
                        optimizer=None,
                        detach_pre_event=True, 
                        grad_clip=None,
                        max_steps_per_epoch=cfg.max_steps_per_epoch,)
                log.update({f'val_{k}': v for k, v in val_metrics.items()})
            best_metric = self._finalize_epoch(
                'stage3', log, history, history_csv_path, checkpoint_dir, monitor, monitor_mode, save_every_epoch, best_metric
            )

        del optimizer
        self.clear_cuda_memory('stage3_return')
        return history

    def joint_finetune(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        cfg: Optional[JointTrainConfig] = None,
        checkpoint_dir: str | Path = './checkpoints',
        history_dir: str | Path = './history',
        monitor: str = 'val_loss',
        monitor_mode: str = 'min',
        save_every_epoch: bool = True,
    ) -> list[dict[str, float]]:
        cfg = cfg or JointTrainConfig()
        self.criterion.lambda_recent_reg = cfg.lambda_recent_reg
        self.criterion.lambda_event_reg = cfg.lambda_event_reg

        unfreeze_module(self.core_model.base)
        unfreeze_module(self.core_model.recent)
        unfreeze_module(self.core_model.event)

        try:
            optimizer = torch.optim.Adam(
                [
                    {'params': self.core_model.base.parameters(), 'lr': cfg.lr_base},
                    {'params': self.core_model.recent.parameters(), 'lr': cfg.lr_recent},
                    {'params': self.core_model.event.parameters(), 'lr': cfg.lr_event},
                ],
                weight_decay=cfg.weight_decay,
            )

            history: list[dict[str, float]] = []
            best_metric: float | None = None
            checkpoint_dir = Path(checkpoint_dir)
            history_csv_path = Path(history_dir) / 'joint_history.csv'

            for epoch in range(1, cfg.epochs + 1):
                self._maybe_set_epoch(train_loader, epoch)
                self._maybe_set_epoch(val_loader, epoch)
                train_metrics = self._run_joint_epoch(
                    train_loader, 
                    optimizer=optimizer, 
                    joint_cfg=cfg,
                    max_steps_per_epoch=cfg.max_steps_per_epoch,
                    )
                log = {'epoch': epoch, 'stage': 'joint', **{f'train_{k}': v for k, v in train_metrics.items()}}
                if val_loader is not None:
                    with torch.no_grad():
                        val_metrics = self._run_joint_epoch(
                            val_loader, 
                            optimizer=None, 
                            joint_cfg=cfg,
                            max_steps_per_epoch=cfg.max_steps_per_epoch,
                        )
                    log.update({f'val_{k}': v for k, v in val_metrics.items()})
                best_metric = self._finalize_epoch(
                    'joint', log, history, history_csv_path, checkpoint_dir, monitor, monitor_mode, save_every_epoch, best_metric
                )

            return history

        finally:
            self.core_model.base.set_checkpoint_enabled(self.core_model.cfg.enable_base_checkpoint)
            self.core_model.recent.set_checkpoint_enabled(self.core_model.cfg.enable_recent_checkpoint)


def _make_incremental_experiment_dir(output_root: Path, experiment_name: str) -> Path:
    output_root = output_root.resolve()
    ensure_dir(output_root)

    match = re.match(r'^(.*?)(\d+)?$', experiment_name)
    assert match is not None
    prefix = match.group(1) or experiment_name

    existing_indices: list[int] = []
    for child in output_root.iterdir():
        if not child.is_dir():
            continue
        child_match = re.match(rf'^{re.escape(prefix)}(\d+)$', child.name)
        if child_match:
            existing_indices.append(int(child_match.group(1)))

    next_idx = (max(existing_indices) + 1) if existing_indices else 1
    return output_root / f'{prefix}{next_idx:02d}'


def build_run_paths(cfg: ExperimentConfig, config_path: str | Path) -> ResolvedPaths:
    config_path = Path(config_path).resolve()
    project_root = PROJECT_ROOT.resolve()
    output_root = Path(cfg.runtime.output_dir)
    if not output_root.is_absolute():
        output_root = (project_root / output_root).resolve()

    output_dir = _make_incremental_experiment_dir(output_root, cfg.runtime.experiment_name)
    checkpoint_dir = output_dir / 'checkpoints'
    history_dir = output_dir / 'history'
    log_file = output_dir / 'train.log'

    ensure_dir(output_dir)
    ensure_dir(checkpoint_dir)
    ensure_dir(history_dir)

    return ResolvedPaths(
        config_path=str(config_path),
        project_root=str(project_root),
        output_dir=str(output_dir),
        checkpoint_dir=str(checkpoint_dir),
        history_dir=str(history_dir),
        log_file=str(log_file),
    )


def setup_logger(log_file: str | Path, is_main_process: bool = True) -> logging.Logger:
    logger = logging.getLogger('traffic_gnn_train')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    if is_main_process:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        logger.addHandler(logging.NullHandler())
    return logger


def run_training(cfg: ExperimentConfig, config_path: str | Path) -> dict[str, Any]:
    requested_device = resolve_device(cfg.runtime.device)
    dist_info = init_distributed_if_needed(requested_device)
    rank = dist_info['rank']
    world_size = dist_info['world_size']
    main_process = rank == 0

    if main_process:
        paths = build_run_paths(cfg, config_path)
        shared_paths = dataclass_to_dict(paths)
    else:
        shared_paths = None

    if dist_info['distributed']:
        obj_list = [shared_paths]
        dist.broadcast_object_list(obj_list, src=0)
        shared_paths = obj_list[0]
    paths = ResolvedPaths(**shared_paths)

    logger = setup_logger(paths.log_file, is_main_process=main_process)
    if main_process:
        logger.info('读取配置: %s', paths.config_path)
        logger.info('输出目录: %s', paths.output_dir)

    set_seed(cfg.runtime.seed + rank, deterministic=cfg.runtime.deterministic)
    resolved_device = dist_info['device']
    if main_process:
        logger.info('使用设备: %s', resolved_device)
        if dist_info['distributed']:
            logger.info('DDP 已启用: backend=%s, world_size=%d', dist_info['backend'], world_size)

    if main_process:
        dump_experiment_config(cfg, Path(paths.output_dir) / 'resolved_config.yaml')
        save_json(dataclass_to_dict(cfg), Path(paths.output_dir) / 'resolved_config.json')

    datasets, train_loader, val_loader, test_loader, data_info = load_datasets_and_dataloaders(
        cfg.data, Path(paths.config_path).parent, logger=logger, rank=rank, world_size=world_size
    )
    if main_process:
        logger.info('数据加载模式: %s', data_info['data_mode'])
        if data_info.get('data_root'):
            logger.info('数据根目录: %s', data_info['data_root'])
        logger.info(
            '总样本数: train=%s, val=%s, test=%s',
            data_info.get('num_train_samples_total', 0),
            data_info.get('num_val_samples_total', 0),
            data_info.get('num_test_samples_total', 0),
        )
        if train_loader is not None:
            logger.info('每个 rank 的 train steps: %d', len(train_loader))
        if val_loader is not None:
            logger.info('每个 rank 的 val steps: %d', len(val_loader))

    if cfg.runtime.resume_checkpoint:
        resume_path = Path(cfg.runtime.resume_checkpoint)
        if not resume_path.is_absolute():
            resume_path = (PROJECT_ROOT / resume_path).resolve()
        if main_process:
            logger.info('从 checkpoint 恢复: %s', resume_path)
        trainer = TrafficGNNTrainer.load_checkpoint(resume_path, device=resolved_device, logger=logger)
    else:
        model = TrafficGNNSystem(cfg.model)
        criterion = TrafficGNNLoss(
            lambda_recent_reg=cfg.joint.lambda_recent_reg,
            lambda_event_reg=cfg.joint.lambda_event_reg,
        )
        trainer = TrafficGNNTrainer(model=model, device=resolved_device, criterion=criterion, logger=logger)

    trainer.set_main_process(main_process)
    if dist_info['distributed']:
        trainer.enable_ddp(find_unused_parameters=True)

    histories: dict[str, Any] = {}
    checkpoint_dir = Path(paths.checkpoint_dir)
    history_dir = Path(paths.history_dir)

    if cfg.runtime.use_stage1:
        if main_process:
            logger.info('开始 Stage 1 训练')
        histories['stage1'] = trainer.train_stage1(
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg.stage1,
            checkpoint_dir=checkpoint_dir,
            history_dir=history_dir,
            monitor=cfg.runtime.monitor,
            monitor_mode=cfg.runtime.monitor_mode,
            save_every_epoch=cfg.runtime.save_every_epoch,
        )
        maybe_barrier()
        trainer.clear_cuda_memory('after_stage1')
        maybe_barrier()

    if cfg.runtime.use_stage2:
        if main_process:
            logger.info('开始 Stage 2 训练')
        histories['stage2'] = trainer.train_stage2(
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg.stage2,
            checkpoint_dir=checkpoint_dir,
            history_dir=history_dir,
            monitor=cfg.runtime.monitor,
            monitor_mode=cfg.runtime.monitor_mode,
            save_every_epoch=cfg.runtime.save_every_epoch,
        )
        maybe_barrier()
        trainer.clear_cuda_memory('after_stage2')
        maybe_barrier()

    if cfg.runtime.use_stage3:
        if main_process:
            logger.info('开始 Stage 3 训练')
        histories['stage3'] = trainer.train_stage3(
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg.stage3,
            checkpoint_dir=checkpoint_dir,
            history_dir=history_dir,
            monitor=cfg.runtime.monitor,
            monitor_mode=cfg.runtime.monitor_mode,
            save_every_epoch=cfg.runtime.save_every_epoch,
        )
        maybe_barrier()
        trainer.clear_cuda_memory('after_stage3')
        maybe_barrier()

    if cfg.runtime.use_joint:
        if main_process:
            logger.info('开始 Joint Finetune')
        histories['joint'] = trainer.joint_finetune(
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg.joint,
            checkpoint_dir=checkpoint_dir,
            history_dir=history_dir,
            monitor=cfg.runtime.monitor,
            monitor_mode=cfg.runtime.monitor_mode,
            save_every_epoch=cfg.runtime.save_every_epoch,
        )
        maybe_barrier()

    if main_process:
        trainer.save_checkpoint(
            checkpoint_dir / 'final_model.pt',
            extra={'histories': histories, 'test_loader_exists': test_loader is not None},
        )
        save_json(histories, history_dir / 'all_histories.json')
        logger.info('训练完成，最终模型已保存到: %s', checkpoint_dir / 'final_model.pt')
        if test_loader is not None:
            logger.info('已检测到 test 集，但当前脚本默认不单独跑 test 指标。')

    maybe_barrier()
    if dist_info['distributed'] and is_dist_initialized():
        dist.destroy_process_group()

    return {'paths': dataclass_to_dict(paths), 'histories': histories}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Traffic GNN Linux DDP 训练脚本')
    parser.add_argument('--config', type=str, required=True, help='yaml 配置文件路径')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_experiment_config(args.config)
    run_training(cfg, args.config)


if __name__ == '__main__':
    main()
