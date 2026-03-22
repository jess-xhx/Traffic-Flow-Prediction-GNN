from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
import sys
from bisect import bisect_right
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader, Dataset, Sampler


PROJECT_ROOT = Path(__file__).resolve().parent
for candidate in [PROJECT_ROOT, PROJECT_ROOT / 'models', PROJECT_ROOT / '3.models']:
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

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
from utils.gnn_data import build_dataloader, load_split_datasets
from utils.gnn_loss import TrafficGNNLoss
from utils.gnn_utils import (
    ensure_bank_layout,
    ensure_dir,
    ensure_node_vector_layout,
    freeze_module,
    prepare_batch,
    save_json,
    set_seed,
    unfreeze_module,
)
from GNN import TrafficGNNSystem


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
        return {'loss': total_loss / denom, 'mae': total_mae / denom}

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
        return {'loss': total_loss / denom, 'mae': total_mae / denom, 'reg': total_reg / denom}

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
        return {'loss': total_loss / denom, 'mae': total_mae / denom, 'reg': total_reg / denom}

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
        self.logger.info('%s', epoch_log)

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

        unfreeze_module(self.model.base)
        freeze_module(self.model.recent)
        freeze_module(self.model.event)

        optimizer = torch.optim.Adam(self.model.base.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        history: list[dict[str, float]] = []
        best_metric: float | None = None
        checkpoint_dir = Path(checkpoint_dir)
        history_csv_path = Path(history_dir) / 'stage1_history.csv'

        for epoch in range(1, cfg.epochs + 1):
            train_metrics = self._run_base_epoch(train_loader, optimizer)
            log = {'epoch': epoch, 'stage': 1, **{f'train_{k}': v for k, v in train_metrics.items()}}
            if val_loader is not None:
                with torch.no_grad():
                    val_metrics = self._run_base_epoch(val_loader, optimizer=None)
                log.update({f'val_{k}': v for k, v in val_metrics.items()})
            best_metric = self._finalize_epoch(
                'stage1', log, history, history_csv_path, checkpoint_dir, monitor, monitor_mode, save_every_epoch, best_metric
            )
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

        freeze_module(self.model.base)
        unfreeze_module(self.model.recent)
        freeze_module(self.model.event)

        optimizer = torch.optim.Adam(self.model.recent.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        history: list[dict[str, float]] = []
        best_metric: float | None = None
        checkpoint_dir = Path(checkpoint_dir)
        history_csv_path = Path(history_dir) / 'stage2_history.csv'

        for epoch in range(1, cfg.epochs + 1):
            train_metrics = self._run_recent_epoch(
                train_loader, optimizer=optimizer, detach_base=True, grad_clip=cfg.grad_clip
            )
            log = {'epoch': epoch, 'stage': 2, **{f'train_{k}': v for k, v in train_metrics.items()}}
            if val_loader is not None:
                with torch.no_grad():
                    val_metrics = self._run_recent_epoch(
                        val_loader, optimizer=None, detach_base=True, grad_clip=None
                    )
                log.update({f'val_{k}': v for k, v in val_metrics.items()})
            best_metric = self._finalize_epoch(
                'stage2', log, history, history_csv_path, checkpoint_dir, monitor, monitor_mode, save_every_epoch, best_metric
            )
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

        freeze_module(self.model.base)
        freeze_module(self.model.recent)
        unfreeze_module(self.model.event)

        optimizer = torch.optim.Adam(self.model.event.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        history: list[dict[str, float]] = []
        best_metric: float | None = None
        checkpoint_dir = Path(checkpoint_dir)
        history_csv_path = Path(history_dir) / 'stage3_history.csv'

        for epoch in range(1, cfg.epochs + 1):
            train_metrics = self._run_event_epoch(
                train_loader, optimizer=optimizer, detach_pre_event=True, grad_clip=cfg.grad_clip
            )
            log = {'epoch': epoch, 'stage': 3, **{f'train_{k}': v for k, v in train_metrics.items()}}
            if val_loader is not None:
                with torch.no_grad():
                    val_metrics = self._run_event_epoch(
                        val_loader, optimizer=None, detach_pre_event=True, grad_clip=None
                    )
                log.update({f'val_{k}': v for k, v in val_metrics.items()})
            best_metric = self._finalize_epoch(
                'stage3', log, history, history_csv_path, checkpoint_dir, monitor, monitor_mode, save_every_epoch, best_metric
            )
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
        best_metric: float | None = None
        checkpoint_dir = Path(checkpoint_dir)
        history_csv_path = Path(history_dir) / 'joint_history.csv'

        for epoch in range(1, cfg.epochs + 1):
            train_metrics = self._run_joint_epoch(train_loader, optimizer=optimizer, joint_cfg=cfg)
            log = {'epoch': epoch, 'stage': 'joint', **{f'train_{k}': v for k, v in train_metrics.items()}}
            if val_loader is not None:
                with torch.no_grad():
                    val_metrics = self._run_joint_epoch(val_loader, optimizer=None, joint_cfg=cfg)
                log.update({f'val_{k}': v for k, v in val_metrics.items()})
            best_metric = self._finalize_epoch(
                'joint', log, history, history_csv_path, checkpoint_dir, monitor, monitor_mode, save_every_epoch, best_metric
            )
        return history





def _safe_torch_load(path: str | Path) -> Any:
    path = Path(path)
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')



def _cast_tensor_tree_fp32(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        if obj.dtype.is_floating_point:
            return obj.float()
        return obj
    if isinstance(obj, dict):
        return {k: _cast_tensor_tree_fp32(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_cast_tensor_tree_fp32(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_cast_tensor_tree_fp32(v) for v in obj)
    return obj



def _resolve_data_path(path_str: str | Path | None, base_dir: Path) -> Optional[Path]:
    if path_str is None or path_str == '':
        return None
    p = Path(path_str)
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    return p



def _get_cfg_value(cfg: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if hasattr(cfg, name):
            value = getattr(cfg, name)
            if value is not None:
                return value
    return default



def _looks_like_sharded_root(path: Path) -> bool:
    return (path / 'manifest.json').exists() and (path / 'reference_graph_and_base.pt').exists()



def detect_sharded_dataset_root(data_cfg: Any, base_dir: Path) -> Optional[Path]:
    candidates: list[Path] = []

    direct_root_fields = [
        'dataset_dir',
        'data_dir',
        'dataset_root',
        'root_dir',
        'root',
        'sharded_root',
        'mock_data_dir',
        'input_dir',
        'manifest_dir',
    ]
    for field in direct_root_fields:
        value = _get_cfg_value(data_cfg, field)
        resolved = _resolve_data_path(value, base_dir)
        if resolved is not None:
            candidates.append(resolved)

    manifest_path = _resolve_data_path(_get_cfg_value(data_cfg, 'manifest_path'), base_dir)
    if manifest_path is not None:
        candidates.append(manifest_path.parent if manifest_path.suffix else manifest_path)

    split_path_fields = [
        'train_path', 'val_path', 'test_path',
        'train_file', 'val_file', 'test_file',
        'train_pt', 'val_pt', 'test_pt',
        'train_data_path', 'val_data_path', 'test_data_path',
    ]
    for field in split_path_fields:
        value = _get_cfg_value(data_cfg, field)
        resolved = _resolve_data_path(value, base_dir)
        if resolved is None:
            continue
        if resolved.is_dir() and resolved.name in {'train', 'val', 'test'}:
            candidates.append(resolved.parent)
        else:
            candidates.append(resolved.parent)

    candidates.extend([
        (base_dir / 'mock_traffic_output').resolve(),
        base_dir.resolve(),
    ])

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if _looks_like_sharded_root(candidate):
            return candidate
    return None



class ShardedTrafficDataset(Dataset):
    def __init__(self, root_dir: str | Path, split: str):
        self.root_dir = Path(root_dir)
        self.split = split

        manifest_path = self.root_dir / 'manifest.json'
        if not manifest_path.exists():
            raise FileNotFoundError(f'找不到 manifest.json: {manifest_path}')
        with manifest_path.open('r', encoding='utf-8') as f:
            self.manifest = json.load(f)

        self.format = self.manifest.get('format')
        if self.format != 'sharded_pt_v2':
            raise ValueError(f'当前仅支持 sharded_pt_v2，实际为: {self.format}')

        split_info = self.manifest.get('splits', {}).get(split)
        if split_info is None:
            raise KeyError(f'manifest 中不存在 split={split}')
        self.split_info = split_info

        reference_name = self.manifest.get('reference_file', 'reference_graph_and_base.pt')
        self.reference = _cast_tensor_tree_fp32(_safe_torch_load(self.root_dir / reference_name))

        self.shard_paths = [self.root_dir / split / name for name in split_info.get('shards', [])]
        self.shard_sizes: list[int] = []
        self.shard_offsets: list[int] = [0]
        for shard_path in self.shard_paths:
            shard_samples = _safe_torch_load(shard_path)
            shard_size = len(shard_samples)
            self.shard_sizes.append(shard_size)
            self.shard_offsets.append(self.shard_offsets[-1] + shard_size)
            del shard_samples

        discovered_num_samples = self.shard_offsets[-1]
        declared_num_samples = int(split_info.get('num_samples', discovered_num_samples))
        self.num_samples = discovered_num_samples if discovered_num_samples > 0 else declared_num_samples
        if declared_num_samples != discovered_num_samples and discovered_num_samples > 0:
            self.num_samples = discovered_num_samples

        self._cached_shard_idx: int | None = None
        self._cached_shard_samples: list[dict[str, Any]] | None = None

    def __len__(self) -> int:
        return self.num_samples

    def _load_shard(self, shard_idx: int) -> list[dict[str, Any]]:
        if self._cached_shard_idx == shard_idx and self._cached_shard_samples is not None:
            return self._cached_shard_samples
        shard_path = self.shard_paths[shard_idx]
        shard_samples = _cast_tensor_tree_fp32(_safe_torch_load(shard_path))
        self._cached_shard_idx = shard_idx
        self._cached_shard_samples = shard_samples
        return shard_samples

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        shard_idx = bisect_right(self.shard_offsets, index) - 1
        local_idx = index - self.shard_offsets[shard_idx]
        shard_samples = self._load_shard(shard_idx)
        sample = shard_samples[local_idx]

        merged = {
            'x_static': self.reference['x_static'],
            'edge_index': self.reference['edge_index'],
            'y_base_bank': self.reference['y_base_bank'],
            'profile_feat': self.reference['profile_feat'],
        }
        merged.update(sample)
        return merged



class ShardGroupedSampler(Sampler[int]):
    def __init__(self, dataset: ShardedTrafficDataset, shuffle: bool = False, seed: int = 42):
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        shard_order = list(range(len(self.dataset.shard_paths)))
        if self.shuffle:
            rng.shuffle(shard_order)

        for shard_idx in shard_order:
            start = self.dataset.shard_offsets[shard_idx]
            end = self.dataset.shard_offsets[shard_idx + 1]
            indices = list(range(start, end))
            if self.shuffle:
                rng.shuffle(indices)
            for index in indices:
                yield index
        self.epoch += 1



def single_item_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if len(batch) != 1:
        raise ValueError(
            '当前 sharded_pt_v2 数据在 train.py 中按单样本图批处理，batch_size 必须为 1。'
        )
    return batch[0]



def build_sharded_dataloader(
    dataset: Optional[ShardedTrafficDataset],
    data_cfg: Any,
    split: str,
    logger: Optional[logging.Logger] = None,
) -> Optional[DataLoader]:
    if dataset is None or len(dataset) == 0:
        return None

    batch_size = int(_get_cfg_value(data_cfg, f'{split}_batch_size', 'batch_size', default=1) or 1)
    if batch_size != 1 and logger is not None:
        logger.warning('检测到 %s batch_size=%s，sharded_pt_v2 模式下已强制改为 1。', split, batch_size)

    pin_memory = bool(_get_cfg_value(data_cfg, 'pin_memory', default=False))
    seed = int(_get_cfg_value(data_cfg, 'seed', default=42) or 42)
    sampler = ShardGroupedSampler(dataset, shuffle=(split == 'train'), seed=seed)

    return DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
        collate_fn=single_item_collate,
    )



def load_datasets_and_dataloaders(
    data_cfg: Any,
    base_dir: Path,
    logger: Optional[logging.Logger] = None,
) -> tuple[dict[str, Any], Optional[DataLoader], Optional[DataLoader], Optional[DataLoader], dict[str, Any]]:
    sharded_root = detect_sharded_dataset_root(data_cfg, base_dir)
    if sharded_root is not None:
        datasets: dict[str, Any] = {}
        for split in ('train', 'val', 'test'):
            try:
                dataset = ShardedTrafficDataset(sharded_root, split)
            except KeyError:
                dataset = None
            if dataset is not None and len(dataset) == 0:
                dataset = None
            datasets[split] = dataset

        train_loader = build_sharded_dataloader(datasets['train'], data_cfg, 'train', logger=logger)
        val_loader = build_sharded_dataloader(datasets['val'], data_cfg, 'val', logger=logger)
        test_loader = build_sharded_dataloader(datasets['test'], data_cfg, 'test', logger=logger)
        data_info = {
            'data_mode': 'sharded_pt_v2',
            'data_root': str(sharded_root),
        }
        return datasets, train_loader, val_loader, test_loader, data_info

    datasets = load_split_datasets(data_cfg, base_dir=base_dir)
    train_loader = build_dataloader(datasets['train'], data_cfg, split='train')
    val_loader = build_dataloader(datasets['val'], data_cfg, split='val') if datasets['val'] is not None else None
    test_loader = build_dataloader(datasets['test'], data_cfg, split='test') if datasets['test'] is not None else None
    data_info = {
        'data_mode': 'legacy_single_pt',
        'data_root': None,
    }
    return datasets, train_loader, val_loader, test_loader, data_info

def resolve_device(device_str: str) -> str:
    if device_str == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return device_str



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



def setup_logger(log_file: str | Path) -> logging.Logger:
    logger = logging.getLogger('traffic_gnn_train')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger



def run_training(cfg: ExperimentConfig, config_path: str | Path) -> dict[str, Any]:
    paths = build_run_paths(cfg, config_path)
    logger = setup_logger(paths.log_file)
    logger.info('读取配置: %s', paths.config_path)
    logger.info('输出目录: %s', paths.output_dir)

    set_seed(cfg.runtime.seed, deterministic=cfg.runtime.deterministic)
    resolved_device = resolve_device(cfg.runtime.device)
    logger.info('使用设备: %s', resolved_device)

    dump_experiment_config(cfg, Path(paths.output_dir) / 'resolved_config.yaml')
    save_json(dataclass_to_dict(cfg), Path(paths.output_dir) / 'resolved_config.json')

    datasets, train_loader, val_loader, test_loader, data_info = load_datasets_and_dataloaders(
        cfg.data, Path(paths.config_path).parent, logger=logger
    )
    logger.info('数据加载模式: %s', data_info['data_mode'])
    if data_info.get('data_root'):
        logger.info('数据根目录: %s', data_info['data_root'])
    logger.info(
        '数据集大小: train=%d, val=%s, test=%s',
        len(datasets['train']),
        len(datasets['val']) if datasets['val'] is not None else 'None',
        len(datasets['test']) if datasets['test'] is not None else 'None',
    )

    if cfg.runtime.resume_checkpoint:
        resume_path = Path(cfg.runtime.resume_checkpoint)
        if not resume_path.is_absolute():
            resume_path = (PROJECT_ROOT / resume_path).resolve()
        logger.info('从 checkpoint 恢复: %s', resume_path)
        trainer = TrafficGNNTrainer.load_checkpoint(resume_path, device=resolved_device, logger=logger)
    else:
        model = TrafficGNNSystem(cfg.model)
        criterion = TrafficGNNLoss(
            lambda_recent_reg=cfg.joint.lambda_recent_reg,
            lambda_event_reg=cfg.joint.lambda_event_reg,
        )
        trainer = TrafficGNNTrainer(model=model, device=resolved_device, criterion=criterion, logger=logger)

    histories: dict[str, Any] = {}
    checkpoint_dir = Path(paths.checkpoint_dir)
    history_dir = Path(paths.history_dir)

    if cfg.runtime.use_stage1:
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

    if cfg.runtime.use_stage2:
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

    if cfg.runtime.use_stage3:
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

    if cfg.runtime.use_joint:
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

    trainer.save_checkpoint(
        checkpoint_dir / 'final_model.pt',
        extra={
            'histories': histories,
            'test_loader_exists': test_loader is not None,
        },
    )
    save_json(histories, history_dir / 'all_histories.json')
    logger.info('训练完成，最终模型已保存到: %s', checkpoint_dir / 'final_model.pt')

    if test_loader is not None:
        logger.info('已检测到 test 集，但当前 train.py 默认不单独跑 test 指标。可按需要继续扩展。')

    return {
        'paths': dataclass_to_dict(paths),
        'histories': histories,
    }



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Traffic GNN 项目入口训练脚本')
    parser.add_argument('--config', type=str, required=True, help='yaml 配置文件路径')
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    cfg = load_experiment_config(args.config)
    run_training(cfg, args.config)


if __name__ == '__main__':
    main()
