from __future__ import annotations

import json
import logging
import math
import random
from bisect import bisect_right
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

try:
    from utils.gnn_data import build_dataloader, load_split_datasets
except ImportError:
    from gnn_data import build_dataloader, load_split_datasets


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


class DistributedShardSampler(Sampler[int]):
    """Shard-aware distributed sampler with equal-length slices on every rank.

    For DDP each rank must iterate the same number of steps. We therefore pad or
    truncate the flattened index list to make it divisible by world_size.
    """

    def __init__(
        self,
        dataset: ShardedTrafficDataset,
        shuffle: bool = False,
        seed: int = 42,
        num_replicas: int = 1,
        rank: int = 0,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.num_replicas = max(1, int(num_replicas))
        self.rank = int(rank)
        self.drop_last = drop_last

        if self.drop_last:
            self.num_samples = len(self.dataset) // self.num_replicas
        else:
            self.num_samples = int(math.ceil(len(self.dataset) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        shard_order = list(range(len(self.dataset.shard_paths)))
        if self.shuffle:
            rng.shuffle(shard_order)

        all_indices: list[int] = []
        for shard_idx in shard_order:
            start = self.dataset.shard_offsets[shard_idx]
            end = self.dataset.shard_offsets[shard_idx + 1]
            indices = list(range(start, end))
            if self.shuffle:
                rng.shuffle(indices)
            all_indices.extend(indices)

        if self.drop_last:
            all_indices = all_indices[:self.total_size]
        else:
            if len(all_indices) < self.total_size:
                padding_size = self.total_size - len(all_indices)
                if padding_size <= len(all_indices):
                    all_indices += all_indices[:padding_size]
                else:
                    repeats = int(math.ceil(padding_size / max(1, len(all_indices))))
                    all_indices += (all_indices * repeats)[:padding_size]
            else:
                all_indices = all_indices[:self.total_size]

        rank_indices = all_indices[self.rank:self.total_size:self.num_replicas]
        assert len(rank_indices) == self.num_samples
        return iter(rank_indices)


def single_item_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if len(batch) != 1:
        raise ValueError('当前 sharded_pt_v2 数据按单样本图批处理，batch_size 必须为 1。')
    return batch[0]


def build_sharded_dataloader(
    dataset: Optional[ShardedTrafficDataset],
    data_cfg: Any,
    split: str,
    logger: Optional[logging.Logger] = None,
    rank: int = 0,
    world_size: int = 1,
) -> Optional[DataLoader]:
    if dataset is None or len(dataset) == 0:
        return None

    batch_size = int(_get_cfg_value(data_cfg, f'{split}_batch_size', 'batch_size', default=1) or 1)
    if batch_size != 1 and logger is not None:
        logger.warning('检测到 %s batch_size=%s，sharded_pt_v2 模式下已强制改为 1。', split, batch_size)

    pin_memory = bool(_get_cfg_value(data_cfg, 'pin_memory', default=False))
    num_workers = int(_get_cfg_value(data_cfg, 'num_workers', default=0) or 0)
    persistent_workers = bool(_get_cfg_value(data_cfg, 'persistent_workers', default=False)) and num_workers > 0
    drop_last = bool(_get_cfg_value(data_cfg, 'drop_last', default=False)) if split == 'train' else False
    seed = int(_get_cfg_value(data_cfg, 'seed', default=42) or 42)
    shuffle_train = bool(_get_cfg_value(data_cfg, 'shuffle_train', default=True))
    sampler = DistributedShardSampler(
        dataset,
        shuffle=(split == 'train' and shuffle_train),
        seed=seed,
        num_replicas=world_size,
        rank=rank,
        drop_last=drop_last,
    )

    return DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=single_item_collate,
    )


def load_datasets_and_dataloaders(
    data_cfg: Any,
    base_dir: Path,
    logger: Optional[logging.Logger] = None,
    rank: int = 0,
    world_size: int = 1,
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

        train_loader = build_sharded_dataloader(datasets['train'], data_cfg, 'train', logger=logger, rank=rank, world_size=world_size)
        val_loader = build_sharded_dataloader(datasets['val'], data_cfg, 'val', logger=logger, rank=rank, world_size=world_size)
        test_loader = build_sharded_dataloader(datasets['test'], data_cfg, 'test', logger=logger, rank=rank, world_size=world_size)
        data_info = {
            'data_mode': 'sharded_pt_v2',
            'data_root': str(sharded_root),
            'num_train_samples_total': len(datasets['train']) if datasets['train'] is not None else 0,
            'num_val_samples_total': len(datasets['val']) if datasets['val'] is not None else 0,
            'num_test_samples_total': len(datasets['test']) if datasets['test'] is not None else 0,
        }
        return datasets, train_loader, val_loader, test_loader, data_info

    if world_size > 1:
        raise NotImplementedError('当前 DDP 版仅支持 sharded_pt_v2 数据模式。')

    datasets = load_split_datasets(data_cfg, base_dir=base_dir)
    train_loader = build_dataloader(datasets['train'], data_cfg, split='train')
    val_loader = build_dataloader(datasets['val'], data_cfg, split='val') if datasets['val'] is not None else None
    test_loader = build_dataloader(datasets['test'], data_cfg, split='test') if datasets['test'] is not None else None
    data_info = {
        'data_mode': 'legacy_single_pt',
        'data_root': None,
        'num_train_samples_total': len(datasets['train']) if datasets['train'] is not None else 0,
        'num_val_samples_total': len(datasets['val']) if datasets['val'] is not None else 0,
        'num_test_samples_total': len(datasets['test']) if datasets['test'] is not None else 0,
    }
    return datasets, train_loader, val_loader, test_loader, data_info


__all__ = [
    'ShardedTrafficDataset',
    'DistributedShardSampler',
    'single_item_collate',
    'build_sharded_dataloader',
    'detect_sharded_dataset_root',
    'load_datasets_and_dataloaders',
]
