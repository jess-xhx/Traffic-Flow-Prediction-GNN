from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader, Dataset

from configs.gnn_config import DatasetConfig


class DictSampleDataset(Dataset):
    """
    样本必须是 dict，至少包含训练阶段所需字段。
    常用字段:
      x_static, profile_feat, edge_index,
      recent_speed_seq, target_weekday, target_slot, event_vector,
      y_base_bank, y_future_bank, y_target_speed,
      base_mask, future_mask, event_mask
    """

    def __init__(self, samples: list[dict[str, Any]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]



def _load_pt(path: Path) -> Any:
    return torch.load(path, map_location='cpu')



def _normalize_samples(obj: Any, source_path: Path) -> list[dict[str, Any]]:
    if isinstance(obj, list):
        if not all(isinstance(x, dict) for x in obj):
            raise TypeError(f'{source_path} 中 list 的元素必须全是 dict')
        return obj

    if isinstance(obj, dict):
        if 'samples' in obj:
            samples = obj['samples']
            if not isinstance(samples, list) or not all(isinstance(x, dict) for x in samples):
                raise TypeError(f'{source_path} 中 samples 字段必须是 list[dict]')
            return samples
        if all(isinstance(v, torch.Tensor) or isinstance(v, (int, float, str, bool, list, tuple, dict)) for v in obj.values()):
            return [obj]

    raise TypeError(
        f'无法从 {source_path} 解析样本。支持: list[dict]、包含 samples 的 dict、或单个样本 dict。'
    )



def _load_dataset_from_file(path: Path) -> DictSampleDataset:
    samples = _normalize_samples(_load_pt(path), path)
    return DictSampleDataset(samples)



def _load_dataset_from_directory(path: Path, file_pattern: str) -> DictSampleDataset:
    files = sorted(path.glob(file_pattern))
    if not files:
        raise FileNotFoundError(f'{path} 下没有匹配 {file_pattern} 的样本文件')
    samples = []
    for file in files:
        obj = _load_pt(file)
        samples.extend(_normalize_samples(obj, file))
    return DictSampleDataset(samples)



def _resolve_path(path_str: str | None, base_dir: Path) -> Path | None:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path



def _try_load_split_from_root(root_dir: Path, split: str, file_pattern: str) -> DictSampleDataset | None:
    split_file = root_dir / f'{split}.pt'
    if split_file.exists():
        return _load_dataset_from_file(split_file)

    split_dir = root_dir / split
    if split_dir.exists() and split_dir.is_dir():
        return _load_dataset_from_directory(split_dir, file_pattern)

    return None



def load_split_datasets(cfg: DatasetConfig, base_dir: str | Path = '.') -> dict[str, Dataset | None]:
    base_dir = Path(base_dir).resolve()
    datasets: dict[str, Dataset | None] = {'train': None, 'val': None, 'test': None}

    root_dir = _resolve_path(cfg.root_dir, base_dir)
    train_path = _resolve_path(cfg.train_path, base_dir)
    val_path = _resolve_path(cfg.val_path, base_dir)
    test_path = _resolve_path(cfg.test_path, base_dir)

    if root_dir is not None:
        if root_dir.is_file():
            obj = _load_pt(root_dir)
            if not isinstance(obj, dict):
                raise TypeError('当 root_dir 指向单个 .pt 文件时，它必须是包含 train/val/test 的 dict')
            for split in datasets.keys():
                split_obj = obj.get(split)
                if split_obj is not None:
                    datasets[split] = DictSampleDataset(_normalize_samples(split_obj, root_dir))
        elif root_dir.is_dir():
            for split in datasets.keys():
                datasets[split] = _try_load_split_from_root(root_dir, split, cfg.file_pattern)
        else:
            raise FileNotFoundError(f'找不到数据根目录: {root_dir}')

    explicit_paths = {'train': train_path, 'val': val_path, 'test': test_path}
    for split, path in explicit_paths.items():
        if path is None:
            continue
        if path.is_file():
            datasets[split] = _load_dataset_from_file(path)
        elif path.is_dir():
            datasets[split] = _load_dataset_from_directory(path, cfg.file_pattern)
        else:
            raise FileNotFoundError(f'找不到 {split} 数据路径: {path}')

    if datasets['train'] is None:
        raise FileNotFoundError('未找到 train 数据。请在 yaml 中设置 data.root_dir 或 data.train_path。')

    return datasets



def default_graph_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if len(batch) == 1:
        return batch[0]

    result: dict[str, Any] = {}
    keys = batch[0].keys()
    for key in keys:
        values = [sample[key] for sample in batch]
        first = values[0]
        if torch.is_tensor(first):
            same_shape = all(torch.is_tensor(v) and v.shape == first.shape for v in values)
            if same_shape:
                result[key] = torch.stack(values, dim=0)
            else:
                raise ValueError(
                    f'batch_size>1 时字段 {key} 的形状不一致，当前默认 collate 无法堆叠。'
                    '建议把 yaml 中 data.batch_size 设为 1。'
                )
        else:
            result[key] = values
    return result



def build_dataloader(
    dataset: Dataset,
    cfg: DatasetConfig,
    split: str,
    collate_fn: Callable[[list[dict[str, Any]]], dict[str, Any]] | None = None,
) -> DataLoader:
    if collate_fn is None:
        collate_fn = default_graph_collate

    is_train = split == 'train'
    num_workers = cfg.num_workers
    persistent_workers = cfg.persistent_workers and num_workers > 0

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle_train if is_train else False,
        num_workers=num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=cfg.drop_last if is_train else False,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
    )


__all__ = [
    'DictSampleDataset',
    'load_split_datasets',
    'default_graph_collate',
    'build_dataloader',
]
