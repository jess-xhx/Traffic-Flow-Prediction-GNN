from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Type, TypeVar

import yaml


T = TypeVar('T')


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
    grad_clip: float | None = 5.0
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
    grad_clip: float | None = 5.0
    log_every: int = 1


@dataclass
class DatasetConfig:
    root_dir: str | None = None
    train_path: str | None = None
    val_path: str | None = None
    test_path: str | None = None
    file_pattern: str = '*.pt'
    batch_size: int = 1
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    shuffle_train: bool = True
    drop_last: bool = False


@dataclass
class RuntimeConfig:
    seed: int = 42
    deterministic: bool = False
    device: str = 'auto'
    output_dir: str = './log'
    experiment_name: str = 'traffic_gnn'
    resume_checkpoint: str | None = None
    save_every_epoch: bool = True
    monitor: str = 'val_loss'
    monitor_mode: str = 'min'
    use_stage1: bool = True
    use_stage2: bool = True
    use_stage3: bool = True
    use_joint: bool = True


@dataclass
class ExperimentConfig:
    model: ModelConfig
    data: DatasetConfig = field(default_factory=DatasetConfig)
    stage1: StageTrainConfig = field(default_factory=StageTrainConfig)
    stage2: StageTrainConfig = field(default_factory=StageTrainConfig)
    stage3: StageTrainConfig = field(default_factory=StageTrainConfig)
    joint: JointTrainConfig = field(default_factory=JointTrainConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


@dataclass
class ResolvedPaths:
    config_path: str
    project_root: str
    output_dir: str
    checkpoint_dir: str
    history_dir: str
    log_file: str


def _dataclass_from_dict(cls: Type[T], data: dict[str, Any] | None) -> T:
    data = data or {}
    valid_names = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in data.items() if k in valid_names}
    return cls(**filtered)


def experiment_config_from_dict(data: dict[str, Any]) -> ExperimentConfig:
    if 'model' not in data:
        raise ValueError('yaml 配置中缺少 model 段')

    return ExperimentConfig(
        model=_dataclass_from_dict(ModelConfig, data.get('model')),
        data=_dataclass_from_dict(DatasetConfig, data.get('data')),
        stage1=_dataclass_from_dict(StageTrainConfig, data.get('stage1')),
        stage2=_dataclass_from_dict(StageTrainConfig, data.get('stage2')),
        stage3=_dataclass_from_dict(StageTrainConfig, data.get('stage3')),
        joint=_dataclass_from_dict(JointTrainConfig, data.get('joint')),
        runtime=_dataclass_from_dict(RuntimeConfig, data.get('runtime')),
    )


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    with path.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    return experiment_config_from_dict(data)


def dataclass_to_dict(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    raise TypeError(f'不支持的对象类型: {type(obj)}')


def dump_experiment_config(cfg: ExperimentConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        yaml.safe_dump(asdict(cfg), f, allow_unicode=True, sort_keys=False)


__all__ = [
    'ModelConfig',
    'StageTrainConfig',
    'JointTrainConfig',
    'DatasetConfig',
    'RuntimeConfig',
    'ExperimentConfig',
    'ResolvedPaths',
    'experiment_config_from_dict',
    'load_experiment_config',
    'dump_experiment_config',
    'dataclass_to_dict',
]
