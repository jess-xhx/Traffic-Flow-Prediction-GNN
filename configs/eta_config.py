from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass
class TripTokenConfig:
    # 数值特征维度，例如 trip_num_feat 的长度
    numeric_dim: int

    # 类别特征相关
    num_categories: int = 5
    cat_bucket_size: int = 4096
    cat_emb_dim: int = 16

    # 编码器结构
    hidden_dim: int = 128
    dropout: float = 0.1


@dataclass
class RouteTokenConfig:
    # 每条 edge 的静态特征维度，通常等于 x_static.shape[1]
    static_dim: int

    # GNN bank 的隐藏维度，通常等于 gnn_config.model.bank_hidden_dim
    bank_hidden_dim: int

    # 路径 token 额外特征
    extra_numeric_dim: int = 0
    turn_feat_dim: int = 0
    time_emb_dim: int = 32

    # 编码器结构
    hidden_dim: int = 128
    dropout: float = 0.1

    # 路径进入时间递推时的最小速度下限
    min_speed_kmh: float = 5.0


@dataclass
class ETAEncoderConfig:
    d_model: int = 128
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.1
    max_route_len: int = 256


@dataclass
class ETAHeadConfig:
    input_dim: int = 128
    hidden_dim: int = 128
    dropout: float = 0.1
    predict_uncertainty: bool = False


@dataclass
class SameCityETAConfig:
    trip_token: TripTokenConfig
    route_token: RouteTokenConfig
    encoder: ETAEncoderConfig = field(default_factory=ETAEncoderConfig)
    head: ETAHeadConfig = field(default_factory=ETAHeadConfig)
    freeze_gnn_backbone: bool = True


def save_eta_config(cfg: SameCityETAConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(cfg), f, allow_unicode=True, sort_keys=False)


def load_eta_config(path: str | Path) -> SameCityETAConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    trip_cfg = TripTokenConfig(**data["trip_token"])
    route_cfg = RouteTokenConfig(**data["route_token"])
    encoder_cfg = ETAEncoderConfig(**data.get("encoder", {}))
    head_cfg = ETAHeadConfig(**data.get("head", {}))

    return SameCityETAConfig(
        trip_token=trip_cfg,
        route_token=route_cfg,
        encoder=encoder_cfg,
        head=head_cfg,
        freeze_gnn_backbone=bool(data.get("freeze_gnn_backbone", True)),
    )
