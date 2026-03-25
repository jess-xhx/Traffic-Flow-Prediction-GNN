from __future__ import annotations

import argparse
import gc
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import torch

try:
    import yaml
except Exception:
    yaml = None

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


# -----------------------------
# 配置
# -----------------------------
@dataclass
class SimConfig:
    graph_bundle_path: Optional[str] = None
    x_static_path: Optional[str] = None
    edge_index_path: Optional[str] = None
    x_static_key: str = "x_static"
    edge_index_key: str = "edge_index"

    # 默认输出到当前脚本同级目录下
    output_dir: str = "./mock_traffic_output"

    num_train: int = 256
    num_val: int = 64
    num_test: int = 64

    recent_len: int = 24
    profile_dim: int = 8
    event_dim: int = 6

    base_speed_min: float = 20.0
    base_speed_max: float = 75.0
    observation_noise_std: float = 1.5
    bank_noise_std: float = 0.8
    future_noise_std: float = 1.2
    event_prob: float = 0.08
    seed: int = 42

    # 新增：防卡死/爆内存相关配置
    show_progress: bool = True
    save_chunk_size: int = 8
    storage_dtype: str = "float32"  # 可选: float32 / float16


# -----------------------------
# 基础工具
# -----------------------------
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _resolve_path(path_str: Optional[str], base_dir: Path) -> Optional[Path]:
    if not path_str:
        return None
    p = Path(path_str)
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    return p


def _load_any(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        return torch.load(path, map_location="cpu", weights_only=False)
    if suffix == ".npy":
        return np.load(path, allow_pickle=True)
    if suffix == ".npz":
        return dict(np.load(path, allow_pickle=True))
    if suffix in {".pkl", ".pickle"}:
        with open(path, "rb") as f:
            return pickle.load(f)
    raise ValueError(f"暂不支持的文件格式: {path}")


def _to_tensor(obj: Any, key: Optional[str] = None) -> torch.Tensor:
    if isinstance(obj, torch.Tensor):
        return obj.float() if obj.dtype.is_floating_point else obj.long()
    if isinstance(obj, np.ndarray):
        tensor = torch.from_numpy(obj)
        return tensor.float() if tensor.dtype.is_floating_point else tensor.long()
    if isinstance(obj, dict):
        if key is not None and key in obj:
            return _to_tensor(obj[key], None)
        if len(obj) == 1:
            return _to_tensor(next(iter(obj.values())), None)

    if key is not None and hasattr(obj, key):
        return _to_tensor(getattr(obj, key), None)

    if key is None and hasattr(obj, "x") and isinstance(getattr(obj, "x"), torch.Tensor):
        return _to_tensor(getattr(obj, "x"), None)

    raise TypeError(f"无法把对象转换成 Tensor。key={key}, type={type(obj)}")


def load_graph_inputs(cfg: SimConfig, base_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """
    支持两种方式：
    1) graph_bundle_path 指向单个文件，里面含 x_static / edge_index
    2) x_static_path 和 edge_index_path 分开给
    """
    if cfg.graph_bundle_path:
        bundle_path = _resolve_path(cfg.graph_bundle_path, base_dir)
        if bundle_path is None or not bundle_path.exists():
            raise FileNotFoundError(f"找不到 graph_bundle_path: {cfg.graph_bundle_path}")
        bundle = _load_any(bundle_path)
        x_static = _to_tensor(bundle, cfg.x_static_key).float()
        edge_index = _to_tensor(bundle, cfg.edge_index_key).long()
    else:
        x_path = _resolve_path(cfg.x_static_path, base_dir)
        e_path = _resolve_path(cfg.edge_index_path, base_dir)
        if x_path is None or not x_path.exists():
            raise FileNotFoundError("请提供有效的 x_static_path")
        if e_path is None or not e_path.exists():
            raise FileNotFoundError("请提供有效的 edge_index_path")
        x_static = _to_tensor(
            _load_any(x_path),
            cfg.x_static_key if x_path.suffix.lower() in {".pt", ".pth", ".npz"} else None,
        ).float()
        edge_index = _to_tensor(
            _load_any(e_path),
            cfg.edge_index_key if e_path.suffix.lower() in {".pt", ".pth", ".npz"} else None,
        ).long()

    if x_static.dim() != 2:
        raise ValueError(f"x_static 应为 [N, F_s]，实际 {tuple(x_static.shape)}")
    if edge_index.dim() != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index 应为 [2, E]，实际 {tuple(edge_index.shape)}")
    return x_static, edge_index


def zscore(x: torch.Tensor, dim: int = 0, eps: float = 1e-6) -> torch.Tensor:
    mean = x.mean(dim=dim, keepdim=True)
    std = x.std(dim=dim, keepdim=True).clamp_min(eps)
    return (x - mean) / std


def minmax_01(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mn = x.min()
    mx = x.max()
    return (x - mn) / (mx - mn + eps)


def compute_degree(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    deg = torch.zeros(num_nodes, dtype=torch.float32)
    src, dst = edge_index
    one = torch.ones_like(src, dtype=torch.float32)
    deg.index_add_(0, src, one)
    deg.index_add_(0, dst, one)
    return deg


def neighbor_mean(x: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    if x.dim() == 1:
        x = x.unsqueeze(-1)
        squeeze_back = True
    else:
        squeeze_back = False

    src, dst = edge_index
    out = torch.zeros((num_nodes, x.shape[1]), dtype=x.dtype)
    deg = torch.zeros((num_nodes, 1), dtype=x.dtype)

    out.index_add_(0, dst, x[src])
    deg.index_add_(0, dst, torch.ones((src.numel(), 1), dtype=x.dtype))
    out = out / deg.clamp_min(1.0)

    if squeeze_back:
        out = out.squeeze(-1)
    return out


def smooth_signal(x: torch.Tensor, edge_index: torch.Tensor, num_steps: int = 2, alpha: float = 0.65) -> torch.Tensor:
    num_nodes = x.shape[0]
    out = x.clone()
    for _ in range(num_steps):
        neigh = neighbor_mean(out, edge_index, num_nodes)
        out = alpha * out + (1.0 - alpha) * neigh
    return out


def ensure_edge_index_in_range(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    if edge_index.min() < 0 or edge_index.max() >= num_nodes:
        raise ValueError("edge_index 中存在越界节点编号")
    return edge_index.long()


def get_storage_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"storage_dtype 仅支持 {list(mapping.keys())}，当前为 {dtype_name}")
    return mapping[dtype_name]


def progress(iterable: Iterable, total: Optional[int], desc: str, enable: bool = True):
    if enable and tqdm is not None:
        return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True, leave=True)
    return iterable


def cast_tensor_tree(obj: Any, target_dtype: torch.dtype) -> Any:
    if isinstance(obj, torch.Tensor):
        if obj.dtype.is_floating_point:
            return obj.to(target_dtype)
        return obj
    if isinstance(obj, dict):
        return {k: cast_tensor_tree(v, target_dtype) for k, v in obj.items()}
    if isinstance(obj, list):
        return [cast_tensor_tree(v, target_dtype) for v in obj]
    if isinstance(obj, tuple):
        return tuple(cast_tensor_tree(v, target_dtype) for v in obj)
    return obj


# -----------------------------
# 生成长期 bank / profile
# -----------------------------
def build_base_bank_and_profile(
    x_static: torch.Tensor,
    edge_index: torch.Tensor,
    cfg: SimConfig,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """
    返回:
      y_base_bank: [7, 288, N]
      profile_feat: [N, 7, 288, F_p]
      aux: 若干中间量，后续近期/事件模拟会继续用
    """
    num_nodes, static_dim = x_static.shape
    x_norm = zscore(x_static, dim=0)

    gen = torch.Generator().manual_seed(cfg.seed)
    proj_w = torch.randn(static_dim, 3, generator=gen) / math.sqrt(max(static_dim, 1))
    proj = x_norm @ proj_w

    degree = compute_degree(edge_index, num_nodes)
    degree_norm = minmax_01(degree)

    road_score = 0.55 * proj[:, 0] + 0.25 * proj[:, 1] + 0.20 * degree_norm
    road_score = smooth_signal(road_score, edge_index, num_steps=2, alpha=0.7)
    road_score_norm = minmax_01(road_score)

    free_flow = cfg.base_speed_min + (cfg.base_speed_max - cfg.base_speed_min) * (0.25 + 0.75 * road_score_norm)
    congestion_amp = 0.14 + 0.20 * (1.0 - road_score_norm)
    volatility = 0.02 + 0.05 * minmax_01(torch.abs(proj[:, 2]))

    slots = torch.arange(288, dtype=torch.float32)
    hours = slots / 12.0

    morning_peak = torch.exp(-0.5 * ((hours - 8.0) / 1.6) ** 2)
    evening_peak = torch.exp(-0.5 * ((hours - 17.8) / 1.9) ** 2)
    midday_bump = torch.exp(-0.5 * ((hours - 13.0) / 2.5) ** 2)
    night_relief = torch.exp(-0.5 * ((hours - 3.0) / 2.8) ** 2)

    weekday_factors = torch.tensor([1.10, 1.05, 1.00, 1.00, 1.08, 0.72, 0.66], dtype=torch.float32)
    weekend_flags = torch.tensor([0, 0, 0, 0, 0, 1, 1], dtype=torch.float32)

    base_bank = torch.zeros((7, 288, num_nodes), dtype=torch.float32)
    for w in range(7):
        day_factor = weekday_factors[w]
        weekend = weekend_flags[w]

        congestion_curve = (
            0.85 * morning_peak * (1.0 - 0.55 * weekend)
            + 0.75 * evening_peak * (1.0 - 0.40 * weekend)
            + 0.22 * midday_bump * (1.0 + 0.25 * weekend)
            - 0.12 * night_relief
        )
        congestion_curve = congestion_curve.clamp_min(0.0)

        speed_day = free_flow.unsqueeze(0) * (
            1.0 - day_factor * congestion_curve.unsqueeze(1) * congestion_amp.unsqueeze(0)
        )

        weekend_bonus = (1.2 * weekend - 0.4 * (1.0 - weekend)) * (0.5 - road_score_norm)
        speed_day = speed_day + weekend_bonus.unsqueeze(0)

        noise = torch.randn((288, num_nodes), generator=gen) * cfg.bank_noise_std
        noise = 0.5 * noise + 0.5 * noise.roll(shifts=1, dims=0)
        speed_day = (speed_day + noise).clamp_min(5.0)
        base_bank[w] = speed_day

    weekday_ids = torch.arange(7, dtype=torch.float32).view(1, 7, 1).expand(1, 7, 288)
    slot_ids = torch.arange(288, dtype=torch.float32).view(1, 1, 288).expand(1, 7, 288)
    weekday_angle = 2 * math.pi * weekday_ids / 7.0
    slot_angle = 2 * math.pi * slot_ids / 288.0

    base_speed_node = free_flow.view(num_nodes, 1, 1).expand(num_nodes, 7, 288)
    speed_norm = base_bank.permute(2, 0, 1) / base_speed_node.clamp_min(1.0)
    congestion_prob = (1.0 - speed_norm).clamp(0.0, 1.0)
    weekend_feat = weekend_flags.view(1, 7, 1).expand(num_nodes, 7, 288)
    volatility_feat = volatility.view(num_nodes, 1, 1).expand(num_nodes, 7, 288)
    degree_feat = degree_norm.view(num_nodes, 1, 1).expand(num_nodes, 7, 288)

    profile_candidates = [
        speed_norm,
        congestion_prob,
        torch.sin(weekday_angle).expand(num_nodes, 7, 288),
        torch.cos(weekday_angle).expand(num_nodes, 7, 288),
        torch.sin(slot_angle).expand(num_nodes, 7, 288),
        torch.cos(slot_angle).expand(num_nodes, 7, 288),
        weekend_feat,
        volatility_feat,
        degree_feat,
    ]
    profile_feat = torch.stack(profile_candidates[: cfg.profile_dim], dim=-1).float()

    aux = {
        "free_flow": free_flow,
        "degree_norm": degree_norm,
        "road_score_norm": road_score_norm,
        "volatility": volatility,
    }
    return base_bank, profile_feat, aux


# -----------------------------
# 生成近期趋势 / 事件 / 监督
# -----------------------------
def _week_index_from_pair(weekday: int, slot: int) -> int:
    return weekday * 288 + slot


def _pair_from_week_index(index: int) -> tuple[int, int]:
    index = index % (7 * 288)
    return index // 288, index % 288


def _gather_week_series(bank: torch.Tensor, start_index: int, length: int) -> torch.Tensor:
    outs = []
    for k in range(length):
        w, s = _pair_from_week_index(start_index + k)
        outs.append(bank[w, s])
    return torch.stack(outs, dim=0)


def build_sample(
    sample_idx: int,
    x_static: torch.Tensor,
    edge_index: torch.Tensor,
    y_base_bank: torch.Tensor,
    profile_feat: torch.Tensor,
    aux: dict[str, torch.Tensor],
    cfg: SimConfig,
) -> dict[str, torch.Tensor]:
    num_nodes = x_static.shape[0]
    gen = torch.Generator().manual_seed(cfg.seed + 1000 + sample_idx)

    degree_norm = aux["degree_norm"]
    road_score_norm = aux["road_score_norm"]

    target_weekday = int(torch.randint(0, 7, (1,), generator=gen).item())
    target_slot = int(torch.randint(0, 288, (1,), generator=gen).item())

    node_drift = torch.randn(num_nodes, generator=gen) * (1.5 + 2.5 * (1.0 - road_score_norm))
    node_drift = smooth_signal(node_drift, edge_index, num_steps=2, alpha=0.72)

    daily_phase = float(torch.rand(1, generator=gen).item()) * 2 * math.pi
    daily_curve = (
        0.65 * torch.sin(torch.linspace(0, 2 * math.pi, 288) + daily_phase)
        + 0.35 * torch.cos(torch.linspace(0, 4 * math.pi, 288) + 0.5 * daily_phase)
    )
    weekly_curve = torch.tensor([0.8, 0.5, 0.2, 0.0, 0.6, -0.3, -0.5], dtype=torch.float32)
    weekly_shift = float(torch.randn(1, generator=gen).item()) * 0.6

    delta_recent_bank = torch.zeros_like(y_base_bank)
    for w in range(7):
        delta_recent_bank[w] = (
            weekly_shift * 2.0
            + daily_curve.view(288, 1) * node_drift.view(1, num_nodes)
            + weekly_curve[w] * (0.8 + 1.5 * (1.0 - road_score_norm)).view(1, num_nodes)
        )

    future_noise = torch.randn(y_base_bank.shape, generator=gen) * cfg.future_noise_std
    y_future_bank = (y_base_bank + delta_recent_bank + future_noise).clamp_min(5.0)

    target_index = _week_index_from_pair(target_weekday, target_slot)
    hist = _gather_week_series(y_future_bank, target_index - cfg.recent_len, cfg.recent_len)
    hist = hist.transpose(0, 1).contiguous()
    obs_noise = torch.randn((num_nodes, cfg.recent_len), generator=gen) * cfg.observation_noise_std
    recent_speed_seq = (hist + obs_noise).unsqueeze(-1).float()

    event_dim = cfg.event_dim
    event_vector = torch.zeros((num_nodes, event_dim), dtype=torch.float32)

    accident = (torch.rand(num_nodes, generator=gen) < cfg.event_prob).float() * torch.rand(num_nodes, generator=gen)
    weather = (torch.rand(num_nodes, generator=gen) < cfg.event_prob * 0.7).float() * torch.rand(num_nodes, generator=gen)
    construction = (torch.rand(num_nodes, generator=gen) < cfg.event_prob * 0.5).float() * torch.rand(num_nodes, generator=gen)
    closure = (torch.rand(num_nodes, generator=gen) < cfg.event_prob * 0.25).float() * torch.rand(num_nodes, generator=gen)

    spill_seed = (accident + 0.7 * weather + 0.9 * construction + 1.2 * closure).clamp(0.0, 1.5)
    spill = smooth_signal(spill_seed, edge_index, num_steps=3, alpha=0.6)
    spill = minmax_01(spill)

    severity = (0.9 * accident + 0.6 * weather + 0.75 * construction + 1.2 * closure).clamp(0.0, 1.5)

    event_candidates = [
        severity,
        accident,
        weather,
        construction,
        closure,
        spill,
        degree_norm,
        road_score_norm,
    ]
    for d in range(min(event_dim, len(event_candidates))):
        event_vector[:, d] = event_candidates[d]
    if event_dim > len(event_candidates):
        extra = torch.randn((num_nodes, event_dim - len(event_candidates)), generator=gen) * 0.1
        extra = smooth_signal(extra, edge_index, num_steps=2, alpha=0.7)
        event_vector[:, len(event_candidates):] = extra

    base_target = y_future_bank[target_weekday, target_slot]
    event_drop = (
        severity * (5.0 + 12.0 * (1.0 - road_score_norm))
        + 4.0 * spill
        + 6.0 * closure
    )
    event_drop = event_drop.clamp_min(0.0)
    y_target_speed = (base_target - event_drop).clamp_min(3.0)

    # 注意：共享量不再在每个样本里重复存一份，避免体积过大
    sample = {
        "recent_speed_seq": recent_speed_seq.float(),
        "target_weekday": torch.tensor(target_weekday, dtype=torch.long),
        "target_slot": torch.tensor(target_slot, dtype=torch.long),
        "event_vector": event_vector.float(),
        "y_future_bank": y_future_bank.float(),
        "y_target_speed": y_target_speed.float(),
        "future_mask": torch.ones_like(y_future_bank, dtype=torch.float32),
        "event_mask": torch.ones_like(y_target_speed, dtype=torch.float32),
    }
    return sample


def flush_chunk(buffer: list[dict[str, torch.Tensor]], split_dir: Path, split: str, shard_idx: int, storage_dtype: torch.dtype) -> str:
    shard_path = split_dir / f"{split}_{shard_idx:05d}.pt"
    payload = cast_tensor_tree(buffer, storage_dtype)
    torch.save(payload, shard_path)
    return shard_path.name


def generate_dataset(cfg: SimConfig, base_dir: Path) -> dict[str, Any]:
    set_seed(cfg.seed)

    x_static, edge_index = load_graph_inputs(cfg, base_dir)
    edge_index = ensure_edge_index_in_range(edge_index, x_static.shape[0])

    y_base_bank, profile_feat, aux = build_base_bank_and_profile(x_static, edge_index, cfg)

    split_sizes = {
        "train": cfg.num_train,
        "val": cfg.num_val,
        "test": cfg.num_test,
    }

    out_root = _resolve_path(cfg.output_dir, base_dir)
    assert out_root is not None
    out_root.mkdir(parents=True, exist_ok=True)

    storage_dtype = get_storage_dtype(cfg.storage_dtype)

    meta = {
        "num_nodes": int(x_static.shape[0]),
        "static_dim": int(x_static.shape[1]),
        "num_edges": int(edge_index.shape[1]),
        "recent_len": cfg.recent_len,
        "profile_dim": cfg.profile_dim,
        "event_dim": cfg.event_dim,
        "splits": split_sizes,
        "seed": cfg.seed,
        "save_chunk_size": cfg.save_chunk_size,
        "storage_dtype": cfg.storage_dtype,
        "format": "sharded_pt_v2",
    }

    shared_reference = {
        "x_static": x_static,
        "edge_index": edge_index,
        "y_base_bank": y_base_bank,
        "profile_feat": profile_feat,
    }
    torch.save(cast_tensor_tree(shared_reference, storage_dtype), out_root / "reference_graph_and_base.pt")

    manifest: dict[str, Any] = {
        "format": "sharded_pt_v2",
        "reference_file": "reference_graph_and_base.pt",
        "storage_dtype": cfg.storage_dtype,
        "save_chunk_size": cfg.save_chunk_size,
        "splits": {},
    }

    for split, size in split_sizes.items():
        split_dir = out_root / split
        split_dir.mkdir(parents=True, exist_ok=True)

        buffer: list[dict[str, torch.Tensor]] = []
        shard_files: list[str] = []
        shard_idx = 0

        iterator = progress(range(size), total=size, desc=f"生成 {split}", enable=cfg.show_progress)
        for i in iterator:
            global_idx = {"train": 0, "val": 100000, "test": 200000}[split] + i
            sample = build_sample(
                sample_idx=global_idx,
                x_static=x_static,
                edge_index=edge_index,
                y_base_bank=y_base_bank,
                profile_feat=profile_feat,
                aux=aux,
                cfg=cfg,
            )
            buffer.append(sample)

            if len(buffer) >= cfg.save_chunk_size:
                shard_name = flush_chunk(buffer, split_dir, split, shard_idx, storage_dtype)
                shard_files.append(shard_name)
                shard_idx += 1
                buffer.clear()
                gc.collect()

        if buffer:
            shard_name = flush_chunk(buffer, split_dir, split, shard_idx, storage_dtype)
            shard_files.append(shard_name)
            buffer.clear()
            gc.collect()

        manifest["splits"][split] = {
            "num_samples": size,
            "num_shards": len(shard_files),
            "shards": shard_files,
        }

    with open(out_root / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    with open(out_root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return {
        **meta,
        "resolved_output_dir": str(out_root),
        "reference_file": str(out_root / "reference_graph_and_base.pt"),
        "manifest_file": str(out_root / "manifest.json"),
    }


# -----------------------------
# 配置解析
# -----------------------------
def load_yaml_config(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise ImportError("缺少 PyYAML，请先安装: pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_config_from_sources(args: argparse.Namespace, base_dir: Path) -> SimConfig:
    cfg = SimConfig()

    if args.config:
        path = _resolve_path(args.config, base_dir)
        if path is None or not path.exists():
            raise FileNotFoundError(f"找不到配置文件: {args.config}")
        raw = load_yaml_config(path)
        for k, v in raw.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    for key, value in vars(args).items():
        if value is None or key == "config":
            continue
        if hasattr(cfg, key):
            setattr(cfg, key, value)

    return cfg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="基于 x_static 和 edge_index 生成 TrafficGNN 所需的模拟数据")
    parser.add_argument("--config", type=str, default=None, help="yaml 配置文件路径")

    parser.add_argument("--graph-bundle-path", type=str, default=None)
    parser.add_argument("--x-static-path", type=str, default=None)
    parser.add_argument("--edge-index-path", type=str, default=None)
    parser.add_argument("--x-static-key", type=str, default=None)
    parser.add_argument("--edge-index-key", type=str, default=None)

    parser.add_argument("--output-dir", type=str, default=None)

    parser.add_argument("--num-train", type=int, default=None)
    parser.add_argument("--num-val", type=int, default=None)
    parser.add_argument("--num-test", type=int, default=None)

    parser.add_argument("--recent-len", type=int, default=None)
    parser.add_argument("--profile-dim", type=int, default=None)
    parser.add_argument("--event-dim", type=int, default=None)

    parser.add_argument("--base-speed-min", type=float, default=None)
    parser.add_argument("--base-speed-max", type=float, default=None)
    parser.add_argument("--observation-noise-std", type=float, default=None)
    parser.add_argument("--bank-noise-std", type=float, default=None)
    parser.add_argument("--future-noise-std", type=float, default=None)
    parser.add_argument("--event-prob", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--show-progress", type=lambda x: str(x).lower() in {"1", "true", "yes", "y"}, default=None)
    parser.add_argument("--save-chunk-size", type=int, default=None)
    parser.add_argument("--storage-dtype", type=str, choices=["float16", "float32"], default=None)
    return parser


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    parser = build_parser()
    args = parser.parse_args()

    cfg = build_config_from_sources(args, base_dir)
    meta = generate_dataset(cfg, base_dir)

    print("模拟数据生成完成。")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
