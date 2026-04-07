from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise ImportError("需要安装 PyYAML 才能使用 --config 读取 yaml 配置") from exc

try:
    from configs.eta_config import ETAEncoderConfig, ETAHeadConfig, RouteTokenConfig, SameCityETAConfig, TripTokenConfig
    from configs.gnn_config import load_experiment_config
    from models.GNN import TrafficGNNSystem
    from models.eta_model_final import FinalHybridETAModel
    from eta_dataset_final import FinalETADataset, discover_trip_paths, final_eta_collate_fn
    from utils.eta_loss import ETALoss
except ImportError:
    from eta_config import ETAEncoderConfig, ETAHeadConfig, RouteTokenConfig, SameCityETAConfig, TripTokenConfig
    from gnn_config import load_experiment_config
    from GNN import TrafficGNNSystem
    from eta_model_final import FinalHybridETAModel
    from eta_dataset_final import FinalETADataset, discover_trip_paths, final_eta_collate_fn
    from eta_loss import ETALoss


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_tensor_any(path: str | Path, candidate_keys: list[str]) -> torch.Tensor:
    obj = torch.load(Path(path), map_location="cpu", weights_only=False)
    if isinstance(obj, torch.Tensor):
        return obj
    if hasattr(obj, "x") and "x" in candidate_keys:
        return obj.x
    if isinstance(obj, dict):
        for k in candidate_keys:
            if k in obj and torch.is_tensor(obj[k]):
                return obj[k]
    raise KeyError(f"{path} 中找不到候选键 {candidate_keys}")


def load_road_bundle_x_edge(path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    obj = torch.load(Path(path), map_location="cpu", weights_only=False)
    if hasattr(obj, "x") and hasattr(obj, "edge_index"):
        return obj.x, obj.edge_index
    if isinstance(obj, dict):
        x = obj.get("x") or obj.get("x_static")
        edge_index = obj.get("edge_index")
        if x is not None and edge_index is not None:
            return x, edge_index
    raise KeyError("road bundle 必须包含 x/x_static 和 edge_index")


def load_checkpoint_state(model: torch.nn.Module, ckpt_path: str | Path) -> None:
    obj = torch.load(Path(ckpt_path), map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        for key in ("model_state_dict", "model", "state_dict"):
            if key in obj and isinstance(obj[key], dict):
                model.load_state_dict(obj[key], strict=False)
                return
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            model.load_state_dict(obj, strict=False)
            return
    raise KeyError(f"无法从 checkpoint {ckpt_path} 识别 state_dict")


def split_paths(paths: list[Path], val_ratio: float, seed: int) -> tuple[list[Path], list[Path]]:
    rng = random.Random(seed)
    paths = list(paths)
    rng.shuffle(paths)
    n_val = max(1, int(len(paths) * val_ratio)) if len(paths) >= 2 else 0
    return paths[n_val:], paths[:n_val]


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, criterion: ETALoss, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    count = 0
    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        pred = model(batch)
        loss_dict = criterion(pred, batch["label_eta_minutes"])
        total_loss += float(loss_dict["loss"].item())
        total_mae += float(loss_dict["mae_minutes"].item())
        count += 1
    denom = max(count, 1)
    return {"loss": total_loss / denom, "mae_minutes": total_mae / denom}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="最终版 ETA 训练脚本（支持 --config yaml）")
    parser.add_argument("--config", default=None, type=str, help="yaml 配置文件路径，例如 configs/eta_train.yaml")

    parser.add_argument("--trip_dir", default=None, type=str)
    parser.add_argument("--road_bundle_path", default=None, type=str)
    parser.add_argument("--profile_feat_path", default=None, type=str)
    parser.add_argument("--gnn_config", default=None, type=str)
    parser.add_argument("--gnn_checkpoint", default=None, type=str)

    parser.add_argument("--bank_mode", default=None, choices=["base", "recent", "joint"])
    parser.add_argument("--recent_speed_seq_path", default=None, type=str)
    parser.add_argument("--event_vector_path", default=None, type=str)

    parser.add_argument("--epochs", default=None, type=int)
    parser.add_argument("--batch_size", default=None, type=int)
    parser.add_argument("--lr", default=None, type=float)
    parser.add_argument("--weight_decay", default=None, type=float)
    parser.add_argument("--grad_clip", default=None, type=float)
    parser.add_argument("--val_ratio", default=None, type=float)
    parser.add_argument("--num_workers", default=None, type=int)

    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--device", default=None, type=str)
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument("--experiment_name", default=None, type=str)

    parser.add_argument("--min_elapsed_min", default=None, type=float)
    parser.add_argument("--step_min", default=None, type=float)
    parser.add_argument("--min_remaining_min", default=None, type=float)
    parser.add_argument("--topk_match", default=None, type=int)
    parser.add_argument("--max_route_len", default=None, type=int)

    parser.add_argument("--freeze_gnn_backbone", action="store_true")
    parser.add_argument("--d_model", default=None, type=int)
    parser.add_argument("--nhead", default=None, type=int)
    parser.add_argument("--num_layers", default=None, type=int)
    parser.add_argument("--predict_uncertainty", action="store_true")
    return parser


def load_yaml_config(config_path: str | Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"配置文件必须是 dict 结构: {config_path}")
    return data


def cfg_get(cfg: dict[str, Any], section: str, key: str, default: Any = None) -> Any:
    sec = cfg.get(section, {})
    if not isinstance(sec, dict):
        return default
    return sec.get(key, default)


def resolve_args(args: argparse.Namespace, cfg: dict[str, Any]) -> argparse.Namespace:
    resolved = argparse.Namespace()

    # paths
    resolved.trip_dir = args.trip_dir or cfg_get(cfg, "paths", "trip_dir")
    resolved.road_bundle_path = args.road_bundle_path or cfg_get(cfg, "paths", "road_bundle_path")
    resolved.profile_feat_path = args.profile_feat_path or cfg_get(cfg, "paths", "profile_feat_path")
    resolved.gnn_config = args.gnn_config or cfg_get(cfg, "paths", "gnn_config")
    resolved.gnn_checkpoint = args.gnn_checkpoint or cfg_get(cfg, "paths", "gnn_checkpoint")
    resolved.recent_speed_seq_path = args.recent_speed_seq_path or cfg_get(cfg, "paths", "recent_speed_seq_path")
    resolved.event_vector_path = args.event_vector_path or cfg_get(cfg, "paths", "event_vector_path")

    # model
    resolved.bank_mode = args.bank_mode or cfg_get(cfg, "model", "bank_mode", "base")
    resolved.freeze_gnn_backbone = bool(args.freeze_gnn_backbone or cfg_get(cfg, "model", "freeze_gnn_backbone", False))
    resolved.d_model = args.d_model if args.d_model is not None else cfg_get(cfg, "model", "d_model", 128)
    resolved.nhead = args.nhead if args.nhead is not None else cfg_get(cfg, "model", "nhead", 8)
    resolved.num_layers = args.num_layers if args.num_layers is not None else cfg_get(cfg, "model", "num_layers", 4)
    resolved.predict_uncertainty = bool(args.predict_uncertainty or cfg_get(cfg, "model", "predict_uncertainty", False))

    # data
    resolved.min_elapsed_min = args.min_elapsed_min if args.min_elapsed_min is not None else cfg_get(cfg, "data", "min_elapsed_min", 30.0)
    resolved.step_min = args.step_min if args.step_min is not None else cfg_get(cfg, "data", "step_min", 30.0)
    resolved.min_remaining_min = args.min_remaining_min if args.min_remaining_min is not None else cfg_get(cfg, "data", "min_remaining_min", 10.0)
    resolved.topk_match = args.topk_match if args.topk_match is not None else cfg_get(cfg, "data", "topk_match", 5)
    resolved.max_route_len = args.max_route_len if args.max_route_len is not None else cfg_get(cfg, "data", "max_route_len", 256)
    resolved.batch_size = args.batch_size if args.batch_size is not None else cfg_get(cfg, "data", "batch_size", 8)
    resolved.num_workers = args.num_workers if args.num_workers is not None else cfg_get(cfg, "data", "num_workers", 0)

    # train
    resolved.epochs = args.epochs if args.epochs is not None else cfg_get(cfg, "train", "epochs", 10)
    resolved.lr = args.lr if args.lr is not None else cfg_get(cfg, "train", "lr", 1e-4)
    resolved.weight_decay = args.weight_decay if args.weight_decay is not None else cfg_get(cfg, "train", "weight_decay", 1e-5)
    resolved.grad_clip = args.grad_clip if args.grad_clip is not None else cfg_get(cfg, "train", "grad_clip", 5.0)
    resolved.val_ratio = args.val_ratio if args.val_ratio is not None else cfg_get(cfg, "train", "val_ratio", 0.15)

    # runtime
    resolved.seed = args.seed if args.seed is not None else cfg_get(cfg, "runtime", "seed", 42)
    resolved.device = args.device or cfg_get(cfg, "runtime", "device", "auto")
    base_output_dir = args.output_dir or cfg_get(cfg, "runtime", "output_dir", "./log")
    experiment_name = args.experiment_name or cfg_get(cfg, "runtime", "experiment_name", "samecity_eta_final_exp01")
    resolved.output_dir = str(Path(base_output_dir) / experiment_name)
    resolved.experiment_name = experiment_name
    resolved.config = args.config
    return resolved


def validate_required_args(args: argparse.Namespace) -> None:
    required = [
        "trip_dir",
        "road_bundle_path",
        "profile_feat_path",
        "gnn_config",
        "gnn_checkpoint",
    ]
    missing = [name for name in required if getattr(args, name, None) in (None, "")]
    if missing:
        raise ValueError(f"缺少必要参数: {missing}")


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def main() -> None:
    parser = build_parser()
    raw_args = parser.parse_args()
    cfg = load_yaml_config(raw_args.config)
    args = resolve_args(raw_args, cfg)
    validate_required_args(args)

    set_seed(args.seed)
    device = resolve_device(args.device)

    trip_paths = discover_trip_paths(args.trip_dir)
    if not trip_paths:
        raise FileNotFoundError(f"{args.trip_dir} 下没有 .txt 轨迹文件")
    train_paths, val_paths = split_paths(trip_paths, args.val_ratio, args.seed)
    if not train_paths:
        raise RuntimeError("训练集为空，请减少 val_ratio 或增加样本文件")

    train_ds = FinalETADataset(
        trip_paths=train_paths,
        road_bundle_path=args.road_bundle_path,
        min_elapsed_min=args.min_elapsed_min,
        step_min=args.step_min,
        min_remaining_min=args.min_remaining_min,
        topk_match=args.topk_match,
        max_route_len=args.max_route_len,
    )
    val_ds = FinalETADataset(
        trip_paths=val_paths if val_paths else train_paths[:1],
        road_bundle_path=args.road_bundle_path,
        min_elapsed_min=args.min_elapsed_min,
        step_min=args.step_min,
        min_remaining_min=args.min_remaining_min,
        topk_match=args.topk_match,
        max_route_len=args.max_route_len,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=final_eta_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=final_eta_collate_fn,
    )

    gnn_exp_cfg = load_experiment_config(args.gnn_config)
    gnn_model = TrafficGNNSystem(gnn_exp_cfg.model)
    load_checkpoint_state(gnn_model, args.gnn_checkpoint)

    x_static, edge_index = load_road_bundle_x_edge(args.road_bundle_path)
    profile_feat = load_tensor_any(args.profile_feat_path, ["profile_feat"])
    recent_speed_seq = None
    event_vector = None
    if args.recent_speed_seq_path:
        recent_speed_seq = load_tensor_any(args.recent_speed_seq_path, ["recent_speed_seq"])
    if args.event_vector_path:
        event_vector = load_tensor_any(args.event_vector_path, ["event_vector"])

    sample0 = train_ds[0]
    eta_cfg = SameCityETAConfig(
        trip_token=TripTokenConfig(numeric_dim=sample0["trip_num_feat"].numel()),
        route_token=RouteTokenConfig(
            static_dim=x_static.size(1),
            bank_hidden_dim=gnn_exp_cfg.model.bank_hidden_dim,
        ),
        encoder=ETAEncoderConfig(
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            max_route_len=args.max_route_len,
        ),
        head=ETAHeadConfig(
            input_dim=args.d_model,
            hidden_dim=args.d_model,
            predict_uncertainty=args.predict_uncertainty,
        ),
        freeze_gnn_backbone=args.freeze_gnn_backbone,
    )

    model = FinalHybridETAModel(
        cfg=eta_cfg,
        gnn_backbone=gnn_model,
        x_static=x_static,
        profile_feat=profile_feat,
        edge_index=edge_index,
        bank_mode=args.bank_mode,
        default_recent_speed_seq=recent_speed_seq,
        default_event_vector=event_vector,
    ).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    criterion = ETALoss(use_uncertainty=args.predict_uncertainty)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_mae = float("inf")

    used_config = {
        "paths": {
            "trip_dir": args.trip_dir,
            "road_bundle_path": args.road_bundle_path,
            "profile_feat_path": args.profile_feat_path,
            "gnn_config": args.gnn_config,
            "gnn_checkpoint": args.gnn_checkpoint,
            "recent_speed_seq_path": args.recent_speed_seq_path,
            "event_vector_path": args.event_vector_path,
        },
        "model": {
            "bank_mode": args.bank_mode,
            "freeze_gnn_backbone": args.freeze_gnn_backbone,
            "d_model": args.d_model,
            "nhead": args.nhead,
            "num_layers": args.num_layers,
            "predict_uncertainty": args.predict_uncertainty,
        },
        "data": {
            "min_elapsed_min": args.min_elapsed_min,
            "step_min": args.step_min,
            "min_remaining_min": args.min_remaining_min,
            "topk_match": args.topk_match,
            "max_route_len": args.max_route_len,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
        },
        "train": {
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "val_ratio": args.val_ratio,
        },
        "runtime": {
            "seed": args.seed,
            "device": str(device),
            "output_dir": str(out_dir.parent),
            "experiment_name": args.experiment_name,
            "resolved_output_dir": str(out_dir),
        },
    }

    with (out_dir / "used_config.json").open("w", encoding="utf-8") as f:
        json.dump(used_config, f, ensure_ascii=False, indent=2)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_mae = 0.0
        count = 0

        for batch in train_loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch)
            loss_dict = criterion(pred, batch["label_eta_minutes"])
            loss = loss_dict["loss"]
            loss.backward()
            if args.grad_clip is not None and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
            optimizer.step()

            total_loss += float(loss.item())
            total_mae += float(loss_dict["mae_minutes"].item())
            count += 1

        train_metrics = {
            "loss": total_loss / max(count, 1),
            "mae_minutes": total_mae / max(count, 1),
        }
        val_metrics = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.4f} train_mae={train_metrics['mae_minutes']:.3f} | "
            f"val_loss={val_metrics['loss']:.4f} val_mae={val_metrics['mae_minutes']:.3f}"
        )

        ckpt = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "eta_cfg": eta_cfg.__dict__,
            "bank_mode": args.bank_mode,
            "used_config": used_config,
        }
        torch.save(ckpt, out_dir / "last.pt")
        if val_metrics["mae_minutes"] < best_mae:
            best_mae = val_metrics["mae_minutes"]
            torch.save(ckpt, out_dir / "best.pt")

    summary = {
        "train_files": [str(p) for p in train_paths],
        "val_files": [str(p) for p in val_paths],
        "num_train_samples": len(train_ds),
        "num_val_samples": len(val_ds),
        "best_val_mae_minutes": best_mae,
        "resolved_output_dir": str(out_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
