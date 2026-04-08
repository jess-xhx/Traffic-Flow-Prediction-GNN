from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch

from configs.eta_config import ETAEncoderConfig, ETAHeadConfig, RouteTokenConfig, SameCityETAConfig, TripTokenConfig
from configs.gnn_config import load_experiment_config
from models.GNN import TrafficGNNSystem
from models.eta_model import FinalHybridETAModel
from Dataset.eta_dataset import build_single_trip_sample
from utils.map_matching import RoadEdgeIndex
from utils.route_planner import EdgeRoutePlanner


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


def load_checkpoint_state(model: torch.nn.Module, ckpt_path: str | Path) -> dict:
    obj = torch.load(Path(ckpt_path), map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        for key in ("model_state_dict", "model", "state_dict"):
            if key in obj and isinstance(obj[key], dict):
                model.load_state_dict(obj[key], strict=False)
                return obj
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            model.load_state_dict(obj, strict=False)
            return {}
    raise KeyError(f"无法从 checkpoint {ckpt_path} 识别 state_dict")


def main() -> None:
    parser = argparse.ArgumentParser(description="最终版 ETA 单样本推理")
    parser.add_argument("--trip_path", required=True, type=str)
    parser.add_argument("--current_time", required=True, type=str, help="格式: YYYY-mm-dd HH:MM:SS")
    parser.add_argument("--road_bundle_path", required=True, type=str)
    parser.add_argument("--profile_feat_path", required=True, type=str)
    parser.add_argument("--gnn_config", required=True, type=str)
    parser.add_argument("--gnn_checkpoint", required=True, type=str)
    parser.add_argument("--eta_checkpoint", required=True, type=str)
    parser.add_argument("--bank_mode", default="base", choices=["base", "recent", "joint"])
    parser.add_argument("--recent_speed_seq_path", default=None, type=str)
    parser.add_argument("--event_vector_path", default=None, type=str)
    args = parser.parse_args()

    current_time = datetime.strptime(args.current_time, "%Y-%m-%d %H:%M:%S")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    road_index = RoadEdgeIndex.from_bundle(args.road_bundle_path)
    route_planner = EdgeRoutePlanner.from_road_index(road_index)
    sample = build_single_trip_sample(
        trip_path=args.trip_path,
        current_time=current_time,
        road_index=road_index,
        route_planner=route_planner,
    )

    gnn_exp_cfg = load_experiment_config(args.gnn_config)
    gnn_model = TrafficGNNSystem(gnn_exp_cfg.model)
    load_checkpoint_state(gnn_model, args.gnn_checkpoint)

    x_static, edge_index = load_road_bundle_x_edge(args.road_bundle_path)
    profile_feat = load_tensor_any(args.profile_feat_path, ["profile_feat"])
    recent_speed_seq = load_tensor_any(args.recent_speed_seq_path, ["recent_speed_seq"]) if args.recent_speed_seq_path else None
    event_vector = load_tensor_any(args.event_vector_path, ["event_vector"]) if args.event_vector_path else None

    eta_cfg = SameCityETAConfig(
        trip_token=TripTokenConfig(numeric_dim=sample["trip_num_feat"].numel()),
        route_token=RouteTokenConfig(
            static_dim=x_static.size(1),
            bank_hidden_dim=gnn_exp_cfg.model.bank_hidden_dim,
        ),
        encoder=ETAEncoderConfig(),
        head=ETAHeadConfig(predict_uncertainty=True),
        freeze_gnn_backbone=True,
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

    eta_ckpt = load_checkpoint_state(model, args.eta_checkpoint)
    model.eval()

    batch = {
        "trip_num_feat": sample["trip_num_feat"].unsqueeze(0),
        "trip_cat_feat": sample["trip_cat_feat"].unsqueeze(0),
        "current_weekday": sample["current_weekday"].view(1),
        "current_slot": sample["current_slot"].view(1),
        "current_edge_id": sample["current_edge_id"].view(1),
        "current_edge_remaining_ratio": sample["current_edge_remaining_ratio"].view(1),
        "current_edge_candidate_ids": sample["current_edge_candidate_ids"].unsqueeze(0),
        "current_edge_candidate_probs": sample["current_edge_candidate_probs"].unsqueeze(0),
        "route_edge_ids": sample["route_edge_ids"].unsqueeze(0),
        "route_edge_lengths_m": sample["route_edge_lengths_m"].unsqueeze(0),
        "route_mask": torch.ones((1, sample["route_edge_ids"].numel()), dtype=torch.bool),
    }
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

    with torch.no_grad():
        pred = model(batch)

    print("trip_path:", args.trip_path)
    print("current_time:", args.current_time)
    print("route_len:", int(sample["route_edge_ids"].numel()))
    print("eta_minutes:", float(pred["eta_minutes"].item()))
    if "log_sigma" in pred:
        print("log_sigma:", float(pred["log_sigma"].item()))
    print("meta:", sample["meta"])


if __name__ == "__main__":
    main()
