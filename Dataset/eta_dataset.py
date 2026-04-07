from __future__ import annotations

import functools
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

try:
    from utils.gps_trip_parser import (
        ParsedTripRecord,
        build_trip_encoder_batch,
        detect_stop_segments,
        parse_real_trip_record,
    )
    from utils.map_matching import MatchResult, RoadEdgeIndex
    from utils.route_planner import EdgeRoutePlanner
except ImportError:
    from gps_trip_parser import (
        ParsedTripRecord,
        build_trip_encoder_batch,
        detect_stop_segments,
        parse_real_trip_record,
    )
    from map_matching import MatchResult, RoadEdgeIndex
    from route_planner import EdgeRoutePlanner


@dataclass
class ETASampleSpec:
    trip_path: str
    sample_time: datetime
    label_eta_minutes: float
    sample_id: str


def infer_destination_coord(record: ParsedTripRecord) -> tuple[float, float]:
    stops = detect_stop_segments(record.points)
    if stops:
        last = stops[-1]
        return float(last.centroid_lat), float(last.centroid_lon)
    if not record.points:
        raise ValueError("轨迹为空，无法推断目的地")
    p = record.points[-1]
    return float(p.lat), float(p.lon)


def discover_trip_paths(trip_dir: str | Path) -> list[Path]:
    p = Path(trip_dir)
    return sorted(x for x in p.glob("*.txt") if x.is_file())


def build_sample_specs_for_trip(
    record: ParsedTripRecord,
    trip_path: str | Path,
    min_elapsed_min: float = 30.0,
    step_min: float = 30.0,
    min_remaining_min: float = 10.0,
) -> list[ETASampleSpec]:
    if not record.points:
        return []
    departure = record.actual_departure_time or record.points[0].gtm
    arrival = record.actual_arrival_time or record.points[-1].gtm
    out: list[ETASampleSpec] = []
    last_take_time: datetime | None = None
    sample_idx = 0
    for p in record.points:
        elapsed_min = (p.gtm - departure).total_seconds() / 60.0
        remain_min = (arrival - p.gtm).total_seconds() / 60.0
        if elapsed_min < min_elapsed_min:
            continue
        if remain_min < min_remaining_min:
            continue
        if last_take_time is not None and (p.gtm - last_take_time).total_seconds() / 60.0 < step_min:
            continue
        out.append(
            ETASampleSpec(
                trip_path=str(trip_path),
                sample_time=p.gtm,
                label_eta_minutes=remain_min,
                sample_id=f"{Path(trip_path).stem}_t{sample_idx:04d}",
            )
        )
        sample_idx += 1
        last_take_time = p.gtm
    return out


@functools.lru_cache(maxsize=1024)
def _cached_parse_trip(path: str) -> ParsedTripRecord:
    return parse_real_trip_record(path)


def build_single_trip_sample(
    trip_path: str | Path,
    current_time: datetime,
    road_index: RoadEdgeIndex,
    route_planner: EdgeRoutePlanner,
    topk_match: int = 5,
    max_route_len: int = 256,
) -> dict[str, Any]:
    record = _cached_parse_trip(str(trip_path))
    destination_coord = infer_destination_coord(record)
    trip_batch = build_trip_encoder_batch(
        trip_path,
        current_time=current_time,
        destination_coord=destination_coord,
    )
    prefix_meta = trip_batch["prefix_meta"]
    prefix_points = prefix_meta["prefix_points"]
    current_match: MatchResult = road_index.match_prefix(prefix_points, topk=topk_match)
    dest_match: MatchResult = road_index.match_point(destination_coord[0], destination_coord[1], topk=1)

    route = route_planner.shortest_path(current_match.edge_id, dest_match.edge_id)
    route_edge_ids = route.route_edge_ids[:max_route_len]
    route_edge_lengths_m = route.route_edge_lengths_m[:max_route_len]
    if not route_edge_ids:
        route_edge_ids = [current_match.edge_id]
        route_edge_lengths_m = [float(road_index.record_by_id[current_match.edge_id].length_m)]

    return {
        "trip_num_feat": trip_batch["trip_num_feat"].float(),
        "trip_cat_feat": trip_batch["trip_cat_feat"].long(),
        "current_weekday": torch.tensor(trip_batch["current_weekday"], dtype=torch.long),
        "current_slot": torch.tensor(trip_batch["current_slot"], dtype=torch.long),
        "current_edge_id": torch.tensor(current_match.edge_id, dtype=torch.long),
        "current_edge_remaining_ratio": torch.tensor(max(1e-3, 1.0 - current_match.offset_ratio), dtype=torch.float32),
        "current_edge_candidate_ids": torch.tensor(current_match.candidate_ids, dtype=torch.long),
        "current_edge_candidate_probs": torch.tensor(current_match.candidate_probs, dtype=torch.float32),
        "route_edge_ids": torch.tensor(route_edge_ids, dtype=torch.long),
        "route_edge_lengths_m": torch.tensor(route_edge_lengths_m, dtype=torch.float32),
        "label_eta_minutes": torch.tensor(
            (record.actual_arrival_time - current_time).total_seconds() / 60.0,
            dtype=torch.float32,
        ),
        "meta": {
            "trip_path": str(trip_path),
            "current_time": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "sample_found_path": bool(route.found_path),
            "destination_coord": destination_coord,
        },
    }


class FinalETADataset(Dataset):
    def __init__(
        self,
        trip_paths: Sequence[str | Path],
        road_bundle_path: str | Path,
        min_elapsed_min: float = 30.0,
        step_min: float = 30.0,
        min_remaining_min: float = 10.0,
        topk_match: int = 5,
        max_route_len: int = 256,
    ):
        self.trip_paths = [str(Path(p)) for p in trip_paths]
        self.road_index = RoadEdgeIndex.from_bundle(road_bundle_path)
        self.route_planner = EdgeRoutePlanner.from_road_index(self.road_index)
        self.topk_match = int(topk_match)
        self.max_route_len = int(max_route_len)
        self.sample_specs: list[ETASampleSpec] = []
        self._cache: dict[str, dict[str, Any]] = {}

        for trip_path in self.trip_paths:
            record = _cached_parse_trip(trip_path)
            self.sample_specs.extend(
                build_sample_specs_for_trip(
                    record=record,
                    trip_path=trip_path,
                    min_elapsed_min=min_elapsed_min,
                    step_min=step_min,
                    min_remaining_min=min_remaining_min,
                )
            )

    def __len__(self) -> int:
        return len(self.sample_specs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        spec = self.sample_specs[index]
        if spec.sample_id in self._cache:
            return self._cache[spec.sample_id]
        sample = build_single_trip_sample(
            trip_path=spec.trip_path,
            current_time=spec.sample_time,
            road_index=self.road_index,
            route_planner=self.route_planner,
            topk_match=self.topk_match,
            max_route_len=self.max_route_len,
        )
        self._cache[spec.sample_id] = sample
        return sample


def final_eta_collate_fn(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("batch 为空")

    bsz = len(batch)
    max_route_len = max(item["route_edge_ids"].numel() for item in batch)
    max_topk = max(item["current_edge_candidate_ids"].numel() for item in batch)

    trip_num_feat = torch.stack([item["trip_num_feat"] for item in batch], dim=0)
    trip_cat_feat = torch.stack([item["trip_cat_feat"] for item in batch], dim=0)
    current_weekday = torch.stack([item["current_weekday"] for item in batch], dim=0)
    current_slot = torch.stack([item["current_slot"] for item in batch], dim=0)
    current_edge_id = torch.stack([item["current_edge_id"] for item in batch], dim=0)
    current_edge_remaining_ratio = torch.stack([item["current_edge_remaining_ratio"] for item in batch], dim=0)
    label_eta_minutes = torch.stack([item["label_eta_minutes"] for item in batch], dim=0)

    route_edge_ids = torch.zeros((bsz, max_route_len), dtype=torch.long)
    route_edge_lengths_m = torch.zeros((bsz, max_route_len), dtype=torch.float32)
    route_mask = torch.zeros((bsz, max_route_len), dtype=torch.bool)

    current_edge_candidate_ids = torch.zeros((bsz, max_topk), dtype=torch.long)
    current_edge_candidate_probs = torch.zeros((bsz, max_topk), dtype=torch.float32)

    metas = []
    for i, item in enumerate(batch):
        rlen = item["route_edge_ids"].numel()
        route_edge_ids[i, :rlen] = item["route_edge_ids"]
        route_edge_lengths_m[i, :rlen] = item["route_edge_lengths_m"]
        route_mask[i, :rlen] = True

        k = item["current_edge_candidate_ids"].numel()
        current_edge_candidate_ids[i, :k] = item["current_edge_candidate_ids"]
        probs = item["current_edge_candidate_probs"]
        probs = probs / probs.sum().clamp_min(1e-6)
        current_edge_candidate_probs[i, :k] = probs
        metas.append(item["meta"])

    return {
        "trip_num_feat": trip_num_feat,
        "trip_cat_feat": trip_cat_feat,
        "current_weekday": current_weekday,
        "current_slot": current_slot,
        "current_edge_id": current_edge_id,
        "current_edge_remaining_ratio": current_edge_remaining_ratio,
        "current_edge_candidate_ids": current_edge_candidate_ids,
        "current_edge_candidate_probs": current_edge_candidate_probs,
        "route_edge_ids": route_edge_ids,
        "route_edge_lengths_m": route_edge_lengths_m,
        "route_mask": route_mask,
        "label_eta_minutes": label_eta_minutes,
        "meta": metas,
    }


__all__ = [
    "ETASampleSpec",
    "FinalETADataset",
    "build_single_trip_sample",
    "discover_trip_paths",
    "final_eta_collate_fn",
    "infer_destination_coord",
]
