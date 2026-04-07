from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch


EARTH_R_M = 6371000.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * EARTH_R_M * math.asin(math.sqrt(max(a, 0.0)))


def _local_xy_m(lat: float, lon: float, ref_lat: float, ref_lon: float) -> tuple[float, float]:
    x = math.radians(lon - ref_lon) * EARTH_R_M * math.cos(math.radians(ref_lat))
    y = math.radians(lat - ref_lat) * EARTH_R_M
    return x, y


def _project_point_to_polyline_m(
    lat: float,
    lon: float,
    coords: Sequence[tuple[float, float]],
) -> tuple[float, float, tuple[float, float]]:
    """
    返回:
      min_dist_m,
      offset_ratio(沿线 0~1),
      projected_point(lat, lon)
    coords 采用 (lon, lat) 顺序.
    """
    if len(coords) == 1:
        c_lon, c_lat = coords[0]
        return haversine_m(lat, lon, c_lat, c_lon), 0.0, (c_lat, c_lon)

    seg_lengths: list[float] = []
    total_len = 0.0
    for (lon1, lat1), (lon2, lat2) in zip(coords[:-1], coords[1:]):
        seg = haversine_m(lat1, lon1, lat2, lon2)
        seg_lengths.append(seg)
        total_len += seg
    total_len = max(total_len, 1e-6)

    best_dist = float("inf")
    best_along = 0.0
    best_pt = (lat, lon)
    prefix = 0.0

    for seg_len, (lon1, lat1), (lon2, lat2) in zip(seg_lengths, coords[:-1], coords[1:]):
        x1, y1 = _local_xy_m(lat1, lon1, lat, lon)
        x2, y2 = _local_xy_m(lat2, lon2, lat, lon)
        vx, vy = x2 - x1, y2 - y1
        seg_norm2 = max(vx * vx + vy * vy, 1e-6)
        t = max(0.0, min(1.0, -(x1 * vx + y1 * vy) / seg_norm2))
        proj_x = x1 + t * vx
        proj_y = y1 + t * vy
        dist = math.sqrt(proj_x * proj_x + proj_y * proj_y)
        if dist < best_dist:
            proj_lon = lon1 + t * (lon2 - lon1)
            proj_lat = lat1 + t * (lat2 - lat1)
            best_dist = dist
            best_along = prefix + t * seg_len
            best_pt = (proj_lat, proj_lon)
        prefix += seg_len

    return best_dist, best_along / total_len, best_pt


def _extract_coords_from_geometry(geometry: Any) -> list[tuple[float, float]]:
    if geometry is None:
        return []
    # shapely LineString
    if hasattr(geometry, "coords"):
        return [(float(x), float(y)) for x, y in geometry.coords]
    # geo-interface
    if hasattr(geometry, "__geo_interface__"):
        geo = geometry.__geo_interface__
        if geo.get("type") == "LineString":
            return [(float(x), float(y)) for x, y in geo["coordinates"]]
    # list-like
    try:
        pts = list(geometry)
        if pts and isinstance(pts[0], (list, tuple)) and len(pts[0]) >= 2:
            return [(float(p[0]), float(p[1])) for p in pts]
    except Exception:
        pass
    return []


@dataclass
class EdgeRecord:
    edge_id: int
    u: int
    v: int
    length_m: float
    coords: list[tuple[float, float]]
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float
    center_lat: float
    center_lon: float


@dataclass
class MatchResult:
    edge_id: int
    offset_ratio: float
    projected_lat: float
    projected_lon: float
    distance_m: float
    candidate_ids: list[int]
    candidate_probs: list[float]


class RoadEdgeIndex:
    def __init__(self, x_static: torch.Tensor, edge_index: torch.Tensor, records: list[EdgeRecord]):
        self.x_static = x_static.float().cpu()
        self.edge_index = edge_index.long().cpu()
        self.records = records
        self.record_by_id = {r.edge_id: r for r in records}

    @property
    def num_edges(self) -> int:
        return len(self.records)

    @classmethod
    def from_bundle(cls, bundle_path: str | Path) -> "RoadEdgeIndex":
        bundle = torch.load(Path(bundle_path), map_location="cpu", weights_only=False)
        if hasattr(bundle, "x") and hasattr(bundle, "edge_index"):
            x_static = bundle.x
            edge_index = bundle.edge_index
            mapping = getattr(bundle, "mapping", None)
        elif isinstance(bundle, dict):
            x_static = bundle.get("x") or bundle.get("x_static")
            edge_index = bundle.get("edge_index")
            mapping = bundle.get("mapping")
        else:
            raise TypeError(f"无法识别 road bundle 类型: {type(bundle)}")

        if x_static is None or edge_index is None or mapping is None:
            raise KeyError("road bundle 必须至少包含 x/x_static、edge_index、mapping")

        records: list[EdgeRecord] = []
        iter_rows = mapping.iterrows() if hasattr(mapping, "iterrows") else enumerate(mapping)
        for idx, row in iter_rows:
            if hasattr(row, "to_dict"):
                row = row.to_dict()
            edge_id = int(row.get("edge_id", idx))
            geom = row.get("geometry")
            coords = _extract_coords_from_geometry(geom)
            if not coords:
                continue
            lons = [p[0] for p in coords]
            lats = [p[1] for p in coords]
            length_m = float(row.get("length", row.get("length_m", 0.0)) or 0.0)
            if length_m <= 0:
                length_m = sum(
                    haversine_m(lat1, lon1, lat2, lon2)
                    for (lon1, lat1), (lon2, lat2) in zip(coords[:-1], coords[1:])
                )
            records.append(
                EdgeRecord(
                    edge_id=edge_id,
                    u=int(row.get("u", -1)),
                    v=int(row.get("v", -1)),
                    length_m=length_m,
                    coords=coords,
                    min_lat=min(lats),
                    max_lat=max(lats),
                    min_lon=min(lons),
                    max_lon=max(lons),
                    center_lat=sum(lats) / len(lats),
                    center_lon=sum(lons) / len(lons),
                )
            )
        if not records:
            raise RuntimeError("未能从 mapping 中提取任何有效路段几何")
        return cls(x_static=x_static, edge_index=edge_index, records=records)

    def _bbox_candidates(self, lat: float, lon: float, radius_m: float) -> list[EdgeRecord]:
        lat_delta = radius_m / 111000.0
        lon_delta = radius_m / max(111000.0 * math.cos(math.radians(lat)), 1.0)
        out = [
            r for r in self.records
            if not (
                r.max_lat < lat - lat_delta or r.min_lat > lat + lat_delta
                or r.max_lon < lon - lon_delta or r.min_lon > lon + lon_delta
            )
        ]
        if out:
            return out
        # 粗回退：取中心点最近的一批
        return sorted(self.records, key=lambda r: haversine_m(lat, lon, r.center_lat, r.center_lon))[:200]

    def point_candidates(
        self,
        lat: float,
        lon: float,
        topk: int = 5,
        radius_m: float = 120.0,
    ) -> list[dict[str, float]]:
        scored = []
        for r in self._bbox_candidates(lat, lon, radius_m=radius_m):
            dist_m, offset_ratio, (p_lat, p_lon) = _project_point_to_polyline_m(lat, lon, r.coords)
            scored.append(
                {
                    "edge_id": float(r.edge_id),
                    "distance_m": dist_m,
                    "offset_ratio": offset_ratio,
                    "projected_lat": p_lat,
                    "projected_lon": p_lon,
                    "length_m": r.length_m,
                }
            )
        scored.sort(key=lambda x: x["distance_m"])
        return scored[:topk]

    @staticmethod
    def _softmax(scores: Sequence[float]) -> list[float]:
        if not scores:
            return []
        m = max(scores)
        exps = [math.exp(s - m) for s in scores]
        total = max(sum(exps), 1e-12)
        return [x / total for x in exps]

    def match_point(
        self,
        lat: float,
        lon: float,
        topk: int = 5,
        radius_m: float = 120.0,
    ) -> MatchResult:
        cands = self.point_candidates(lat, lon, topk=topk, radius_m=radius_m)
        if not cands:
            raise RuntimeError("point_candidates 为空，无法匹配当前 GPS 点")
        scores = [-(c["distance_m"] / 25.0) for c in cands]
        probs = self._softmax(scores)
        best = cands[0]
        return MatchResult(
            edge_id=int(best["edge_id"]),
            offset_ratio=float(best["offset_ratio"]),
            projected_lat=float(best["projected_lat"]),
            projected_lon=float(best["projected_lon"]),
            distance_m=float(best["distance_m"]),
            candidate_ids=[int(c["edge_id"]) for c in cands],
            candidate_probs=[float(p) for p in probs],
        )

    def match_prefix(
        self,
        prefix_points: Sequence[Any],
        lookback_points: int = 6,
        topk: int = 5,
        radius_m: float = 120.0,
    ) -> MatchResult:
        if not prefix_points:
            raise ValueError("prefix_points 为空")
        tail = list(prefix_points[-lookback_points:])
        candidate_score: dict[int, float] = {}
        candidate_last_detail: dict[int, dict[str, float]] = {}
        for idx, p in enumerate(tail):
            weight = 0.6 + 0.4 * (idx + 1) / len(tail)
            cands = self.point_candidates(float(p.lat), float(p.lon), topk=topk, radius_m=radius_m)
            for c in cands:
                edge_id = int(c["edge_id"])
                score = weight * math.exp(-float(c["distance_m"]) / 30.0)
                candidate_score[edge_id] = candidate_score.get(edge_id, 0.0) + score
                if idx == len(tail) - 1:
                    candidate_last_detail[edge_id] = c
        if not candidate_score:
            last = tail[-1]
            return self.match_point(float(last.lat), float(last.lon), topk=topk, radius_m=radius_m)

        ranked = sorted(candidate_score.items(), key=lambda kv: kv[1], reverse=True)[:topk]
        ids = [eid for eid, _ in ranked]
        probs = self._softmax([s for _, s in ranked])
        best_id = ids[0]
        detail = candidate_last_detail.get(best_id)
        if detail is None:
            last = tail[-1]
            direct = self.match_point(float(last.lat), float(last.lon), topk=topk, radius_m=radius_m)
            detail = {
                "offset_ratio": direct.offset_ratio,
                "projected_lat": direct.projected_lat,
                "projected_lon": direct.projected_lon,
                "distance_m": direct.distance_m,
            }
        return MatchResult(
            edge_id=best_id,
            offset_ratio=float(detail["offset_ratio"]),
            projected_lat=float(detail["projected_lat"]),
            projected_lon=float(detail["projected_lon"]),
            distance_m=float(detail["distance_m"]),
            candidate_ids=ids,
            candidate_probs=probs,
        )


__all__ = [
    "EdgeRecord",
    "MatchResult",
    "RoadEdgeIndex",
    "haversine_m",
]
