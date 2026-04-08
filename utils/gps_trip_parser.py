from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import torch


EARTH_R_M = 6371000.0
TIME_PATTERNS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y/%m/%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y/%m/%d %H:%M",
    "%Y-%m-%d %H:%M",
)


@dataclass(frozen=True)
class TripPoint:
    lat: float
    lon: float
    gtm: datetime


@dataclass(frozen=True)
class StopSegment:
    start_time: datetime
    end_time: datetime
    centroid_lat: float
    centroid_lon: float
    duration_minutes: float


@dataclass(frozen=True)
class ParsedTripRecord:
    points: list[TripPoint]
    actual_departure_time: datetime | None = None
    actual_arrival_time: datetime | None = None


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * EARTH_R_M * math.asin(math.sqrt(max(a, 0.0)))


def _try_parse_datetime(text: str) -> datetime | None:
    raw = text.strip()
    if not raw:
        return None
    raw = raw.replace("Z", "")
    for pattern in TIME_PATTERNS:
        try:
            return datetime.strptime(raw, pattern)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def _try_parse_timestamp(token: str) -> datetime | None:
    token = token.strip()
    if not token:
        return None
    dt = _try_parse_datetime(token)
    if dt is not None:
        return dt
    if re.fullmatch(r"\d{10}", token):
        return datetime.fromtimestamp(int(token))
    if re.fullmatch(r"\d{13}", token):
        return datetime.fromtimestamp(int(token) / 1000.0)
    return None


def _candidate_lat_lon(values: Sequence[float]) -> tuple[float, float] | None:
    if len(values) < 2:
        return None
    for i in range(len(values)):
        for j in range(len(values)):
            if i == j:
                continue
            lat = float(values[i])
            lon = float(values[j])
            if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                return lat, lon
    return None


def _normalize_trip_point(dt: datetime, lat: float, lon: float) -> TripPoint | None:
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None
    return TripPoint(lat=float(lat), lon=float(lon), gtm=dt)


def _parse_json_line(line: str) -> TripPoint | None:
    try:
        obj = json.loads(line)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None

    lat = obj.get("lat", obj.get("latitude"))
    lon = obj.get("lon", obj.get("lng", obj.get("longitude")))
    ts = obj.get("gtm", obj.get("time", obj.get("timestamp", obj.get("datetime"))))
    if lat is None or lon is None or ts is None:
        return None
    dt = _try_parse_timestamp(str(ts))
    if dt is None:
        return None
    try:
        return _normalize_trip_point(dt, float(lat), float(lon))
    except Exception:
        return None


def _parse_delimited_line(line: str) -> TripPoint | None:
    tokens = [tok.strip() for tok in re.split(r"[\t,;| ]+", line.strip()) if tok.strip()]
    if len(tokens) < 3:
        return None

    dt = None
    dt_idx: int | None = None
    skip_two = False
    for idx in range(len(tokens)):
        if idx + 1 < len(tokens):
            joined = f"{tokens[idx]} {tokens[idx + 1]}"
            dt = _try_parse_timestamp(joined)
            if dt is not None:
                dt_idx = idx
                skip_two = True
                break
        dt = _try_parse_timestamp(tokens[idx])
        if dt is not None:
            dt_idx = idx
            break
    if dt is None:
        return None

    numeric_values: list[float] = []
    for idx, token in enumerate(tokens):
        if dt_idx is not None and idx == dt_idx:
            continue
        if skip_two and dt_idx is not None and idx == dt_idx + 1:
            continue
        try:
            numeric_values.append(float(token))
        except ValueError:
            continue

    lat_lon = _candidate_lat_lon(numeric_values)
    if lat_lon is None:
        return None
    return _normalize_trip_point(dt, lat_lon[0], lat_lon[1])


def parse_real_trip_record(path: str | Path) -> ParsedTripRecord:
    trip_path = Path(path)
    points: list[TripPoint] = []

    with trip_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            point = _parse_json_line(line)
            if point is None:
                point = _parse_delimited_line(line)
            if point is not None:
                points.append(point)

    points.sort(key=lambda p: p.gtm)
    if not points:
        raise ValueError(f"无法从轨迹文件解析任何有效点: {trip_path}")

    return ParsedTripRecord(
        points=points,
        actual_departure_time=points[0].gtm,
        actual_arrival_time=points[-1].gtm,
    )


def detect_stop_segments(
    points: Sequence[TripPoint],
    min_stop_minutes: float = 10.0,
    max_radius_m: float = 80.0,
) -> list[StopSegment]:
    if len(points) < 2:
        return []

    segments: list[StopSegment] = []
    start = 0
    while start < len(points) - 1:
        end = start
        while end + 1 < len(points):
            dist = _haversine_m(points[start].lat, points[start].lon, points[end + 1].lat, points[end + 1].lon)
            if dist > max_radius_m:
                break
            end += 1

        duration_min = (points[end].gtm - points[start].gtm).total_seconds() / 60.0
        if end > start and duration_min >= min_stop_minutes:
            seg_points = points[start:end + 1]
            centroid_lat = sum(p.lat for p in seg_points) / len(seg_points)
            centroid_lon = sum(p.lon for p in seg_points) / len(seg_points)
            segments.append(
                StopSegment(
                    start_time=seg_points[0].gtm,
                    end_time=seg_points[-1].gtm,
                    centroid_lat=centroid_lat,
                    centroid_lon=centroid_lon,
                    duration_minutes=duration_min,
                )
            )
            start = end + 1
        else:
            start += 1

    return segments


def _prefix_points(record: ParsedTripRecord, current_time: datetime) -> list[tuple[float, float]]:
    prefix = [(p.lat, p.lon) for p in record.points if p.gtm <= current_time]
    if not prefix:
        prefix = [(record.points[0].lat, record.points[0].lon)]
    return prefix


def _trip_num_features(record: ParsedTripRecord, current_time: datetime, destination_coord: tuple[float, float]) -> torch.Tensor:
    departure = record.actual_departure_time or record.points[0].gtm
    prefix_pts = [p for p in record.points if p.gtm <= current_time]
    if not prefix_pts:
        prefix_pts = [record.points[0]]
    current = prefix_pts[-1]

    elapsed_min = max(0.0, (current_time - departure).total_seconds() / 60.0)
    traveled_m = 0.0
    for a, b in zip(prefix_pts[:-1], prefix_pts[1:]):
        traveled_m += _haversine_m(a.lat, a.lon, b.lat, b.lon)
    remain_m = _haversine_m(current.lat, current.lon, destination_coord[0], destination_coord[1])
    mean_speed_kmh = (traveled_m / 1000.0) / max(elapsed_min / 60.0, 1e-3)
    total_duration_min = max(
        0.0,
        ((record.actual_arrival_time or record.points[-1].gtm) - departure).total_seconds() / 60.0,
    )

    return torch.tensor(
        [
            elapsed_min,
            traveled_m / 1000.0,
            remain_m / 1000.0,
            mean_speed_kmh,
            float(len(prefix_pts)),
            total_duration_min,
        ],
        dtype=torch.float32,
    )


def _trip_cat_features(current_time: datetime) -> torch.Tensor:
    return torch.tensor(
        [
            current_time.weekday(),
            current_time.hour,
            current_time.month - 1,
            int(current_time.weekday() >= 5),
            current_time.timetuple().tm_yday - 1,
        ],
        dtype=torch.long,
    )


def build_trip_encoder_batch(
    trip_path: str | Path,
    current_time: datetime,
    destination_coord: tuple[float, float],
) -> dict[str, object]:
    record = parse_real_trip_record(trip_path)
    return {
        "trip_num_feat": _trip_num_features(record, current_time, destination_coord),
        "trip_cat_feat": _trip_cat_features(current_time),
        "current_weekday": current_time.weekday(),
        "current_slot": current_time.hour * 12 + current_time.minute // 5,
        "prefix_meta": {
            "prefix_points": _prefix_points(record, current_time),
        },
    }


__all__ = [
    "ParsedTripRecord",
    "StopSegment",
    "TripPoint",
    "build_trip_encoder_batch",
    "detect_stop_segments",
    "parse_real_trip_record",
]
