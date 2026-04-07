from __future__ import annotations

import heapq
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from .map_matching import RoadEdgeIndex


@dataclass
class PlannedRoute:
    route_edge_ids: list[int]
    route_edge_lengths_m: list[float]
    total_length_m: float
    found_path: bool


class EdgeRoutePlanner:
    def __init__(self, adjacency: dict[int, list[int]], edge_lengths_m: dict[int, float]):
        self.adjacency = adjacency
        self.edge_lengths_m = edge_lengths_m

    @classmethod
    def from_road_index(cls, road_index: RoadEdgeIndex) -> "EdgeRoutePlanner":
        adjacency: dict[int, list[int]] = {}
        src, dst = road_index.edge_index.long()
        for s, d in zip(src.tolist(), dst.tolist()):
            adjacency.setdefault(int(s), []).append(int(d))
        edge_lengths_m = {r.edge_id: float(r.length_m) for r in road_index.records}
        return cls(adjacency=adjacency, edge_lengths_m=edge_lengths_m)

    def shortest_path(self, start_edge_id: int, dest_edge_id: int, max_expand: int = 200000) -> PlannedRoute:
        if start_edge_id == dest_edge_id:
            l = float(self.edge_lengths_m.get(start_edge_id, 1.0))
            return PlannedRoute([start_edge_id], [l], l, True)

        pq: list[tuple[float, int]] = [(0.0, int(start_edge_id))]
        dist = {int(start_edge_id): 0.0}
        parent: dict[int, int | None] = {int(start_edge_id): None}
        expand = 0

        while pq and expand < max_expand:
            cur_cost, cur = heapq.heappop(pq)
            expand += 1
            if cur == int(dest_edge_id):
                break
            if cur_cost > dist.get(cur, float("inf")):
                continue
            for nxt in self.adjacency.get(cur, []):
                new_cost = cur_cost + float(self.edge_lengths_m.get(nxt, 1.0))
                if new_cost < dist.get(nxt, float("inf")):
                    dist[nxt] = new_cost
                    parent[nxt] = cur
                    heapq.heappush(pq, (new_cost, nxt))

        if int(dest_edge_id) not in parent:
            # 工程回退：至少返回 [current, dest]
            edges = [int(start_edge_id)]
            lengths = [float(self.edge_lengths_m.get(int(start_edge_id), 1.0))]
            if int(dest_edge_id) != int(start_edge_id):
                edges.append(int(dest_edge_id))
                lengths.append(float(self.edge_lengths_m.get(int(dest_edge_id), 1.0)))
            return PlannedRoute(edges, lengths, sum(lengths), False)

        path = []
        cur = int(dest_edge_id)
        while cur is not None:
            path.append(cur)
            cur = parent.get(cur)
        path.reverse()
        lengths = [float(self.edge_lengths_m.get(eid, 1.0)) for eid in path]
        return PlannedRoute(path, lengths, float(sum(lengths)), True)


__all__ = [
    "PlannedRoute",
    "EdgeRoutePlanner",
]
