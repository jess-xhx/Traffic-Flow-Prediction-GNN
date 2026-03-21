from __future__ import annotations

from typing import Any, Mapping, Union

import torch
import torch.nn as nn


Tensor = torch.Tensor
Batch = Mapping[str, Any]


def to_device(obj: Any, device: torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_device(v, device) for v in obj)
    return obj


def maybe_squeeze_graph_batch(x: Any) -> Any:
    """
    DataLoader 常见情况: batch_size=1 时会在最前面多一维。
    这里仅在第一维为 1 时去掉它，避免改坏真实张量形状。
    """
    if not torch.is_tensor(x):
        return x
    if x.dim() > 0 and x.size(0) == 1:
        return x.squeeze(0)
    return x


def prepare_batch(batch: Batch, device: torch.device) -> dict[str, Any]:
    batch = to_device(dict(batch), device)
    for k, v in list(batch.items()):
        if torch.is_tensor(v):
            batch[k] = maybe_squeeze_graph_batch(v)
    return batch


def scalar_int(x: Union[int, Tensor]) -> int:
    if isinstance(x, int):
        return x
    x = maybe_squeeze_graph_batch(x)
    if x.numel() != 1:
        raise ValueError(f"期望标量，但拿到 shape={tuple(x.shape)}")
    return int(x.item())


def ensure_bank_layout(y: Tensor, num_nodes: int) -> Tensor:
    """
    把 bank 监督目标统一成 [7, 288, N]。
    支持:
      - [7, 288, N]
      - [N, 7, 288]
      - [1, 7, 288, N]
      - [1, N, 7, 288]
    """
    y = maybe_squeeze_graph_batch(y)
    if y.dim() != 3:
        raise ValueError(f"bank 监督张量应为 3 维，实际为 {tuple(y.shape)}")

    if y.shape[0] == 7 and y.shape[1] == 288:
        return y
    if y.shape[1] == 7 and y.shape[2] == 288 and y.shape[0] == num_nodes:
        return y.permute(1, 2, 0).contiguous()

    raise ValueError(
        f"无法识别 bank 目标形状 {tuple(y.shape)}。支持 [7,288,N] 或 [N,7,288]。"
    )


def ensure_node_vector_layout(y: Tensor, num_nodes: int) -> Tensor:
    """
    把单时刻速度监督统一成 [N]。
    支持 [N], [1, N], [N, 1], [1, N, 1]。
    """
    y = maybe_squeeze_graph_batch(y)
    if y.dim() == 1 and y.shape[0] == num_nodes:
        return y
    if y.dim() == 2:
        if y.shape == (num_nodes, 1):
            return y.squeeze(-1)
        if y.shape[0] == 1 and y.shape[1] == num_nodes:
            return y.squeeze(0)
    raise ValueError(f"无法识别单时刻监督形状 {tuple(y.shape)}，期望 [N] / [N,1] / [1,N]。")


def masked_mae(pred: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
    if mask is None:
        return torch.mean(torch.abs(pred - target))
    mask = mask.to(pred.dtype)
    loss = torch.abs(pred - target) * mask
    denom = mask.sum().clamp_min(1.0)
    return loss.sum() / denom


def freeze_module(module: nn.Module) -> None:
    module.eval()
    for p in module.parameters():
        p.requires_grad = False



def unfreeze_module(module: nn.Module) -> None:
    module.train()
    for p in module.parameters():
        p.requires_grad = True
