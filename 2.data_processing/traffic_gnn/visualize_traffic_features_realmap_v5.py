from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import torch

try:
    from matplotlib import font_manager
except Exception:
    font_manager = None

try:
    from torch_geometric.data import Data  # noqa: F401
except Exception:
    Data = None  # noqa: N816

try:
    from shapely.geometry import LineString, MultiLineString
except Exception as e:
    raise RuntimeError('缺少 shapely，请先安装 shapely') from e


# ---------- 基础工具 ----------
def load_torch(path: Path):
    return torch.load(path, map_location='cpu', weights_only=False)


def to_numpy(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.asarray(x)


def configure_chinese_font():
    """尽量在本机自动选择可用中文字体。"""
    candidates = [
        'Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'Source Han Sans SC',
        'WenQuanYi Zen Hei', 'Arial Unicode MS', 'PingFang SC', 'Heiti SC'
    ]
    chosen = None
    if font_manager is not None:
        installed = {f.name for f in font_manager.fontManager.ttflist}
        for name in candidates:
            if name in installed:
                chosen = name
                break
    if chosen is None:
        # 退化为候选列表，让 matplotlib 再自己尝试。
        plt.rcParams['font.sans-serif'] = candidates
    else:
        plt.rcParams['font.sans-serif'] = [chosen] + candidates
    plt.rcParams['axes.unicode_minus'] = False
    return chosen


def resolve_graph_bundle(args, data_root: Path) -> Path:
    candidates: List[Path] = []
    if args.graph_bundle:
        candidates.append(Path(args.graph_bundle))
    candidates.extend([
        data_root.parent / 'Static_road' / 'baoding_static_road_gnn_dataset.pt',
        data_root.parent.parent / 'Static_road' / 'baoding_static_road_gnn_dataset.pt',
        Path('./baoding_static_road_gnn_dataset.pt'),
    ])
    for p in candidates:
        p = p.resolve()
        if p.exists():
            return p
    raise FileNotFoundError(
        '找不到原始静态路网文件 baoding_static_road_gnn_dataset.pt，请用 --graph-bundle 指定。'
    )


def extract_mapping(graph_bundle):
    if isinstance(graph_bundle, dict) and 'mapping' in graph_bundle:
        return graph_bundle['mapping']
    if hasattr(graph_bundle, 'mapping'):
        return graph_bundle.mapping
    raise KeyError('原始静态路网文件中未找到 mapping / gdf_edges')


def load_manifest_and_reference(data_root: Path):
    manifest_path = data_root / 'manifest.json'
    ref_path = data_root / 'reference_graph_and_base.pt'
    if not manifest_path.exists():
        raise FileNotFoundError(f'未找到 {manifest_path}')
    if not ref_path.exists():
        raise FileNotFoundError(f'未找到 {ref_path}')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    reference = load_torch(ref_path)
    return manifest, reference


def pick_sample(data_root: Path, manifest: dict, split: str, global_index: int):
    split_info = manifest['splits'][split]
    shard_files: Sequence[str] = split_info.get('shard_files') or split_info.get('shards') or []
    if not shard_files:
        raise KeyError('manifest.json 中未找到 shard_files / shards 字段')
    total_num = int(split_info.get('num_samples', -1))
    if total_num >= 0 and (global_index < 0 or global_index >= total_num):
        raise IndexError(f'sample-index 超出范围，{split} 只有 {total_num} 个样本')

    seen = 0
    for shard_name in shard_files:
        shard_path = data_root / split / shard_name
        shard_samples = load_torch(shard_path)
        if not isinstance(shard_samples, list):
            raise TypeError(f'{shard_path} 不是 list 分片格式')
        if global_index < seen + len(shard_samples):
            local_index = global_index - seen
            return shard_name, local_index, shard_samples[local_index]
        seen += len(shard_samples)
    raise RuntimeError('未能在分片中找到目标样本')


def compute_degree(edge_index: np.ndarray, num_nodes: int) -> np.ndarray:
    deg = np.zeros(num_nodes, dtype=np.float32)
    src = edge_index[0].astype(np.int64)
    dst = edge_index[1].astype(np.int64)
    np.add.at(deg, src, 1.0)
    np.add.at(deg, dst, 1.0)
    return deg


def robust_rescale(values: np.ndarray, q_low: float = 2.0, q_high: float = 98.0, eps: float = 1e-8):
    values = values.astype(np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values), 0.0, 1.0
    x = values[finite]
    lo = float(np.percentile(x, q_low))
    hi = float(np.percentile(x, q_high))
    if hi - lo < eps:
        lo = float(np.min(x))
        hi = float(np.max(x))
    if hi - lo < eps:
        return np.ones_like(values), lo, hi + 1.0
    clipped = np.clip(values, lo, hi)
    scaled = (clipped - lo) / (hi - lo + eps)
    return scaled.astype(np.float32), lo, hi


def robust_feature_name(sample):
    tw = int(to_numpy(sample['target_weekday']).reshape(-1)[0])
    ts = int(to_numpy(sample['target_slot']).reshape(-1)[0])
    return tw, ts


def extract_features(sample: dict, reference: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, str, str]:
    y_base_bank = to_numpy(reference['y_base_bank'])  # [7, 288, N]
    recent_speed_seq = to_numpy(sample['recent_speed_seq'])  # [N, L, 1]
    event_vector = to_numpy(sample['event_vector'])  # [N, D]

    target_weekday, target_slot = robust_feature_name(sample)

    long_term = y_base_bank[target_weekday, target_slot, :].astype(np.float32)
    recent_term = recent_speed_seq.mean(axis=(1, 2)).astype(np.float32)
    if event_vector.ndim == 2 and event_vector.shape[1] > 0:
        # 用 L2 范数比单取第 0 维更稳定
        event_term = np.linalg.norm(event_vector, axis=1).astype(np.float32)
    else:
        event_term = event_vector.reshape(-1).astype(np.float32)

    long_title = f'长期周期特征（星期 {target_weekday + 1}，时隙 {target_slot}）'
    recent_title = '近期速度特征（近期序列均值）'
    event_title = '当前事件特征（事件向量强度）'
    return long_term, recent_term, event_term, long_title, recent_title, event_title


def extract_segments(geometry_obj) -> List[np.ndarray]:
    segs: List[np.ndarray] = []
    if geometry_obj is None:
        return segs
    if isinstance(geometry_obj, LineString):
        coords = np.asarray(geometry_obj.coords, dtype=np.float32)
        if len(coords) >= 2:
            for i in range(len(coords) - 1):
                segs.append(coords[i:i+2])
        return segs
    if isinstance(geometry_obj, MultiLineString):
        for line in geometry_obj.geoms:
            segs.extend(extract_segments(line))
        return segs
    if hasattr(geometry_obj, 'geoms'):
        for item in geometry_obj.geoms:
            segs.extend(extract_segments(item))
        return segs
    return segs


def scientific_cmap(kind: str):
    if kind == 'long':
        return plt.get_cmap('cividis')
    if kind == 'recent':
        return plt.get_cmap('turbo')
    if kind == 'event':
        # 黑紫红黄，事件更醒目
        return plt.get_cmap('inferno')
    return plt.get_cmap('viridis')


def build_line_collections(gdf_edges, raw_values: np.ndarray, width_signal: np.ndarray,
                           value_q=(2.0, 98.0), width_q=(10.0, 95.0),
                           min_width=0.7, max_width=3.4):
    backbone_segments: List[np.ndarray] = []
    feature_segments: List[np.ndarray] = []
    feature_values: List[float] = []
    feature_widths: List[float] = []

    value_norm, v_lo, v_hi = robust_rescale(raw_values, value_q[0], value_q[1])
    width_norm, _, _ = robust_rescale(width_signal, width_q[0], width_q[1])
    widths = min_width + (max_width - min_width) * np.power(width_norm, 0.85)

    for idx, geom in enumerate(gdf_edges.geometry):
        segs = extract_segments(geom)
        if not segs:
            continue
        backbone_segments.extend(segs)
        feature_segments.extend(segs)
        feature_values.extend([float(value_norm[idx])] * len(segs))
        feature_widths.extend([float(widths[idx])] * len(segs))

    return (
        backbone_segments,
        feature_segments,
        np.asarray(feature_values, dtype=np.float32),
        np.asarray(feature_widths, dtype=np.float32),
        v_lo,
        v_hi,
    )


def plot_single_map(
    gdf_edges,
    values: np.ndarray,
    width_signal: np.ndarray,
    title: str,
    out_path: Path,
    cmap,
    value_label: str,
    value_q=(2.0, 98.0),
    width_q=(10.0, 95.0),
    skeleton_color='#CFCFCF',
    skeleton_width=0.55,
):
    (
        backbone_segments,
        feature_segments,
        feature_values,
        feature_widths,
        v_lo,
        v_hi,
    ) = build_line_collections(
        gdf_edges, values, width_signal,
        value_q=value_q, width_q=width_q,
        min_width=0.8, max_width=3.6,
    )
    if len(feature_segments) == 0:
        raise RuntimeError('geometry 为空，无法绘图')

    fig, ax = plt.subplots(figsize=(10.2, 8.8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # 底图骨架更淡，避免抢颜色层
    lc_backbone = LineCollection(
        backbone_segments,
        colors=skeleton_color,
        linewidths=skeleton_width,
        alpha=0.9,
        zorder=1,
        capstyle='round',
        joinstyle='round',
    )
    ax.add_collection(lc_backbone)

    # 只给高于低阈值的部分更多存在感，低值也保留但颜色更浅
    norm = Normalize(vmin=0.0, vmax=1.0, clip=True)
    lc_feature = LineCollection(
        feature_segments,
        array=feature_values,
        cmap=cmap,
        norm=norm,
        linewidths=feature_widths,
        alpha=0.98,
        zorder=2,
        capstyle='round',
        joinstyle='round',
    )
    ax.add_collection(lc_feature)

    ax.autoscale()
    ax.margins(0.01)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=18, pad=14, fontweight='semibold')

    cbar = fig.colorbar(lc_feature, ax=ax, fraction=0.035, pad=0.02)
    cbar.ax.set_ylabel(f'{value_label}（分位拉伸后）', rotation=90, fontsize=11)
    cbar.ax.tick_params(labelsize=9)
    # 色条显示原始数值范围，更好解释
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    tick_vals = [v_lo, v_lo + 0.25 * (v_hi - v_lo), v_lo + 0.5 * (v_hi - v_lo), v_lo + 0.75 * (v_hi - v_lo), v_hi]
    cbar.set_ticklabels([f'{x:.2f}' for x in tick_vals])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=320, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='真实道路形状交通特征可视化（中文+增强对比版）')
    parser.add_argument('--data-root', type=str, required=True, help='mock_traffic_output 目录')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--sample-index', type=int, default=0, help='split 内全局样本编号')
    parser.add_argument('--size-by', type=str, default='event', choices=['event', 'degree'], help='线宽映射依据')
    parser.add_argument('--graph-bundle', type=str, default=None, help='原始 baoding_static_road_gnn_dataset.pt 路径')
    parser.add_argument('--output-dir', type=str, default=None, help='输出目录')
    parser.add_argument('--value-qlow', type=float, default=2.0, help='颜色映射下分位')
    parser.add_argument('--value-qhigh', type=float, default=98.0, help='颜色映射上分位')
    args = parser.parse_args()

    chosen_font = configure_chinese_font()

    data_root = Path(args.data_root).resolve()
    manifest, reference = load_manifest_and_reference(data_root)
    _, _, sample = pick_sample(data_root, manifest, args.split, args.sample_index)

    graph_bundle_path = resolve_graph_bundle(args, data_root)
    graph_bundle = load_torch(graph_bundle_path)
    gdf_edges = extract_mapping(graph_bundle)

    long_term, recent_term, event_term, long_title, recent_title, event_title = extract_features(sample, reference)

    edge_index = to_numpy(reference['edge_index'])
    num_edges = len(gdf_edges)
    degree = compute_degree(edge_index, num_edges)
    width_signal = event_term if args.size_by == 'event' else degree

    output_dir = Path(args.output_dir).resolve() if args.output_dir else (data_root / 'visualizations_realmap_v5')
    prefix = f'{args.split}_sample_{args.sample_index:05d}'

    q = (args.value_qlow, args.value_qhigh)
    plot_single_map(
        gdf_edges, long_term, width_signal, long_title,
        output_dir / f'{prefix}_长期特征.png',
        cmap=scientific_cmap('long'),
        value_label='长期速度值',
        value_q=q,
    )
    plot_single_map(
        gdf_edges, recent_term, width_signal, recent_title,
        output_dir / f'{prefix}_近期特征.png',
        cmap=scientific_cmap('recent'),
        value_label='近期速度均值',
        value_q=q,
    )
    plot_single_map(
        gdf_edges, event_term, width_signal, event_title,
        output_dir / f'{prefix}_事件特征.png',
        cmap=scientific_cmap('event'),
        value_label='事件强度',
        value_q=(5.0, 99.0),
    )

    print('已完成可视化导出:')
    print(f'  中文字体: {chosen_font or "未显式命中，已启用候选字体列表"}')
    print(f'  原始路网: {graph_bundle_path}')
    print(f'  输出目录: {output_dir}')
    print(f'  样本编号: {args.split}/{args.sample_index}')


if __name__ == '__main__':
    main()
