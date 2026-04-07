import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch.utils.checkpoint import checkpoint


def build_batched_edge_index(edge_index: torch.Tensor, num_graphs: int, num_nodes: int) -> torch.Tensor:
    """
    把同一张路网图复制 num_graphs 份，拼成一个大 batch 图。
    edge_index: [2, E]
    return: [2, num_graphs * E]
    """
    device = edge_index.device
    offsets = torch.arange(num_graphs, device=device).view(num_graphs, 1, 1) * num_nodes
    batched_edge_index = edge_index.unsqueeze(0) + offsets            # [G, 2, E]
    batched_edge_index = batched_edge_index.permute(1, 0, 2)          # [2, G, E]
    batched_edge_index = batched_edge_index.reshape(2, -1)            # [2, G*E]
    return batched_edge_index


class StaticGraphEncoder(nn.Module):
    """
    静态路网特征 -> 静态图嵌入
    输入:  x_static [N, F_s]
    输出:  static_emb [N, D_s]
    """
    def __init__(self, static_dim: int, static_hidden_dim: int):
        super().__init__()
        self.gnn1 = GATConv(static_dim, static_hidden_dim, heads=4, concat=False)
        self.gnn2 = GATConv(static_hidden_dim, static_hidden_dim, heads=1, concat=False)

    def forward(self, x_static: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.gnn1(x_static, edge_index))
        x = F.relu(self.gnn2(x, edge_index))
        return x   # [N, D_s]


class CalendarEncoder(nn.Module):
    """
    周期时间编码器
    输入:
        weekday_ids [7, 288]
        slot_ids    [7, 288]
    输出:
        cal_emb     [7, 288, D_c]
    """
    def __init__(self, weekday_emb_dim: int = 8, slot_emb_dim: int = 16, out_dim: int = 32):
        super().__init__()
        self.weekday_emb = nn.Embedding(7, weekday_emb_dim)
        self.slot_emb = nn.Embedding(288, slot_emb_dim)

        cyc_dim = 4   # sin/cos(一天中的时间), sin/cos(weekday)
        flag_dim = 1  # weekend flag

        in_dim = weekday_emb_dim + slot_emb_dim + cyc_dim + flag_dim
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, weekday_ids: torch.Tensor, slot_ids: torch.Tensor) -> torch.Tensor:
        """
        weekday_ids: [7, 288], 值域 0..6
        slot_ids:    [7, 288], 值域 0..287
        """
        w_emb = self.weekday_emb(weekday_ids)  # [7, 288, weekday_emb_dim]
        s_emb = self.slot_emb(slot_ids)        # [7, 288, slot_emb_dim]

        slot_float = slot_ids.float()
        time_angle = 2.0 * math.pi * slot_float / 288.0
        time_sin = torch.sin(time_angle).unsqueeze(-1)
        time_cos = torch.cos(time_angle).unsqueeze(-1)

        weekday_float = weekday_ids.float()
        week_angle = 2.0 * math.pi * weekday_float / 7.0
        week_sin = torch.sin(week_angle).unsqueeze(-1)
        week_cos = torch.cos(week_angle).unsqueeze(-1)

        cyc_feat = torch.cat([time_sin, time_cos, week_sin, week_cos], dim=-1)  # [7, 288, 4]
        is_weekend = ((weekday_ids == 5) | (weekday_ids == 6)).float().unsqueeze(-1)

        feat = torch.cat([w_emb, s_emb, cyc_feat, is_weekend], dim=-1)
        cal_emb = self.proj(feat)  # [7, 288, D_c]
        return cal_emb


class ProfileEncoder(nn.Module):
    """
    长期周期统计特征编码器
    输入:
        profile_feat [N, 7, 288, F_p]
    输出:
        profile_emb  [7, 288, N, D_p]
    """
    def __init__(self, profile_dim: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(profile_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, profile_feat: torch.Tensor) -> torch.Tensor:
        """
        profile_feat: [N, 7, 288, F_p]
        return:       [7, 288, N, D_p]
        """
        x = self.mlp(profile_feat)                 # [N, 7, 288, D_p]
        x = x.permute(1, 2, 0, 3).contiguous()    # [7, 288, N, D_p]
        return x


class TimeSliceGraphEncoder(nn.Module):
    """
    对 7*288 个时间片逐片做空间图传播。

    注意：
    这里仍然只负责“同一时间片内部”的空间传播，
    不负责时间连续性；时间连续性由后面的 TemporalBankRefiner 显式建模。

    输入:
        fused_feat [7, 288, N, F]
    输出:
        spatial_bank [7, 288, N, D]
    """
    def __init__(self, input_dim: int, hidden_dim: int, time_chunk_size: int = 256):
        super().__init__()
        self.gnn1 = GATConv(input_dim, hidden_dim, heads=4, concat=False)
        self.gnn2 = GATConv(hidden_dim, hidden_dim, heads=1, concat=False)
        self.time_chunk_size = time_chunk_size

    def _forward_chunk(self, chunk_feat: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # chunk_feat: [Cg, N, F]
        Cg, N, Fdim = chunk_feat.shape

        x = chunk_feat.reshape(Cg * N, Fdim)
        batched_edge_index = build_batched_edge_index(edge_index, Cg, N)

        x = F.relu(self.gnn1(x, batched_edge_index))
        x = F.relu(self.gnn2(x, batched_edge_index))
        x = x.view(Cg, N, -1)
        return x

    def forward(self, fused_feat: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        W, S, N, Fdim = fused_feat.shape
        G = W * S

        fused_flat = fused_feat.reshape(G, N, Fdim)   # [2016, N, F]
        outputs = []

        for start in range(0, G, self.time_chunk_size):
            end = min(start + self.time_chunk_size, G)
            chunk = fused_flat[start:end]             # [Cg, N, F]
            out_chunk = self._forward_chunk(chunk, edge_index)
            outputs.append(out_chunk)

        x = torch.cat(outputs, dim=0)                 # [2016, N, D]
        x = x.view(W, S, N, -1)
        return x


class TemporalConvBlock(nn.Module):
    """
    显式时间建模块：在每个路段自己的时间轴上做残差时序卷积。

    输入 / 输出:
        x [N, D, T]

    设计思路：
    - depthwise conv 负责沿时间轴提取局部连续变化
    - pointwise MLP 负责通道混合
    - dilation 让模块同时看到短期和中期的时间依赖
    - 残差连接保证不会破坏原始空间 bank
    """
    def __init__(self, hidden_dim: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.0):
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation

        self.norm1 = nn.GroupNorm(1, hidden_dim)
        self.depthwise = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=hidden_dim,
        )
        self.pointwise1 = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=1)
        self.pointwise2 = nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=1)
        self.norm2 = nn.GroupNorm(1, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        y = self.norm1(x)
        y = self.depthwise(y)
        y = F.gelu(y)

        y = self.pointwise1(y)
        y = F.gelu(y)
        y = self.dropout(y)
        y = self.pointwise2(y)
        y = self.norm2(y)

        return residual + y


class TemporalBankRefiner(nn.Module):
    """
    对 [7, 288, N, D] 的周 bank 进行显式时间连续性建模。

    关键区别：
    - 旧版模块1：每个 (weekday, slot) 主要靠时间标签独立生成
    - 新版模块1：先得到 spatial_bank，再沿 T=2016 的时间轴显式传播

    这样相邻 slot 的状态会互相影响：
    08:05 能读到 08:00/07:55，08:10 也能读到 08:05/08:00。

    为了避免一次性把全部路段 reshape 成 [N, D, T] 导致显存爆炸，
    这里按 node chunk 分块做时间卷积；每块内部仍然是完整的 2016 时间轴。
    """
    def __init__(
        self,
        hidden_dim: int,
        kernel_size: int = 3,
        dilations: tuple[int, ...] = (1, 2, 4),
        dropout: float = 0.0,
        node_chunk_size: int = 128,
    ):
        super().__init__()
        self.node_chunk_size = int(node_chunk_size) if node_chunk_size is not None else 0
        self.use_checkpoint = False
        self.blocks = nn.ModuleList([
            TemporalConvBlock(
                hidden_dim=hidden_dim,
                kernel_size=kernel_size,
                dilation=dilation,
                dropout=dropout,
            )
            for dilation in dilations
        ])

    def set_checkpoint_enabled(self, enabled: bool) -> None:
        self.use_checkpoint = bool(enabled)

    def _forward_chunk(self, bank_chunk: torch.Tensor) -> torch.Tensor:
        # bank_chunk: [W, S, C, D]
        W, S, C, D = bank_chunk.shape
        T = W * S

        x = bank_chunk.permute(2, 3, 0, 1).contiguous().reshape(C, D, T)  # [C, D, T]
        for block in self.blocks:
            if self.use_checkpoint and self.training and torch.is_grad_enabled():
                x = checkpoint(lambda t, m=block: m(t), x, use_reentrant=False)
            else:
                x = block(x)
        x = x.view(C, D, W, S).permute(2, 3, 0, 1).contiguous()           # [W, S, C, D]
        return x

    def forward(self, bank: torch.Tensor) -> torch.Tensor:
        # bank: [W, S, N, D]
        W, S, N, D = bank.shape

        if self.node_chunk_size <= 0 or self.node_chunk_size >= N:
            return self._forward_chunk(bank)

        outputs = []
        for start in range(0, N, self.node_chunk_size):
            end = min(start + self.node_chunk_size, N)
            bank_chunk = bank[:, :, start:end, :]                          # [W, S, C, D]
            out_chunk = self._forward_chunk(bank_chunk)
            outputs.append(out_chunk)

        return torch.cat(outputs, dim=2)                                   # [W, S, N, D]


class BaseWeeklyBank(nn.Module):
    """
    模块 1：BaseWeeklyBank

    相比旧版，这一版把模块1拆成两步：
    1) 空间 bank：每个时间片内部做 GNN 图传播
    2) 时间精炼：在每个路段自己的整周时间轴上做显式时间建模

    输入:
        x_static:     [N, F_s]
        profile_feat: [N, 7, 288, F_p]
        edge_index:   [2, E]

    输出:
        H_base_bank:    [7, 288, N, D]
        pred_speed_bank:[7, 288, N]   (用于训练监督)
    """
    def __init__(
        self,
        static_dim: int,
        profile_dim: int,
        static_hidden_dim: int = 32,
        calendar_hidden_dim: int = 32,
        profile_hidden_dim: int = 32,
        bank_hidden_dim: int = 64,
        use_speed_head: bool = True,
        time_chunk_size: int = 256,
        temporal_kernel_size: int = 3,
        temporal_dropout: float = 0.0,
        temporal_dilations: tuple[int, ...] = (1, 2, 4),
        temporal_node_chunk_size: int = 128,
    ):
        super().__init__()
        self.speed_head_chunk_size = time_chunk_size
        
        self.static_encoder = StaticGraphEncoder(static_dim, static_hidden_dim)
        self.calendar_encoder = CalendarEncoder(
            weekday_emb_dim=8,
            slot_emb_dim=16,
            out_dim=calendar_hidden_dim,
        )
        self.profile_encoder = ProfileEncoder(profile_dim, profile_hidden_dim)

        fused_dim = static_hidden_dim + calendar_hidden_dim + profile_hidden_dim
        self.time_slice_gnn = TimeSliceGraphEncoder(
            input_dim=fused_dim,
            hidden_dim=bank_hidden_dim,
            time_chunk_size=time_chunk_size,
        )
        self.temporal_refiner = TemporalBankRefiner(
            hidden_dim=bank_hidden_dim,
            kernel_size=temporal_kernel_size,
            dilations=temporal_dilations,
            dropout=temporal_dropout,
            node_chunk_size=temporal_node_chunk_size,
        )

        self.use_speed_head = use_speed_head
        if use_speed_head:
            self.speed_head = nn.Sequential(
                nn.Linear(bank_hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

    def _predict_speed_bank_chunked(self, H_bank: torch.Tensor) -> torch.Tensor:
        """
        对 [W, S, N, D] 的 bank 分块做速度头预测，降低显存峰值
        返回: [W, S, N]
        """
        W, S, N, D = H_bank.shape
        flat = H_bank.view(W * S, N, D)   # [2016, N, D]

        chunk_size = max(1, int(self.speed_head_chunk_size))
        pred_chunks = []

        for start in range(0, W * S, chunk_size):
            end = min(start + chunk_size, W * S)
            pred_chunk = self.speed_head(flat[start:end]).squeeze(-1)   # [Fc, N]
            pred_chunks.append(pred_chunk)

        pred_flat = torch.cat(pred_chunks, dim=0)   # [2016, N]
        return pred_flat.view(W, S, N)

    def set_checkpoint_enabled(self, enabled: bool) -> None:
        self.temporal_refiner.set_checkpoint_enabled(enabled)

    def _build_calendar_ids(self, device: torch.device):
        """
        构造固定的 [7, 288] weekday_ids 和 slot_ids
        """
        weekday_ids = torch.arange(7, device=device).unsqueeze(1).repeat(1, 288)   # [7, 288]
        slot_ids = torch.arange(288, device=device).unsqueeze(0).repeat(7, 1)      # [7, 288]
        return weekday_ids, slot_ids

    def build_bank(
        self,
        x_static: torch.Tensor,      # [N, F_s]
        profile_feat: torch.Tensor,  # [N, 7, 288, F_p]
        edge_index: torch.Tensor     # [2, E]
    ):
        device = x_static.device
        N = x_static.size(0)

        # 1) 静态图骨架编码，只和路段有关
        static_emb = self.static_encoder(x_static, edge_index)      # [N, D_s]

        # 2) 周期时间编码，只和 (weekday, slot) 有关
        weekday_ids, slot_ids = self._build_calendar_ids(device)
        calendar_emb = self.calendar_encoder(weekday_ids, slot_ids) # [7, 288, D_c]

        # 3) 长期 profile 编码
        profile_emb = self.profile_encoder(profile_feat)            # [7, 288, N, D_p]

        # 4) 广播拼接三类特征
        static_expand = static_emb.unsqueeze(0).unsqueeze(0).expand(7, 288, N, -1)   # [7, 288, N, D_s]
        calendar_expand = calendar_emb.unsqueeze(2).expand(7, 288, N, -1)             # [7, 288, N, D_c]
        fused_feat = torch.cat([static_expand, calendar_expand, profile_emb], dim=-1) # [7, 288, N, F]

        # 5) 先做“每个时间片内部”的空间图传播
        spatial_bank = self.time_slice_gnn(fused_feat, edge_index)                    # [7, 288, N, D]

        # 6) 再做“整周时间轴”的显式时间连续性建模
        H_base_bank = self.temporal_refiner(spatial_bank)                             # [7, 288, N, D]

        if not self.use_speed_head:
            return H_base_bank

        # 7) 训练时速度监督头
        pred_speed_bank = self._predict_speed_bank_chunked(H_base_bank)                    # [7, 288, N]
        return H_base_bank, pred_speed_bank
