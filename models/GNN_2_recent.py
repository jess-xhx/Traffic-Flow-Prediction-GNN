import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


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


class CalendarQueryEncoder(nn.Module):
    """
    周期查询编码器
    输出 [7, 288, D_c]
    """
    def __init__(self, weekday_emb_dim: int = 8, slot_emb_dim: int = 16, out_dim: int = 32):
        super().__init__()
        self.weekday_emb = nn.Embedding(7, weekday_emb_dim)
        self.slot_emb = nn.Embedding(288, slot_emb_dim)

        cyc_dim = 4   # sin/cos(time_of_day), sin/cos(weekday)
        flag_dim = 1  # is_weekend

        in_dim = weekday_emb_dim + slot_emb_dim + cyc_dim + flag_dim
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, weekday_ids: torch.Tensor, slot_ids: torch.Tensor) -> torch.Tensor:
        """
        weekday_ids: [7, 288]
        slot_ids:    [7, 288]
        """
        w_emb = self.weekday_emb(weekday_ids)     # [7, 288, weekday_emb_dim]
        s_emb = self.slot_emb(slot_ids)           # [7, 288, slot_emb_dim]

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
        return self.proj(feat)  # [7, 288, D_c]


class RecentSequenceEncoder(nn.Module):
    """
    最近速度序列编码器
    输入:
        recent_speed_seq [N, K, 1]
    输出:
        recent_summary   [N, D_r]
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, recent_speed_seq: torch.Tensor) -> torch.Tensor:
        """
        recent_speed_seq: [N, K, 1]
        """
        _, h_n = self.gru(recent_speed_seq)   # h_n: [1, N, D_r]
        recent_summary = h_n.squeeze(0)       # [N, D_r]
        recent_summary = self.proj(recent_summary)
        return recent_summary

class ResidualGraphPropagator(nn.Module):
    """
    对 7*288 个时间片的残差做图传播
    输入:
        delta_feat [7, 288, N, D]
    输出:
        delta_feat [7, 288, N, D]
    """
    def __init__(self, hidden_dim: int, time_chunk_size: int = 6):
        super().__init__()
        self.gnn1 = GATConv(hidden_dim, hidden_dim, heads=2, concat=False)
        self.gnn2 = GATConv(hidden_dim, hidden_dim, heads=1, concat=False)
        self.time_chunk_size = time_chunk_size

    def _forward_chunk(self, chunk_feat: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # chunk_feat: [Cg, N, D]
        Cg, N, D = chunk_feat.shape

        x = chunk_feat.reshape(Cg * N, D)
        batched_edge_index = build_batched_edge_index(edge_index, Cg, N)

        x = F.relu(self.gnn1(x, batched_edge_index))
        x = F.relu(self.gnn2(x, batched_edge_index))
        x = x.view(Cg, N, D)
        return x

    def forward(self, delta_feat: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        W, S, N, D = delta_feat.shape
        G = W * S

        flat = delta_feat.reshape(G, N, D)
        outputs = []

        for start in range(0, G, self.time_chunk_size):
            end = min(start + self.time_chunk_size, G)
            chunk = flat[start:end]
            out_chunk = self._forward_chunk(chunk, edge_index)
            outputs.append(out_chunk)

        x = torch.cat(outputs, dim=0)
        x = x.view(W, S, N, D)
        return x


class RecentResidualBank(nn.Module):
    """
    模块 2：RecentResidualBank

    输入:
        H_base_bank:       [7, 288, N, D]
        recent_speed_seq:  [N, K, 1]
        edge_index:        [2, E]

    输出:
        delta_recent_bank: [7, 288, N, D]
        H_adapted_bank:    [7, 288, N, D]   # optional
        pred_speed_bank:   [7, 288, N]      # optional, 训练监督用
    """
    def __init__(
        self,
        bank_hidden_dim: int = 64,
        recent_hidden_dim: int = 32,
        calendar_hidden_dim: int = 32,
        use_speed_head: bool = True
    ):
        super().__init__()

        self.recent_encoder = RecentSequenceEncoder(recent_hidden_dim)
        self.calendar_encoder = CalendarQueryEncoder(
            weekday_emb_dim=8,
            slot_emb_dim=16,
            out_dim=calendar_hidden_dim
        )

        fusion_in_dim = bank_hidden_dim + recent_hidden_dim + calendar_hidden_dim

        # 生成原始残差
        self.delta_mlp = nn.Sequential(
            nn.Linear(fusion_in_dim, bank_hidden_dim),
            nn.ReLU(),
            nn.Linear(bank_hidden_dim, bank_hidden_dim)
        )

        # 生成门控，控制残差幅度
        self.gate_mlp = nn.Sequential(
            nn.Linear(fusion_in_dim, bank_hidden_dim),
            nn.Sigmoid()
        )

        # 图传播平滑
        # self.graph_propagator = ResidualGraphPropagator(bank_hidden_dim)
        self.graph_propagator = ResidualGraphPropagator(bank_hidden_dim, time_chunk_size=6)

        self.use_speed_head = use_speed_head
        if use_speed_head:
            self.speed_head = nn.Sequential(
                nn.Linear(bank_hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )

    def _build_calendar_ids(self, device: torch.device):
        weekday_ids = torch.arange(7, device=device).unsqueeze(1).repeat(1, 288)  # [7, 288]
        slot_ids = torch.arange(288, device=device).unsqueeze(0).repeat(7, 1)     # [7, 288]
        return weekday_ids, slot_ids

    def build_delta_bank(
        self,
        H_base_bank: torch.Tensor,       # [7, 288, N, D]
        recent_speed_seq: torch.Tensor,  # [N, K, 1]
        edge_index: torch.Tensor,        # [2, E]
        return_full: bool = True,
    ):
        device = H_base_bank.device
        W, S, N, D = H_base_bank.shape

        # 1) 最近序列摘要
        recent_summary = self.recent_encoder(recent_speed_seq)          # [N, D_r]

        # 2) 周期查询编码
        weekday_ids, slot_ids = self._build_calendar_ids(device)
        calendar_emb = self.calendar_encoder(weekday_ids, slot_ids)     # [7, 288, D_c]

        # 3) 广播对齐维度
        recent_expand = recent_summary.unsqueeze(0).unsqueeze(0).expand(W, S, N, -1)     # [7, 288, N, D_r]
        calendar_expand = calendar_emb.unsqueeze(2).expand(W, S, N, -1)                  # [7, 288, N, D_c]

        # 4) 残差生成
        fusion_feat = torch.cat([H_base_bank, recent_expand, calendar_expand], dim=-1)   # [7, 288, N, D+D_r+D_c]

        delta_raw = self.delta_mlp(fusion_feat)   # [7, 288, N, D]
        gate = self.gate_mlp(fusion_feat)         # [7, 288, N, D]

        delta_recent_bank = delta_raw * gate      # [7, 288, N, D]

        # 5) 图传播平滑
        delta_recent_bank = self.graph_propagator(delta_recent_bank, edge_index)          # [7, 288, N, D]

        # 6) 适配后的 bank
        H_adapted_bank = H_base_bank + delta_recent_bank                                  # [7, 288, N, D]

        if not self.use_speed_head:
            return delta_recent_bank, H_adapted_bank

        # 7) 训练时的速度监督头
        pred_speed_bank = self.speed_head(H_adapted_bank).squeeze(-1)                     # [7, 288, N]

        return delta_recent_bank, H_adapted_bank, pred_speed_bank