import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv


class EventEncoder(nn.Module):
    """
    路段级事件特征编码器
    输入:
        event_vector [N, F_e]
    输出:
        event_emb    [N, D]
    """
    def __init__(self, event_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(event_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, event_vector: torch.Tensor) -> torch.Tensor:
        return self.mlp(event_vector)   # [N, D]


class EventDiffusion(nn.Module):
    """
    在图上扩散事件扰动
    输入:
        event_emb [N, D]
    输出:
        diffused_event_emb [N, D]
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gnn1 = GCNConv(hidden_dim, hidden_dim)
        self.gnn2 = GATConv(hidden_dim, hidden_dim, heads=1, concat=False)

    def forward(self, event_emb: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.gnn1(event_emb, edge_index))
        x = F.relu(self.gnn2(x, edge_index))
        return x   # [N, D]


class GateFusion(nn.Module):
    """
    门控融合器
    输入:
        adapted_state   [N, D]
        diffused_event  [N, D]
    输出:
        gate            [N, D]
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, adapted_state: torch.Tensor, diffused_event: torch.Tensor) -> torch.Tensor:
        fusion = torch.cat([adapted_state, diffused_event], dim=-1)  # [N, 2D]
        gate = self.gate_mlp(fusion)                                 # [N, D]
        return gate


class EventResidualDecoder(nn.Module):
    """
    事件残差生成器
    输入:
        adapted_state   [N, D]
        diffused_event  [N, D]
    输出:
        delta_event_raw [N, D]
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.residual_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, adapted_state: torch.Tensor, diffused_event: torch.Tensor) -> torch.Tensor:
        fusion = torch.cat([adapted_state, diffused_event], dim=-1)  # [N, 2D]
        delta_raw = self.residual_mlp(fusion)                        # [N, D]
        return delta_raw


class EventResidualInjector(nn.Module):
    """
    模块 3：EventResidualInjector

    输入:
        H_adapted_t: [N, D]
        event_vector:[N, F_e]
        edge_index:  [2, E]

    输出:
        delta_event_t: [N, D]
        H_final_t:     [N, D]      # optional
        pred_speed_t:  [N]         # optional, 训练监督用
    """
    def __init__(
        self,
        hidden_dim: int = 64,
        event_dim: int = 8,
        use_speed_head: bool = True
    ):
        super().__init__()

        self.event_encoder = EventEncoder(event_dim, hidden_dim)
        self.event_diffusion = EventDiffusion(hidden_dim)
        self.gate_fusion = GateFusion(hidden_dim)
        self.residual_decoder = EventResidualDecoder(hidden_dim)

        self.use_speed_head = use_speed_head
        if use_speed_head:
            self.speed_head = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )

    def inject(
        self,
        H_adapted_t: torch.Tensor,   # [N, D]
        event_vector: torch.Tensor,  # [N, F_e]
        edge_index: torch.Tensor     # [2, E]
    ):
        # 1) 编码事件
        event_emb = self.event_encoder(event_vector)                 # [N, D]

        # 2) 图扩散
        diffused_event = self.event_diffusion(event_emb, edge_index) # [N, D]

        # 3) 门控
        gate = self.gate_fusion(H_adapted_t, diffused_event)         # [N, D]

        # 4) 原始残差
        delta_raw = self.residual_decoder(H_adapted_t, diffused_event)  # [N, D]

        # 5) 最终事件残差
        delta_event_t = gate * delta_raw                             # [N, D]

        # 6) 最终状态
        H_final_t = H_adapted_t + delta_event_t                      # [N, D]

        if not self.use_speed_head:
            return delta_event_t, H_final_t

        # 7) 当前时刻速度监督头
        pred_speed_t = self.speed_head(H_final_t).squeeze(-1)        # [N]

        return delta_event_t, H_final_t, pred_speed_t