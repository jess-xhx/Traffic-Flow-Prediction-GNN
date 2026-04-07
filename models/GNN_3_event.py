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
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, event_vector: torch.Tensor) -> torch.Tensor:
        return self.mlp(event_vector)  # [N, D]


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
        return x  # [N, D]


class GateFusion(nn.Module):
    """
    当前时刻门控融合器
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
            nn.Sigmoid(),
        )

    def forward(self, adapted_state: torch.Tensor, diffused_event: torch.Tensor) -> torch.Tensor:
        fusion = torch.cat([adapted_state, diffused_event], dim=-1)  # [N, 2D]
        gate = self.gate_mlp(fusion)                                 # [N, D]
        return gate


class EventResidualDecoder(nn.Module):
    """
    当前时刻原始事件残差生成器
    输入:
        adapted_state   [N, D]
        diffused_event  [N, D]
    输出:
        delta_raw       [N, D]
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.residual_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, adapted_state: torch.Tensor, diffused_event: torch.Tensor) -> torch.Tensor:
        fusion = torch.cat([adapted_state, diffused_event], dim=-1)  # [N, 2D]
        delta_raw = self.residual_mlp(fusion)                        # [N, D]
        return delta_raw


class RelativeTimeEncoder(nn.Module):
    """
    相对时间编码器
    输入:
        delta_slots [F]，表示距离事件发生已过去多少个 5-min slot
    输出:
        time_emb    [F, D]
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, delta_slots: torch.Tensor) -> torch.Tensor:
        # delta_slots: [F], float32
        day_pos = delta_slots / 288.0
        week_pos = delta_slots / 2016.0

        day_angle = 2.0 * torch.pi * day_pos
        week_angle = 2.0 * torch.pi * week_pos

        feat = torch.stack(
            [
                day_pos,
                week_pos,
                torch.sin(day_angle),
                torch.cos(day_angle),
                torch.sin(week_angle),
                torch.cos(week_angle),
            ],
            dim=-1,
        )  # [F, 6]
        return self.proj(feat)  # [F, D]


class DecayRateHead(nn.Module):
    """
    预测每个节点的衰减率 rho
    输入:
        [H0, diffused_event, delta_seed] -> [N, 3D]
    输出:
        rho [N, 1]
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h0: torch.Tensor, diffused_event: torch.Tensor, delta_seed: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h0, diffused_event, delta_seed], dim=-1)  # [N, 3D]
        rho = F.softplus(self.mlp(x)) + 1e-4                     # [N, 1]
        return rho


class FutureGate(nn.Module):
    """
    对未来每个 slot 生成门控
    输入:
        H_future[h], diffused_event, delta_seed, time_emb[h]
    输出:
        gate_future[h] [N, D]
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        h_future: torch.Tensor,       # [F, N, D]
        diffused_event: torch.Tensor, # [N, D]
        delta_seed: torch.Tensor,     # [N, D]
        time_emb: torch.Tensor,       # [F, D]
    ) -> torch.Tensor:
        Fh, N, D = h_future.shape
        event_expand = diffused_event.unsqueeze(0).expand(Fh, N, D)
        seed_expand = delta_seed.unsqueeze(0).expand(Fh, N, D)
        time_expand = time_emb.unsqueeze(1).expand(Fh, N, D)

        fusion = torch.cat([h_future, event_expand, seed_expand, time_expand], dim=-1)  # [F, N, 4D]
        gate = self.mlp(fusion)  # [F, N, D]
        return gate


class EventResidualInjector(nn.Module):
    """
    模块 3：只修正当前周 bank 的后半段（从事件时刻到本周结束）

    输入:
        H_adapted_bank : [7, 288, N, D]
        event_weekday  : int / scalar tensor
        event_slot     : int / scalar tensor
        event_vector   : [N, F_e]
        edge_index     : [2, E]

    输出:
        delta_event_bank : [7, 288, N, D]   (事件残差，仅后半段非零)
        H_final_bank     : [7, 288, N, D]
        pred_speed_bank  : [7, 288, N]
        event_bank_mask  : [7, 288, N]      (事件时刻之前为 0，之后为 1)
    """
    def __init__(
        self,
        hidden_dim: int = 64,
        event_dim: int = 8,
        use_speed_head: bool = True,
        future_chunk_size: int = 128,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.future_chunk_size = future_chunk_size

        self.event_encoder = EventEncoder(event_dim, hidden_dim)
        self.event_diffusion = EventDiffusion(hidden_dim)
        self.gate_fusion = GateFusion(hidden_dim)
        self.residual_decoder = EventResidualDecoder(hidden_dim)

        self.time_encoder = RelativeTimeEncoder(hidden_dim)
        self.decay_rate_head = DecayRateHead(hidden_dim)
        self.future_gate = FutureGate(hidden_dim)

        self.use_speed_head = use_speed_head
        if use_speed_head:
            self.speed_head = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

    @staticmethod
    def _to_flat_index(weekday: int, slot: int, slots_per_day: int) -> int:
        return weekday * slots_per_day + slot

    
    def _predict_speed_bank_chunked(self, H_bank: torch.Tensor) -> torch.Tensor:
        """
        对 [W, S, N, D] 的 bank 分块做速度头预测，降低显存峰值
        返回: [W, S, N]
        """
        W, S, N, D = H_bank.shape
        flat = H_bank.view(W * S, N, D)   # [2016, N, D]

        chunk_size = max(1, int(self.future_chunk_size))
        pred_chunks = []

        for start in range(0, W * S, chunk_size):
            end = min(start + chunk_size, W * S)
            pred_chunk = self.speed_head(flat[start:end]).squeeze(-1)   # [Fc, N]
            pred_chunks.append(pred_chunk)

        pred_flat = torch.cat(pred_chunks, dim=0)   # [2016, N]
        return pred_flat.view(W, S, N)

    def inject_seed(
        self,
        H_adapted_t: torch.Tensor,   # [N, D]
        event_vector: torch.Tensor,  # [N, F_e]
        edge_index: torch.Tensor,    # [2, E]
    ):
        # 1) 编码事件
        event_emb = self.event_encoder(event_vector)                    # [N, D]

        # 2) 图扩散
        diffused_event = self.event_diffusion(event_emb, edge_index)    # [N, D]

        # 3) 当前时刻门控
        gate_t = self.gate_fusion(H_adapted_t, diffused_event)          # [N, D]

        # 4) 当前时刻原始残差
        delta_raw_t = self.residual_decoder(H_adapted_t, diffused_event)  # [N, D]

        # 5) 事件种子残差
        delta_seed = gate_t * delta_raw_t                               # [N, D]

        return delta_seed, diffused_event

    def inject_week(
        self,
        H_adapted_bank: torch.Tensor,  # [7, 288, N, D]
        event_weekday: int,
        event_slot: int,
        event_vector: torch.Tensor,    # [N, F_e]
        edge_index: torch.Tensor,      # [2, E]
    ):
        W, S, N, D = H_adapted_bank.shape
        assert D == self.hidden_dim, f'hidden dim mismatch: got {D}, expect {self.hidden_dim}'

        device = H_adapted_bank.device
        dtype = H_adapted_bank.dtype

        flat_len = W * S
        start_idx = self._to_flat_index(int(event_weekday), int(event_slot), S)

        H_flat = H_adapted_bank.view(flat_len, N, D)   # [2016, N, D]
        H0 = H_flat[start_idx]                         # [N, D]

        # 1) 当前事件种子
        delta_seed, diffused_event = self.inject_seed(H0, event_vector, edge_index)

        # 2) 当前周剩余 future 状态
        H_future = H_flat[start_idx:]
        future_len = H_future.size(0)

        # 3) 相对时间编码
        delta_slots_fp32 = torch.arange(future_len, device=device, dtype=torch.float32)
        time_emb = self.time_encoder(delta_slots_fp32).to(dtype=dtype)

        # 4) 每个节点的衰减率
        rho = self.decay_rate_head(H0, diffused_event, delta_seed).to(dtype=dtype)

        # 5) 把相对 slot 差换成“天”为单位
        delta_days = (delta_slots_fp32 / 288.0).to(dtype=dtype).view(future_len, 1, 1)

        # 6) 先分配整周事件残差张量，默认全 0
        delta_flat = torch.zeros_like(H_flat)

        # 7) 按 future_chunk_size 分块计算未来时段事件影响
        chunk_size = max(1, int(self.future_chunk_size))
        for local_start in range(0, future_len, chunk_size):
            local_end = min(local_start + chunk_size, future_len)

            global_start = start_idx + local_start
            global_end = start_idx + local_end

            H_future_chunk = H_flat[global_start:global_end]
            time_emb_chunk = time_emb[local_start:local_end]
            delta_days_chunk = delta_days[local_start:local_end]

            alpha_chunk = torch.exp(-delta_days_chunk * rho.unsqueeze(0))
            gate_chunk = self.future_gate(
                H_future_chunk,
                diffused_event,
                delta_seed,
                time_emb_chunk,
            )

            delta_chunk = alpha_chunk * gate_chunk * delta_seed.unsqueeze(0)
            delta_flat[global_start:global_end] = delta_chunk

        delta_event_bank = delta_flat.view(W, S, N, D)
        H_final_bank = H_adapted_bank + delta_event_bank

        # 8) 事件 mask：事件前为 0，事件后为 1
        event_mask_flat = torch.zeros(flat_len, N, device=device, dtype=dtype)
        event_mask_flat[start_idx:] = 1.0
        event_bank_mask = event_mask_flat.view(W, S, N)

        if not self.use_speed_head:
            return {
                'delta_seed': delta_seed,
                'diffused_event': diffused_event,
                'delta_event_bank': delta_event_bank,
                'H_final_bank': H_final_bank,
                'event_bank_mask': event_bank_mask,
                'pred_speed_bank': None,
            }

        pred_speed_bank = self._predict_speed_bank_chunked(H_final_bank)

        return {
            'delta_seed': delta_seed,
            'diffused_event': diffused_event,
            'delta_event_bank': delta_event_bank,
            'H_final_bank': H_final_bank,
            'pred_speed_bank': pred_speed_bank,
            'event_bank_mask': event_bank_mask,
        }