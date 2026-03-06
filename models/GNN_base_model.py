import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv


# ==========================================
# 第一层：静态拓扑层 
# OpenStreetMap (OSM) 原始数据 → 路网静态特征 baoding_static_road_gnn_dataset.pt
# ==========================================



# ==========================================
# 第二层：常态规律建模 
# 利用 GNN 和时间插件学习交通流的周期性
# ==========================================
class TimeEncoder(nn.Module):
    def __init__(self, time_emb_dim=16, day_emb_dim=16):
        super(TimeEncoder, self).__init__()
        # 0-287 表示一天24小时每5分钟一个切片 (24 * 60 / 5 = 288)
        self.time_of_day_emb = nn.Embedding(288, time_emb_dim)
        # 0-6 表示周一到周日
        self.day_of_week_emb = nn.Embedding(7, day_emb_dim)
        
    def forward(self, time_idx, day_idx):
        # time_idx: [Batch] 或 [Batch, N]
        t_emb = self.time_of_day_emb(time_idx)
        d_emb = self.day_of_week_emb(day_idx)
        return torch.cat([t_emb, d_emb], dim=-1) # [..., time_emb_dim + day_emb_dim]


class BaseLayer(nn.Module):
    def __init__(self, static_dim, hist_seq_len, rnn_hidden_dim, gnn_hidden_dim, time_emb_dim=32):
        super(BaseLayer, self).__init__()
        self.hist_seq_len = hist_seq_len
        self.rnn_hidden_dim = rnn_hidden_dim
        
        # 1. 时间编码器
        self.time_encoder = TimeEncoder(time_emb_dim // 2, time_emb_dim // 2)
        
        # 2. 序列特征提取 (处理 T 个历史时刻的车速/流量)
        # 输入维度: 1 (假如只输入速度) -> 输出维度: rnn_hidden_dim
        self.temporal_encoder = nn.GRU(input_size=1, hidden_size=rnn_hidden_dim, batch_first=True)
        
        # 3. 时空融合 GNN (融合静态特征、时间特征、历史动态特征)
        gnn_input_dim = static_dim + time_emb_dim + rnn_hidden_dim
        # 采用 GAT (图注意力网络)，能自适应学习不同相邻路段的流量影响权重
        self.gnn1 = GATConv(gnn_input_dim, gnn_hidden_dim, heads=4, concat=False)
        self.gnn2 = GATConv(gnn_hidden_dim, gnn_hidden_dim, heads=1, concat=False)

    def forward(self, x_static, hist_speed, time_idx, day_idx, edge_index):
        """
        x_static: [N, static_dim] 静态路网特征
        hist_speed:[N, T, 1] 过去 T 个时间步的速度序列
        time_idx/day_idx: [N] 当前预测目标时刻的时间戳索引
        edge_index: [2, E] 你的对偶图邻接矩阵
        """
        N = x_static.size(0)
        
        # 1. 时间特征编码
        time_feat = self.time_encoder(time_idx, day_idx) # [N, time_emb_dim]
        
        # 2. 历史速度时序建模 (T -> 1)
        # hist_speed: [N, T, 1]
        _, h_n = self.temporal_encoder(hist_speed) 
        temporal_feat = h_n.squeeze(0) # 取 GRU 最后一个 hidden state: [N, rnn_hidden_dim]
        
        # 3. 拼接所有常态特征
        base_input = torch.cat([x_static, time_feat, temporal_feat], dim=-1) # [N, combined_dim]
        
        # 4. GNN 空间信息传递
        base_out = F.relu(self.gnn1(base_input, edge_index))
        base_out = F.relu(self.gnn2(base_out, edge_index)) # [N, gnn_hidden_dim]
        
        return base_out


# ==========================================
# 第三层：偶发事件注入
# 负责处理突发状况（如事故、暴雨），对 Base Layer 的预测进行修正
# ==========================================
class InjectionLayer(nn.Module):
    def __init__(self, event_dim, gnn_hidden_dim):
        super(InjectionLayer, self).__init__()
        
        # 将稀疏的事件向量 (如 [0,1,0] 天气, [1,0] 事故) 映射到高维
        self.event_proj = nn.Linear(event_dim, gnn_hidden_dim)
        
        # 利用 GCN 或 GAT 进行“拥堵阻力”的拓扑扩散
        # 事故发生在一个节点，GNN 传递 1-2 层就能扩散到上下游路段
        self.diffusion_gnn = GCNConv(gnn_hidden_dim, gnn_hidden_dim)
        
        # 修正门控机制 (Gate Mechanism) - 决定多大程度上用异常事件修正常态预测
        self.gate = nn.Sequential(
            nn.Linear(gnn_hidden_dim * 2, gnn_hidden_dim),
            nn.Sigmoid()
        )
        
        # 融合层
        self.fusion_mlp = nn.Linear(gnn_hidden_dim * 2, gnn_hidden_dim)

    def forward(self, base_feat, event_vector, edge_index):
        """
        base_feat: [N, gnn_hidden_dim] 第二层常态预测的隐向量
        event_vector: [N, event_dim] 突发事件向量 (大部分节点可能是全0，仅事故节点有值)
        """
        # 1. 突发事件特征映射
        event_feat = F.relu(self.event_proj(event_vector)) #[N, gnn_hidden_dim]
        
        # 2. 突发事件沿路网拓扑扩散 (模拟拥堵蔓延)
        diffused_event = F.relu(self.diffusion_gnn(event_feat, edge_index))
        
        # 3. 门控融合 (让模型自适应学习：没有事件的地方保留 Base 预测，有事件的地方大幅修正)
        gate_value = self.gate(torch.cat([base_feat, diffused_event], dim=-1))
        
        # 结合常态与修正项
        corrected_feat = base_feat * (1 - gate_value) + diffused_event * gate_value
        
        # 通过多层感知机进一步融合
        out_feat = F.relu(self.fusion_mlp(torch.cat([base_feat, corrected_feat], dim=-1)))
        
        return out_feat


# ==========================================
# 总装配：三层流预测模型
# ==========================================
class TrafficFlowPredictor(nn.Module):
    def __init__(self, static_dim, hist_seq_len, rnn_hidden_dim, gnn_hidden_dim, event_dim):
        super(TrafficFlowPredictor, self).__init__()
        
        # Layer 2: 常态规律基座
        self.base_layer = BaseLayer(static_dim, hist_seq_len, rnn_hidden_dim, gnn_hidden_dim)
        
        # Layer 3: 异常事件注入与修正
        self.injection_layer = InjectionLayer(event_dim, gnn_hidden_dim)
        
        # 输出层：预测 T+1 时刻的速度 (标量)
        self.predict_head = nn.Sequential(
            nn.Linear(gnn_hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # 预测速度值
        )

    def forward(self, x_static, hist_speed, time_idx, day_idx, event_vector, edge_index):
        
        # Step 1: Base Layer 获取基于历史和周期的常态表示
        base_feat = self.base_layer(x_static, hist_speed, time_idx, day_idx, edge_index)
        
        # Step 2: Injection Layer 注入突发事件并进行拓扑扩散修正
        corrected_feat = self.injection_layer(base_feat, event_vector, edge_index)
        
        # Step 3: 输出最终预测的 T+1 速度
        pred_speed = self.predict_head(corrected_feat) # [N, 1]
        
        return pred_speed.squeeze(-1) # [N]