import os
import torch
from torch_geometric.data import Data
from tqdm import tqdm

def generate_mock_dataset(static_data_path, save_path, num_samples=1500, hist_seq_len=12):
    """
    生成伪造的动态路网数据集
    num_samples: 生成多少个时间切片的数据 (比如 1500 个样本，用来训练和验证)
    hist_seq_len: 模型需要过去多少个时间步的数据 (12个5分钟 = 1小时)
    """
    print("1. 加载保定市静态路网骨架...")
    static_data = torch.load(static_data_path,weights_only=False)
    
    # 获取真实的节点（路段）数量和静态特征维度
    num_nodes = static_data.x.size(0)
    static_dim = static_data.x.size(1)
    edge_index = static_data.edge_index
    
    print(f"   - 节点数量: {num_nodes}")
    print(f"   - 静态特征维度: {static_dim}")
    print(f"   - 边数量: {edge_index.size(1)}")

    print(f"\n2. 开始生成 {num_samples} 个时空样本...")
    dataset_list =[]
    
    for i in tqdm(range(num_samples)):
        # -- 模拟动态特征 --
        
        # 1. 历史车速序列 [N, T, 1] (假设基础限速 60，均速在 30-50 波动)
        hist_speed = torch.randn((num_nodes, hist_seq_len, 1)) * 5 + 40 
        
        # 2. 时间编码 (0-287的时间片, 0-6的星期几)
        time_idx = torch.full((num_nodes,), i % 288, dtype=torch.long)
        day_idx = torch.full((num_nodes,), (i // 288) % 7, dtype=torch.long)
        
        # 3. 突发事件向量[N, 3] (假设维度为3：天气, 事故, 拥堵)
        event_vector = torch.zeros((num_nodes, 3))
        # 随机让 20% 的路段发生事故
        accident_mask = torch.rand(num_nodes) < 0.2
        event_vector[accident_mask, 1] = 1.0 
        
        # 如果发生事故，历史最后 3 个时间步速度暴跌至 5-15 km/h
        hist_speed[accident_mask, -3:, 0] = torch.randn((accident_mask.sum(), 3)) * 2 + 10
        
        # -- 模拟未来预测目标 (Label) --
        
        # 4. Y (T+1 时刻的真实速度)
        # 逻辑：下一时刻的速度基本等于历史最后时刻的速度 + 微小噪音
        y_true = hist_speed[:, -1, 0] + torch.randn(num_nodes) * 2 
        # 事故节点速度依然很低
        y_true[accident_mask] = torch.randn(accident_mask.sum()) * 2 + 10 
        # 保证速度不为负数
        y_true = torch.clamp(y_true, min=5.0, max=80.0)

        # -- 组装成 PyG 的 Data 对象 --
        # 注意：每一份 Data 都共享同一个 static_data.x 和 edge_index
        data = Data(
            x_static=static_data.x, 
            edge_index=edge_index,
            hist_speed=hist_speed,
            time_idx=time_idx,
            day_idx=day_idx,
            event_vector=event_vector,
            y=y_true
        )
        dataset_list.append(data)

    print("\n3. 保存伪造数据集...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(dataset_list, save_path)
    print(f"   ✅ 保存成功: {save_path}")

if __name__ == "__main__":

    STATIC_PATH = r"E:\vscode项目文件\Traffic_Flow_Prediction\data\processed\baoding_static_road_gnn_dataset.pt"

    SAVE_PATH = r"E:\vscode项目文件\Traffic_Flow_Prediction\data\processed\mock_dynamic_dataset.pt"
    
    generate_mock_dataset(STATIC_PATH, SAVE_PATH)