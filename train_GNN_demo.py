import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import numpy as np
from tqdm import tqdm
import time
import random
import os

# 导入你之前创建的三层模型 (假设你把代码保存在 models/traffic_flow_model.py)
from models.GNN_base_model import TrafficFlowPredictor 

# 固定随机种子
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

def train():

    # 获取当前时间戳
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    # 设置保存路径
    output_file = f"E:/vscode项目文件/Traffic_Flow_Prediction/log/training_losses_{timestamp}.txt"  # 指定路径并使用时间戳

    # --- 1. 配置参数 ---
    DATA_PATH = r"E:\vscode项目文件\Traffic_Flow_Prediction\data\processed\mock_dynamic_dataset.pt"
    BATCH_SIZE = 8
    EPOCHS = 20
    LEARNING_RATE = 0.001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用计算设备: {DEVICE}")

    # --- 2. 加载数据 ---
    print("加载数据集...")
    dataset = torch.load(DATA_PATH, weights_only=False)

    # 计算并设置 num_nodes 属性
    for data in dataset:
        num_nodes = data.edge_index.max().item() + 1  # 假设 `edge_index` 的最大值代表节点数
        data.num_nodes = num_nodes  # 为每个 data 对象设置 num_nodes 属性

    # 划分训练集和验证集 (80% 训练, 20% 验证)
    train_size = int(len(dataset) * 0.8)
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:]
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- 3. 初始化模型 ---
    # 获取一下维度信息
    sample_data = dataset[0]
    STATIC_DIM = sample_data.x_static.size(1) # 你之前提取的17维等
    HIST_SEQ_LEN = sample_data.hist_speed.size(1) # 12
    EVENT_DIM = sample_data.event_vector.size(1) # 3
    
    model = TrafficFlowPredictor(
        static_dim=STATIC_DIM,
        hist_seq_len=HIST_SEQ_LEN,
        rnn_hidden_dim=32,
        gnn_hidden_dim=64,
        event_dim=EVENT_DIM
    ).to(DEVICE)
    
    criterion = nn.MSELoss() # 回归任务使用均方误差
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

     # 创建 .txt 文件并写入表头
    with open(output_file, "w") as f:
        f.write("Epoch, Train Loss (MSE), Val Loss (MSE)\n")  # 写入表头

    # --- 4. 开始训练 ---
    print("开始训练...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        # 使用 tqdm 显示进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in pbar:
            batch = batch.to(DEVICE)
            
            optimizer.zero_grad()
            
            # 前向传播 (PyG DataLoader 会自动把 batch.xxx 拼接展开)
            preds = model(
                x_static=batch.x_static, 
                hist_speed=batch.hist_speed, 
                time_idx=batch.time_idx, 
                day_idx=batch.day_idx, 
                event_vector=batch.event_vector, 
                edge_index=batch.edge_index
            )
            
            # 计算损失
            loss = criterion(preds, batch.y)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch.num_graphs
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        avg_train_loss = total_loss / len(train_dataset)
        
        # --- 5. 验证模型 ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                preds = model(
                    batch.x_static, batch.hist_speed, batch.time_idx, 
                    batch.day_idx, batch.event_vector, batch.edge_index
                )
                loss = criterion(preds, batch.y)
                val_loss += loss.item() * batch.num_graphs
                
        avg_val_loss = val_loss / len(val_dataset)
        print(f"Epoch[{epoch+1}/{EPOCHS}] | Train Loss(MSE): {avg_train_loss:.2f} | Val Loss(MSE): {avg_val_loss:.2f}")

        # 自动记录每轮损失到 txt 文件
        with open(output_file, "a") as f:  # 追加模式打开文件
            f.write(f"{epoch+1}, {avg_train_loss:.2f}, {avg_val_loss:.2f}\n")

    print("\n GNN模型基座构建完毕。")
    # 可以保存模型权重
    torch.save(model.state_dict(), "st_base_model.pth")

if __name__ == "__main__":
    train()