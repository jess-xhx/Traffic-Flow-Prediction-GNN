import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import numpy as np
from tqdm import tqdm

# 导入你之前创建的三层模型 (假设你把代码保存在 models/traffic_flow_model.py)
from models.traffic_flow_model import TrafficFlowPredictor 

def train():
    # --- 1. 配置参数 ---
    DATA_PATH = r"E:\vscode项目文件\Traffic_Flow_Prediction\data\processed\mock_dynamic_dataset.pt"
    BATCH_SIZE = 8
    EPOCHS = 20
    LEARNING_RATE = 0.001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用计算设备: {DEVICE}")

    # --- 2. 加载数据 ---
    print("加载数据集...")
    dataset = torch.load(DATA_PATH)
    
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
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

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

    print("\n✅ 训练全流程跑通！模型基座构建完毕。")
    # 可以保存模型权重
    torch.save(model.state_dict(), "st_base_model.pth")

if __name__ == "__main__":
    train()