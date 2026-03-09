import os
import torch
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import osmnx as ox

# ==========================================
# 1. 配置文件路径 (请根据你的实际情况修改)
# ==========================================
OR_GPS_PATH = r"E:\vscode项目文件\Traffic_Flow_Prediction\data\raw\OR_GPS_20250710314.txt"
F_GPS_PATH = r"E:\vscode项目文件\Traffic_Flow_Prediction\data\raw\F_GPS_20250710314.txt"

GRAPHML_PATH = r"E:\vscode项目文件\Traffic_Flow_Prediction\data\raw\baoding_raw.graphml"
PT_DATASET_PATH = r"E:\vscode项目文件\Traffic_Flow_Prediction\data\processed\baoding_static_road_gnn_dataset.pt"

OUTPUT_IMG_PATH = r"E:\vscode项目文件\Traffic_Flow_Prediction\data\raw\Trajectory_Comparison.png"

def main():
    print("正在加载数据，请稍候...")
    
    # ------------------------------------------
    # 2. 读取原始 GPS 数据
    # ------------------------------------------
    print("-> 读取原始 GPS 坐标...")
    or_gps_df = pd.read_csv(OR_GPS_PATH, header=None, names=['lon', 'lat'])

    # ------------------------------------------
    # 3. 读取匹配后的 Edge ID 数据
    # ------------------------------------------
    print("-> 读取匹配后的路段 ID...")
    with open(F_GPS_PATH, 'r') as f:
        # 逐行读取并转换为整数
        matched_edge_ids = [int(line.strip()) for line in f if line.strip()]

    # ------------------------------------------
    # 4. 加载路网数据
    # ------------------------------------------
    print("-> 加载原始 GraphML 路网数据 (可能需要一点时间)...")
    # 使用 osmnx 加载 graphml，它非常适合处理这种路网文件
    G_raw = ox.load_graphml(GRAPHML_PATH)
    # 转换为 GeoDataFrame 以便提取几何形状进行绘制
    nodes, edges_raw = ox.graph_to_gdfs(G_raw)

    print("-> 加载静态 GNN 数据集...")
    static_data = torch.load(PT_DATASET_PATH, weights_only=False)
    edges_pt = static_data.mapping

    # ------------------------------------------
    # 5. 开始绘图 (双子图对比)
    # ------------------------------------------
    print("-> 正在生成可视化图像...")
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # =================
    # 图 1: 原始路网 + 原始 GPS 点
    # =================
    ax1 = axes[0]
    # 绘制原始路网底图 (浅灰色)
    edges_raw.plot(ax=ax1, color='#d3d3d3', linewidth=0.5, zorder=1)
    # 叠加原始 GPS 点 (红色散点)
    ax1.scatter(or_gps_df['lon'], or_gps_df['lat'], 
                c='red', s=15, alpha=0.8, zorder=2, label='Raw GPS Points')
    
    ax1.set_title("Original GPS Trajectory on Raw Network", fontsize=16)
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    ax1.legend(loc="upper right")
    
    # 获取 GPS 点的边界，让图 1 自动聚焦到车辆行驶的区域
    minx, miny = or_gps_df['lon'].min(), or_gps_df['lat'].min()
    maxx, maxy = or_gps_df['lon'].max(), or_gps_df['lat'].max()

    # 分别设置左右和上下的留白缓冲 (数值代表经纬度的度数)
    x_buffer = 0.005  # 左右视野保持原样 (大约500米)
    y_buffer = 0.015  # 将上下视野调大 3 倍 (大约1.5公里)


    ax1.set_xlim(minx - x_buffer, maxx + x_buffer)
    ax1.set_ylim(miny - y_buffer, maxy + y_buffer)

    # =================
    # 图 2: GNN对偶图映射 + 匹配路段
    # =================
    ax2 = axes[1]
    # 绘制全部路段作为底图 (浅灰色)
    edges_pt.plot(ax=ax2, color='#d3d3d3', linewidth=0.5, zorder=1)
    
    # 筛选出被匹配到的路段
    matched_edges_gdf = edges_pt[edges_pt['edge_id'].isin(matched_edge_ids)]
    
    # 绘制高亮的匹配路段 (蓝色粗线)
    if not matched_edges_gdf.empty:
        matched_edges_gdf.plot(ax=ax2, color='blue', linewidth=3, zorder=2, label='Matched Edges')
    
    ax2.set_title("Map-Matched Edges on GNN Dataset", fontsize=16)
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    if not matched_edges_gdf.empty:
        ax2.legend(loc="upper right")
        
    # 图 2 的视角与图 1 保持一致，方便肉眼对比
    ax2.set_xlim(minx - x_buffer, maxx + x_buffer)
    ax2.set_ylim(miny - y_buffer, maxy + y_buffer)

    # ------------------------------------------
    # 6. 保存并展示
    # ------------------------------------------
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG_PATH, dpi=300, bbox_inches='tight')
    print(f"\n[成功] 可视化对比图已保存至: {OUTPUT_IMG_PATH}")
    plt.show()

if __name__ == "__main__":
    main()