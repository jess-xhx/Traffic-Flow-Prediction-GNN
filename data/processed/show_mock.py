import torch
import osmnx as ox
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import sys
import os
# 将 data_processing 文件夹的绝对路径加入 Python 的搜索环境变量中
sys.path.append(r"E:\vscode项目文件\Traffic_Flow_Prediction\data_processing")


# 从你现有的脚本中导入图处理函数，以保证路段的提取顺序与 15590 这个维度绝对一致
from Static_road_data import keep_logistics_roads, topological_simplification

def visualize_geo_heatmap(raw_graph_path, dynamic_data_path, time_step=0):
    print("1. 正在加载保定市原始路网 (这可能需要十几秒)...")
    G_raw = ox.load_graphml(raw_graph_path)

    print("2. 正在执行路网过滤与化简，对齐 15590 个骨架节点...")
    # 这一步保证了我们提取的物理边顺序和生成 .pt 时一模一样
    G_logistics = keep_logistics_roads(G_raw)
    G_simplified = topological_simplification(G_logistics)
    
    # 转换为 DataFrame 获取 (u, v, key) 的排列顺序
    gdf_edges = ox.graph_to_gdfs(G_simplified, nodes=False, edges=True)
    gdf_edges = gdf_edges.reset_index()

    print("3. 加载伪造的时空动态数据...")
    dynamic_data = torch.load(dynamic_data_path, weights_only=False)
    
    # 获取指定时间片 (比如第 0 个样本)
    sample = dynamic_data[time_step]
    
    # 提取该时刻所有路段的"当前速度"
    # hist_speed 形状为 [15590, 12, 1]，提取最后一个时刻 -> [15590]
    speeds = sample.hist_speed[:, -1, 0].numpy()

    print("4. 正在将车速映射到物理街道...")
    # 建立 (起点u, 终点v, 边编号key) 到 车速 的映射字典
    speed_dict = {}
    for idx, row in gdf_edges.iterrows():
        u = row['u']
        v = row['v']
        # osmnx 有可能存在多重边，所以需要 key
        key = row.get('key', 0) 
        speed_dict[(u, v, key)] = speeds[idx]

    print("5. 正在生成热力图着色方案...")
    # 速度范围限制在 5km/h (极度拥堵/深红) 到 60km/h (畅通/深绿)
    norm = mcolors.Normalize(vmin=5, vmax=60)
    cmap = plt.get_cmap('RdYlGn') # 红-黄-绿 颜色条

    edge_colors = []
    edge_linewidths = []
    
    # 遍历化简后路网的每一条边进行着色
    for u, v, k, data in G_simplified.edges(keys=True, data=True):
        if (u, v, k) in speed_dict:
            speed = speed_dict[(u, v, k)]
            edge_colors.append(cmap(norm(speed)))
            edge_linewidths.append(1.5)  # 我们的骨架路网稍微加粗显示
        else:
            # 理论上不会走到这里，加个兜底
            edge_colors.append((0.5, 0.5, 0.5, 0.5)) 
            edge_linewidths.append(0.5)

    print("6. 正在利用真实地理坐标绘制保定市交通图...")
    fig, ax = ox.plot_graph(
        G_simplified,
        node_size=0,                # 隐藏节点，只显示街道线段
        edge_color=edge_colors,
        edge_linewidth=edge_linewidths,
        bgcolor='#111111',          # 深色背景让红绿交通线更醒目
        show=False,
        close=False,
        figsize=(12, 12)            # 输出高清大图
    )

    # 添加右侧的颜色图例 (Colorbar)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, fraction=0.046, pad=0.04)
    cbar.set_label('Traffic Speed (km/h)', color='white', fontsize=12)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    plt.title(f"Baoding City Traffic Heatmap (Sample {time_step})", color='white', fontsize=18, pad=20)
    
    # 保存高清大图并显示
    save_img_path = r"E:\vscode项目文件\Traffic_Flow_Prediction\data\processed\baoding_real_map_traffic.png"
    plt.savefig(save_img_path, dpi=300, bbox_inches='tight', facecolor='#111111')
    print(f" ✅ 可视化成功！真实地理分布图已保存为 {save_img_path}")
    plt.show()

if __name__ == "__main__":
    # 使用你截图中展示的正确路径
    RAW_GRAPH = r"E:\vscode项目文件\Traffic_Flow_Prediction\data\raw\baoding_raw.graphml"
    DYNAMIC_DATA = r"E:\vscode项目文件\Traffic_Flow_Prediction\data\processed\mock_dynamic_dataset.pt"
    
    # 渲染第 0 个时间切片（你可以改成其他数字看不同时间的拥堵变化）
    visualize_geo_heatmap(RAW_GRAPH, DYNAMIC_DATA, time_step=0)