import os
import osmnx as ox
import networkx as nx
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tqdm import tqdm
from torch_geometric.data import Data


# 第一步 过滤非物流道路 + 拓扑简化
def keep_logistics_roads(G_input):
    """
    针对物流运输场景的过滤道路
    """
    print("开始过滤非物流道路...")

    # 1. 创建副本，防止修改原始变量
    G = G_input.copy()
    
    # 2. 定义物流道路类型白名单
    logistics_types = {
        'motorway', 'motorway_link',
        'trunk', 'trunk_link',
        'primary', 'primary_link',
        'secondary', 'secondary_link',
        'tertiary', 'tertiary_link',
        'service'
    }

    print(f"输入路网 边数量: {len(G.edges)}")
    
    edges_to_remove = []
    
    for u, v, k, data in G.edges(keys=True, data=True):
        hw = data.get('highway')
        
        # 1. 基础白名单检查
        is_target_road = False
        hw_list = hw if isinstance(hw, list) else [hw]
        
        if set(hw_list).intersection(logistics_types):
            is_target_road = True
        
        # 2. 针对 service 的特殊精细化清洗
        if is_target_road:
            service_type = data.get('service')
            # 剔除停车场内部路、私家车道、胡同等
            if service_type in ['parking_aisle', 'driveway', 'drive-through', 'alley']:
                is_target_road = False
        
        if not is_target_road:
            edges_to_remove.append((u, v, k))

    # 3. 删除边
    G.remove_edges_from(edges_to_remove)
    print(f"已移除 {len(edges_to_remove)} 条非物流道路")
    
    # 4. 移除孤立节点
    if len(G.nodes) > 0:
        # nx.isolates 返回一个迭代器，列出所有度为0的节点
        isolates = list(nx.isolates(G))
        G.remove_nodes_from(isolates)
        print(f"已移除 {len(isolates)} 个孤立节点")
    
    print(f"最终路网 边数量: {len(G.edges)}")
    print(f"最终路网 节点数量: {len(G.nodes)}")
    
    return G


def topological_simplification(G):
    """
    对路网进行拓扑简化
    """
    print("开始拓扑简化...")
    
    # 1. 创建副本
    G_simplified = G.copy()
    
    # 2. 拓扑简化
    G_simplified = ox.simplify_graph(G_simplified)
    
    print(f"拓扑简化后 边数量: {len(G_simplified.edges)}")
    print(f"拓扑简化后 节点数量: {len(G_simplified.nodes)}")
    
    return G_simplified

# 第二步 特征工程
def parse_list_attribute(val, dtype=float, default=None):
    """
    辅助函数：处理OSM中可能出现的列表字段 (例如 lanes=['2', '3'])
    策略：如果是列表，取第一个元素；如果是单一值直接转换。
    """
    if val is None or val != val: # Check for NaN
        return default
    
    if isinstance(val, list):
        val = val[0]
        
    try:
        # 清洗单位，例如 '50 mph' -> 50
        if isinstance(val, str):
            val = val.replace(' mph', '').replace(' km/h', '')
        return dtype(val)
    except:
        return default

def process_graph_features(G):
    """
    输入: G (NetworkX图)
    输出: 
        x_tensor: PyTorch张量 [Num_Edges（路段数量）, Num_Features（路段特征）]
        gdf_edges: 包含边信息的GeoDataFrame (充当映射表)
    """
    print("开始特征工程 (提取边特征 -> Tensor X)...")

    # 1. 将图转换为 DataFrame 格式，方便批量处理
    # nodes=False 表示只提取边
    gdf_edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
    
    # 保证 index 是 0, 1, 2... 对应 Tensor 的行号
    # 这一步非常关键！后续构建对偶图必须依此索引
    gdf_edges = gdf_edges.reset_index()
    gdf_edges['edge_id'] = gdf_edges.index
    
    print(f"   - 待处理路段总数: {len(gdf_edges)}")

    # ==========================================
    # A. 类别特征处理 
    # ==========================================
    
    # --- 1. Highway (道路等级) ---
    # 定义主要道路白名单，其余归为 'other'
    valid_highways = [
        'motorway', 'motorway_link', 'trunk', 'trunk_link', 
        'primary', 'primary_link', 'secondary', 'secondary_link',
        'tertiary', 'tertiary_link', 'residential', 'service'
    ]
    
    def clean_highway(h):
        if isinstance(h, list): h = h[0] # 取列表第一个
        return h if h in valid_highways else 'other'

    gdf_edges['highway_clean'] = gdf_edges['highway'].apply(clean_highway)
    
    # One-Hot 编码
    encoder = OneHotEncoder(sparse_output=False, categories='auto')
    highway_onehot = encoder.fit_transform(gdf_edges[['highway_clean']])
    print(f"   - Highway One-Hot 维度: {highway_onehot.shape[1]}")

    # --- 2. Boolean 特征 (Oneway, Bridge, Tunnel, Junction) ---
    def clean_bool(x):
        # OSM中通常是 boolean, 'yes'/'no', '1'/'0' 或者 NaN
        if x is None: return 0.0
        s = str(x).lower()
        if s in ['true', 'yes', '1']: return 1.0
        return 0.0

    # 提取并转换为 (N, 1) 的列向量
    feat_oneway = gdf_edges['oneway'].apply(clean_bool).values.reshape(-1, 1)
    
    # 注意：某些数据可能没有 bridge/tunnel/junction 字段，需容错处理
    feat_bridge = gdf_edges['bridge'].apply(clean_bool).values.reshape(-1, 1) if 'bridge' in gdf_edges else np.zeros((len(gdf_edges), 1))
    feat_tunnel = gdf_edges['tunnel'].apply(clean_bool).values.reshape(-1, 1) if 'tunnel' in gdf_edges else np.zeros((len(gdf_edges), 1))
    feat_junction = gdf_edges['junction'].apply(clean_bool).values.reshape(-1, 1) if 'junction' in gdf_edges else np.zeros((len(gdf_edges), 1))

    # ==========================================
    # B. 数值特征处理 (Numerical)
    # ==========================================
    
    # --- 1. Lanes (车道数) ---
    # 缺失值填充为 2 (城市道路常见默认值)
    gdf_edges['lanes_val'] = gdf_edges['lanes'].apply(lambda x: parse_list_attribute(x, default=2.0))
    
    # --- 2. Maxspeed (最高限速) ---
    # 缺失值填充为 40 (km/h)
    gdf_edges['speed_val'] = gdf_edges['maxspeed'].apply(lambda x: parse_list_attribute(x, default=40.0))
    
    # --- 3. Length (长度) ---
    # 对数变换 (Log Transform) 以处理长尾分布
    # log1p = log(x + 1) 避免 log(0)
    gdf_edges['len_log'] = np.log1p(gdf_edges['length'].astype(float))

    # --- 4. 归一化 (Standardization) ---
    # 将数值特征组合在一起进行归一化
    numerical_data = gdf_edges[['lanes_val', 'speed_val', 'len_log']].values
    scaler = StandardScaler()
    numerical_norm = scaler.fit_transform(numerical_data)

    # ==========================================
    # C. 特征拼接 (Concatenation)
    # ==========================================
    
    # 拼接所有特征矩阵
    # 顺序: One-Hot (N, H) + Boolean (N, 4) + Numerical (N, 3)
    feature_matrix = np.hstack([
        highway_onehot, 
        feat_oneway, 

        feat_bridge, 
        feat_tunnel,
        feat_junction,
        numerical_norm
    ])
    
    # 转换为 PyTorch FloatTensor
    x_tensor = torch.tensor(feature_matrix, dtype=torch.float32)
    
    print(f"   特征工程完成!")
    print(f"   - 输入张量 X 形状: {x_tensor.shape}")
    print(f"   - 包含特征: Highway({highway_onehot.shape[1]}) + Bool(4) + Num(3) = {x_tensor.shape[1]} 维")
    
    return x_tensor, gdf_edges

# 第三步 构建对偶图
def build_dual_graph(gdf_edges):
    """
    构建对偶图Line Graph的邻接关系
    
    输入:
        gdf_edges: 经过特征工程处理后的 DataFrame,必须包含 'edge_id', 'u', 'v' 列
        
    输出:
        edge_index: PyTorch LongTensor [2, Num_Connections]
                    Row 0: 源路段 ID (Source Edge)
                    Row 1: 目标路段 ID (Target Edge)
    """
    print(" 开始构建对偶图连接关系 ...")
    
    # 1. 建立索引加速查找
    # 我们需要快速知道：对于节点 n，哪些路段是以它为起点的？
    # 格式: node_id -> [edge_id_1, edge_id_2, ...]
    node_to_outgoing_edges = {}
    
    print("   - 正在建立节点查找表...")
    # 遍历所有路段，记录它们的起点 (u)
    for idx, row in gdf_edges.iterrows():
        u_node = row['u']      # 该路段的起点物理ID
        eid = row['edge_id']   # 该路段的逻辑ID (0...N)
        
        if u_node not in node_to_outgoing_edges:
            node_to_outgoing_edges[u_node] = []
        node_to_outgoing_edges[u_node].append(eid)
        
    # 2. 构建连边 
    # 逻辑：遍历每一条路段 E_in (u -> v)，找到所有以 v 为起点的路段 E_out (v -> w)
    source_edges = []
    target_edges = []
    
    print("   - 正在计算路段间的连通性...")
    # 使用 tqdm 显示进度，因为路段可能很多
    for _, row in tqdm(gdf_edges.iterrows(), total=len(gdf_edges), desc="Scanning Connections"):
        current_edge_id = row['edge_id']
        connection_node = row['v']  # 当前路段的终点
        
        # 查找：有哪些路段是从 connection_node 出发的？
        # 如果有，说明 current_edge_id 可以驶入这些路段
        if connection_node in node_to_outgoing_edges:
            next_edges = node_to_outgoing_edges[connection_node]
            
            for next_edge_id in next_edges:
                # 排除掉头 (U-turn) 的情况：
                # 只有当这是双向道路且完全重合时才需要小心，
                # 但一般 GNN 允许保留掉头连接，除非明确禁止。
                # 这里我们只排除"自环"（虽然物理上很少见）
                if current_edge_id != next_edge_id:
                    source_edges.append(current_edge_id)
                    target_edges.append(next_edge_id)
    
    # 3. 转换为 PyTorch Tensor
    if len(source_edges) == 0:
        print(" 警告: 未找到任何连接！请检查路网数据的连通性。")
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        # 堆叠成 [2, M]
        edge_index = torch.tensor([source_edges, target_edges], dtype=torch.long)
        
    print(f"  对偶图构建完成!")
    print(f"   - 路段节点总数: {len(gdf_edges)}")
    print(f"   - 路段连接总数: {edge_index.shape[1]}")
    print(f"   - 平均度数 (连接密度): {edge_index.shape[1] / len(gdf_edges):.2f}")
    
    return edge_index

# 第四步 保存结果（x_tensor, edge_index, gdf_edges）
def save_gnn_dataset(x_tensor, edge_index, gdf_edges):
    """
    将特征和图结构组合并保存为 PyG 数据集
    """
    print("正在打包并保存 PyG 数据对象...")

    # 1. 组合成 PyG Data 对象

    # x_tensor: 节点特征矩阵 [N, Features]
    # edge_index: 邻接矩阵 [2, M]
    # mapping: 保存映射表 (gdf_edges)，预测时用来查具体的路段信息

    data = Data(x=x_tensor, edge_index=edge_index, mapping=gdf_edges)

    # 2. 定义保存路径 (使用你指定的绝对路径)
    # 使用 r"" 原始字符串避免 Windows 路径反斜杠转义问题
    save_dir = r"E:\vscode项目文件\Traffic_Flow_Prediction\2.data_processing\Static_road"
    file_name = "baoding_static_road_gnn_dataset.pt"
    save_path = os.path.join(save_dir, file_name)

    # 3. 确保目录存在 (如果 data 文件夹不存在则自动创建)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"   - 创建目录: {save_dir}")

    # 4. 保存文件
    try:
        torch.save(data, save_path)
        print(f" 成功! 数据集已保存至:")
        print(f"    {save_path}")
        print("-" * 30)
        print(f"   - Data对象概览: {data}")
        # 输出示例: Data(x=[15376, 17], edge_index=[2, 40000], mapping=[15376 rows x ...])
    except Exception as e:
        print(f" 保存失败: {e}")

if __name__ == "__main__":
    
    # 加载原始路网数据
    print("加载原始路网数据...")
    G_raw = ox.load_graphml(r"E:\vscode项目文件\Traffic_Flow_Prediction\2.data_processing\Static_road\baoding_raw.graphml")

    ### 第一步 ###
    print("\n开始第一步处理...")
    # 过滤非物流道路
    G_logistics = keep_logistics_roads(G_raw)
    # 拓扑简化
    G_simplified = topological_simplification(G_logistics)

    ### 第三步 ###
    print("\n开始第二步处理...")
    # 特征工程
    x_tensor, gdf_edges = process_graph_features(G_simplified)

    ### 第四步 ###
    print("\n开始第三步处理...")
    # 构建对偶图
    edge_index = build_dual_graph(gdf_edges)

    ### 第五步 ###
    print("\n开始第四步处理...")
    # 保存结果
    save_gnn_dataset(x_tensor, edge_index, gdf_edges)



 