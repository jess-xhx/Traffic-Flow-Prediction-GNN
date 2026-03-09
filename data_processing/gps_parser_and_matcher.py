import json
import torch
import geopandas as gpd
from shapely.geometry import Point
import re
import os

# ==========================================
# 第一步：数据解析与清洗 (Parsing) 
# ==========================================
def parse_and_clean_gps(file_path):
    """
    解析 JSON 文本，提取 (经度, 纬度) 序列，并剔除连续停靠的静止点。
    兼容单行 JSON 或多行 JSONL 格式（修复 Extra data 报错）。
    """
    print(f"1. 正在解析文件: {file_path}")
    
    trajectory_coords = []
    total_raw_points = 0
    
    with open(file_path, "r", encoding="utf-8") as f:
        # 逐行读取文件，应对 "Extra data" 错误
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue  # 跳过空行
            
            try:
                # 1. 解析当前行的 JSON
                raw_data = json.loads(line)
                
                # 2. 解析内嵌的 GpsPointsJson 字符串
                gps_json_str = raw_data.get("GpsPointsJson", "{}")
                gps_data = json.loads(gps_json_str)
                track_array = gps_data.get("result", {}).get("trackArray", [])
                
                total_raw_points += len(track_array)
                
                # 3. 提取坐标并进行基础清洗（去重）
                for pt in track_array:
                    lon = float(pt["lon"])
                    lat = float(pt["lat"])
                    
                    # 清洗逻辑：如果车辆停在原地（如等红绿灯、装卸货），坐标不变，则跳过不记录
                    if not trajectory_coords or trajectory_coords[-1] != (lon, lat):
                        trajectory_coords.append((lon, lat))
                        
            except json.JSONDecodeError as e:
                print(f"   [警告] 第 {line_num} 行解析 JSON 失败，已跳过。原因: {e}")
            except Exception as e:
                print(f"   [警告] 第 {line_num} 行处理时发生未知错误: {e}")

    print(f"   -> 从所有行中共提取原始 GPS 点数量: {total_raw_points}")
    print(f"   -> 清洗去重后有效坐标数量: {len(trajectory_coords)}")
    
    return trajectory_coords

# ==========================================
# 第二步：地图匹配 (Map Matching)
# ==========================================
def map_matching_to_edge_sequence(trajectory_coords, static_dataset_path):
    """
    将孤立的 GPS 点匹配到静态路网的对偶图节点 (Edge IDs) 上
    """
    print("\n2. 开始地图匹配 (GPS 映射至图节点 Edge_ID)...")
    
    # 1. 加载静态路网数据集，提取路段映射表 (GeoDataFrame)
    static_data = torch.load(static_dataset_path, weights_only=False)
    gdf_edges = static_data.mapping
    
    # 确保路段数据具有坐标系 (WGS84)
    if gdf_edges.crs is None:
        gdf_edges.set_crs("EPSG:4326", inplace=True)
        
    # 2. 将清洗后的 GPS 坐标转化为 GeoPandas 格式
    geometry =[Point(lon, lat) for lon, lat in trajectory_coords]
    gps_gdf = gpd.GeoDataFrame(geometry=geometry, crs="EPSG:4326")
    
    # 3. 核心步骤：空间最近邻查询 (Spatial Nearest Neighbor)
    # 利用 R-tree 空间索引，极其快速地找到距离每个 GPS 点最近的路段 (LineString)
    nearest_edge_indices = gdf_edges.sindex.nearest(gps_gdf.geometry)[1]
    
    # 4. 获取对应的 edge_id (即你的 GNN 里的节点 ID)
    raw_edge_ids = gdf_edges.iloc[nearest_edge_indices]['edge_id'].tolist()
    
    # 5. 序列拓扑清洗
    # 物理意义：一辆车在一条较长的路段上行驶时，会产生多个 GPS 点，
    # 这些点都会匹配到同一个 edge_id。在输入给模型时，我们只需记录“经过了该路段1次”。
    clean_edge_seq =[]
    for eid in raw_edge_ids:
        if not clean_edge_seq or clean_edge_seq[-1] != eid:
            clean_edge_seq.append(eid)
            
    print(f"   -> 映射的原始路段序列长度: {len(raw_edge_ids)}")
    print(f"   -> 拓扑清洗后最终 Edge ID 序列长度: {len(clean_edge_seq)}")
    print(f"\n  最终生成的路段 ID 序列 : \n{clean_edge_seq}")
    
    return clean_edge_seq


if __name__ == "__main__":
    # 1. 配置输入输出路径
    TXT_FILE_PATH = r"E:\vscode项目文件\Traffic_Flow_Prediction\data\raw\gps_trajectory\MRT20250710314-保定-保定.txt"  
    STATIC_DATASET_PATH = r"E:\vscode项目文件\Traffic_Flow_Prediction\data\processed\baoding_static_road_gnn_dataset.pt"
    
    # 定义输出文件夹的绝对路径，并确保其存在
    OUTPUT_DIR = r"E:\vscode项目文件\Traffic_Flow_Prediction\data\raw"
    os.makedirs(OUTPUT_DIR, exist_ok=True) 
    
    # 2. 从文件路径中提取数字作为 ID (current_time)
    file_name = os.path.basename(TXT_FILE_PATH)
    match = re.search(r'\d+', file_name)
    current_time = match.group() if match else "unknown_time"
    
    # 3. 执行流水线 - 解析与清洗
    coords = parse_and_clean_gps(TXT_FILE_PATH)
    
    if coords:
        # ======= 拼接路径并保存原始 GPS 点 =======
        or_gps_filename = f"OR_GPS_{current_time}.txt"
        or_gps_filepath = os.path.join(OUTPUT_DIR, or_gps_filename)  # 生成完整路径
        
        with open(or_gps_filepath, "w", encoding="utf-8") as f:
            for lon, lat in coords:
                f.write(f"{lon},{lat}\n")
        print(f"\n 原始 GPS 点已成功保存至: {or_gps_filepath}")

        # 4. 执行流水线 - 地图匹配
        edge_sequence = map_matching_to_edge_sequence(coords, STATIC_DATASET_PATH)
        
        # ======= 拼接路径并保存最终的路段 ID 序列 =======
        f_gps_filename = f"F_GPS_{current_time}.txt"
        f_gps_filepath = os.path.join(OUTPUT_DIR, f_gps_filename)  # 生成完整路径
        
        with open(f_gps_filepath, "w", encoding="utf-8") as f:
            for edge_id in edge_sequence:
                f.write(f"{edge_id}\n")
        print(f" 最终路段 ID 序列已成功保存至: {f_gps_filepath}")
        
    else:
        print("未提取到有效的 GPS 坐标！")