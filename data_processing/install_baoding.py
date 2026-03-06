import os
import time
import threading
import osmnx as ox
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

# --- 配置区域 ---
PLACE_NAME = "Baoding, China"
# 建议文件名加上 _raw 以示区别
SAVE_PATH = "./data/processed/baoding_raw.graphml" 

console = Console()

def download_raw_map_task(result_container):
    """
    只负责下载和保存最原始的数据，不做任何处理
    """
    try:
        # 1. 下载 (关键修改：simplify=False) 
        # simplify=False: 保留所有节点（包括弯道上的形状点），不合并路段。
        # 这是"最原始"的状态，对应你 Slide 左侧的那种图。
        console.print(f"[dim]正在向 OSM 发送请求 (simplify=False)...[/dim]")
        
        G_raw = ox.graph_from_place(
            PLACE_NAME, 
            network_type='drive', 
            simplify=False  # <--- 关键！禁止自动简化
        )
        
        # 2. 直接保存
        # 此时 G_raw 里的坐标还是经纬度 (WGS84)，且包含所有中间点
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
        ox.save_graphml(G_raw, filepath=SAVE_PATH)
        
        result_container['graph'] = G_raw
        result_container['success'] = True
        
    except Exception as e:
        result_container['error'] = e
        result_container['success'] = False

def main():
    # 检查文件是否存在
    if os.path.exists(SAVE_PATH):
        console.print(f"[bold yellow]警告:[/bold yellow] 原始地图文件 {SAVE_PATH} 已存在。")
        if input("是否覆盖重新下载? (y/n): ").lower() != 'y':
            console.print("[green]已取消。[/green]")
            return

    console.print(f"[bold blue]开始获取 {PLACE_NAME} 的原始路网 (Raw OSM)...[/bold blue]")
    console.print("注: 由于不进行简化(simplify=False)，数据量会比平时大，解析时间稍长。")
    print("-" * 50)

    result = {}
    t = threading.Thread(target=download_raw_map_task, args=(result,))
    t.start()

    # 进度动画
    with Progress(
        SpinnerColumn("dots"),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        transient=True
    ) as progress:
        task = progress.add_task(f"[cyan]正在下载并解析原始数据...", total=None)
        while t.is_alive():
            time.sleep(0.1)

    # 结果反馈
    if result.get('success'):
        G = result['graph']
        file_size = os.path.getsize(SAVE_PATH) / (1024 * 1024)
        
        console.print(f"[bold green]✅ 下载成功![/bold green]")
        console.print(f"📂 文件路径: [underline]{SAVE_PATH}[/underline]")
        console.print(f"📊 [bold]原始数据统计 (未简化):[/bold]")
        console.print(f"   - 文件大小: {file_size:.2f} MB")
        console.print(f"   - 原始节点数: {len(G.nodes)} (包含所有形状点)")
        console.print(f"   - 原始边数:   {len(G.edges)}")
    else:
        console.print(f"[bold red]❌ 失败![/bold red] {result.get('error')}")

if __name__ == "__main__":
    main()