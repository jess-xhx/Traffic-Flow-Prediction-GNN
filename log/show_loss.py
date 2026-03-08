import matplotlib.pyplot as plt
import pandas as pd

# 加载数据
file_path = "E:/vscode项目文件/Traffic_Flow_Prediction/log/training_losses_20260308_211628.txt"
data = pd.read_csv(file_path)

# 打印列名
print("Columns in the data:", data.columns)

# 清除列名中的空格
data.columns = data.columns.str.strip()

# 绘制训练损失和验证损失
plt.figure(figsize=(10, 6))

# 训练损失曲线
plt.plot(data['Epoch'], data['Train Loss (MSE)'], label='Train Loss (MSE)', color='b', marker='o')

# 验证损失曲线
plt.plot(data['Epoch'], data['Val Loss (MSE)'], label='Val Loss (MSE)', color='r', marker='x')

# 添加标题和标签
plt.title('Training and Validation Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()

# 显示网格
plt.grid(True)

# 显示图形
plt.show()