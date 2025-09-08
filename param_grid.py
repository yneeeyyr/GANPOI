'''from itertools import product
import numpy as np

param_grid={
    'lr':[0.0001,0.001,0.01,0.1],
    'gen_loss_weight':[],
    'dis_loss_weight':[],
    'geo_loss_weight':[]
}

#生成所有超参数组合
param_combinations=list(product(*param_grid.values()))

results=[]

#网格搜索
for params in param_combinations:
    param_dict=dict(zip(param_grid.keys(),params))
    print(f"正在训练，组合参数：{param_dict}")
    score
    results.append((param_dict,score))

best_params,best_score=max(results,key=lambda x:x[1])
print("\n最佳参数组合:",best_params)
print("最佳得分:",best_score)'''

'''

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, ArrowStyle

# Initialize the figure
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis("off")

# Add labeled modules with boxes
def add_module(ax, pos, size, label, color="lightblue"):
    x, y = pos
    w, h = size
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3", fc=color, ec="black", lw=1.5)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10, wrap=True)

# Draw arrows for connections
def draw_arrow(ax, start, end, color="black"):
    ax.annotate(
        "", xy=end, xycoords="data", xytext=start, textcoords="data",
        arrowprops=dict(arrowstyle=ArrowStyle.CurveFilledB(head_length=1, head_width=1.2), color=color, lw=1.5)
    )

# Define positions and connections for each module
# Input Module
add_module(ax, (1, 7), (2, 1), "Input Data", "lightgreen")

# Graph Learning Module
add_module(ax, (4, 6.5), (4, 3), "Graph Learning\n(Trajectory Flow + GCN)", "lightyellow")

# Embedding Module
add_module(ax, (9, 6.5), (4, 3), "Embedding Module\n(User + Time + Geo)", "lightblue")

# Embedding Fusion
add_module(ax, (14, 7.5), (2, 1), "Fusion Layer", "pink")

# Transformer Layer
add_module(ax, (17, 6.5), (4, 3), "Transformer\n(Encoder-Decoder)", "lightcyan")

# Output Module
add_module(ax, (22, 7.5), (2, 1), "Output Heads\n(POI + Category)", "orange")

# Arrow connections between modules
arrow_positions = [
    ((2.5, 7.5), (4, 8)),  # Input to Graph Learning
    ((6, 7.5), (9, 8)),    # Graph Learning to Embedding
    ((13, 8), (14, 8)),    # Embedding to Fusion
    ((16, 8), (17, 8)),    # Fusion to Transformer
    ((21, 8), (22, 8)),    # Transformer to Output
]

# Draw arrows
for start, end in arrow_positions:
    draw_arrow(ax, start, end)

# Set axis limits
ax.set_xlim(0, 25)
ax.set_ylim(0, 10)

# Save the diagram
plt.savefig("model_architecture.png", dpi=300)
print("Model architecture diagram saved as 'model_architecture.png'")
#plt.show()
'''
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

# 创建模型包装器类
class ModelWrapper:
    def __init__(self, args):
        self.args = args

    def fit(self, X_train, y_train):
        """
        模型训练方法，使用传入的超参数 args 训练模型。
        """
        self.args.data_train = X_train
        self.args.data_val = y_train
        train(self.args)  # 这里调用现有的训练函数

    def score(self, X_test, y_test):
        """
        模型验证方法，返回验证集的性能分数（如准确率）。
        """
        val_score = evaluate_model(X_test, y_test)  # 自定义验证函数
        return val_score
   