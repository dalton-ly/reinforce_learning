import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from grid_world import GridWorld

# 定义网格状态
NORMAL = 0
FORBIDDEN = 1
TARGET = 2

# 定义网格
grid = [
    [NORMAL, NORMAL, NORMAL, NORMAL, NORMAL],
    [NORMAL, FORBIDDEN, FORBIDDEN, NORMAL, NORMAL],
    [NORMAL, NORMAL, FORBIDDEN, NORMAL, NORMAL],
    [NORMAL, FORBIDDEN, TARGET, FORBIDDEN, NORMAL],
    [NORMAL, FORBIDDEN, NORMAL, NORMAL, NORMAL],
]
# 定义颜色映射
color_map = {NORMAL: "white", FORBIDDEN: "orange", TARGET: "cyan"}


actions = [(0, -1), (0, 1), (-1, 0), (1, 0), (0, 0)]  # 对应 LEFT, RIGHT, UP, DOWN, STAY


env = GridWorld(env_size=(5, 5),
                start_state=(0, 0),
                target_state=(2, 3),
                forbidden_states=[(1, 1), (2, 1), (2, 2), (1, 3), (3, 3),(1,4)])

state, _ = env.reset()


steps=[100,1000,10000,1000000]
for i in steps:
    visit_count = np.loadtxt(f"/home/ly/code/cpp/data/1/visit_count_{i}.csv", delimiter=",").reshape(5, 5, 5)
    if i==1e6:
        flattened_counts = visit_count.flatten()

        x_indices = np.arange(len(flattened_counts))

        plt.figure(figsize=(8, 6))
        plt.scatter(x_indices, flattened_counts, color='orange', s=10)
        plt.ylim(14000, 17000)
        plt.title("State-Action Visit Counts")
        plt.xlabel("State-Action Pair Index")
        plt.ylabel("Visit Count")
        plt.grid(True)
        plt.show()
        break
    env.render_from_visit_count(visit_count)    
steps=[100,1000,10000,1000000]
for i in steps:
    visit_count = np.loadtxt(f"/home/ly/code/cpp/data/0.500000/visit_count_{i}.csv", delimiter=",").reshape(5, 5, 5)
    if i==1e6:
        flattened_counts = visit_count.flatten()

        x_indices = np.arange(len(flattened_counts))
        plt.figure(figsize=(8, 6))
        plt.scatter(x_indices, flattened_counts, color='orange', s=10)
        plt.title("State-Action Visit Counts")
        plt.xlabel("State-Action Pair Index")
        plt.ylabel("Visit Count")
        plt.grid(True)
        plt.show()
        break
    env.render_from_visit_count(visit_count)  
    

    


