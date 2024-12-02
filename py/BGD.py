import numpy as np
import matplotlib.pyplot as plt

# 随机生成 400 个样本，均匀分布在以原点为中心、边长为 30 的正方形区域
np.random.seed(0)
samples = np.random.uniform(-15, 15, size=(400, 2))

# 目标值为均值（理论真值为 0）
true_mean = np.mean(samples, axis=0)

# 小批量梯度下降函数
def minibatch_gradient_descent(samples, initial_w, batch_size, learning_rate_func, max_iter):
    w = initial_w
    errors = []
    trajectory = [w.copy()]
    n = len(samples)
    for k in range(1, max_iter + 1):
        # 随机选取小批量数据
        indices = np.random.choice(n, batch_size, replace=False)
        batch = samples[indices]
        gradient = w - np.mean(batch, axis=0)  # 计算小批量梯度
        alpha = learning_rate_func(k)
        w -= alpha * gradient
        errors.append(np.linalg.norm(w - true_mean))
        trajectory.append(w.copy())
    
    return np.array(trajectory), errors

# 学习率策略：\alpha_k = 1/k
learning_rate = lambda k: 1 / k

# 设置初值和最大迭代次数
initial_w = np.array([50.0, 50.0])
max_iter = 100

# 不同小批量大小
batch_sizes = [1, 10, 50, 100]

# 存储每种小批量大小对应的轨迹和误差
trajectories = []
errors_list = []

for batch_size in batch_sizes:
    traj, errs = minibatch_gradient_descent(samples, [50.0,50.0], batch_size, learning_rate, max_iter)
    trajectories.append(traj)
    errors_list.append(errs)


fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs = axs.flatten()
colors = ['blue', 'green', 'orange', 'purple']
# 图 1：收敛轨迹
for i, (traj, batch_size) in enumerate(zip(trajectories, batch_sizes)):
    axs[i].plot(traj[:, 0], traj[:, 1], '-o', color=colors[i], label=f'Batch size: {batch_size}')
    axs[i].scatter(samples[:, 0], samples[:, 1], s=10, color='gray', label='Samples')
    axs[i].scatter(true_mean[0], true_mean[1], c='red', label='True Mean', zorder=5)
    axs[i].set_title(f'Batch size: {batch_size}')
    axs[i].set_xlabel('X1')
    axs[i].set_ylabel('X2')
    axs[i].set_xlim(-15, 15)  # 设置 x 轴范围
    axs[i].set_ylim(-15, 15)  # 设置 y 轴范围
    axs[i].legend()

plt.title('Figure 1: Convergence Trajectory')
plt.tight_layout()
plt.show()


plt.figure()
for i, (errors, batch_size) in enumerate(zip(errors_list, batch_sizes)):
    plt.plot(errors, label=f'Batch size: {batch_size}')
plt.title('Figure 2: Error Convergence')
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.legend()
plt.show()

