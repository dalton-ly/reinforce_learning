import numpy as np
import matplotlib.pyplot as plt

# 随机生成 400 个样本，均匀分布在以原点为中心、边长为 30 的正方形区域
np.random.seed(0)
samples = np.random.uniform(-15, 15, size=(400, 2))

# 目标值为均值（理论真值为 0）
true_mean = np.array([0, 0])
print(true_mean)

# 随机梯度下降函数
def stochastic_gradient_descent(samples, initial_w, learning_rate_func, max_iter):
    w = initial_w
    errors = []
    trajectory = [w.copy()]
    for k in range(1, max_iter + 1):
        idx = np.random.randint(0, len(samples))  # 随机选取一个样本
        sample = samples[idx]
        gradient = w - sample
        alpha = learning_rate_func(k)
        w -= alpha * gradient
        errors.append(np.linalg.norm(w - true_mean))
        trajectory.append(w.copy())
    return np.array(trajectory), errors

# 学习率策略
def constant_learning_rate(alpha):
    return lambda k: alpha

def decreasing_learning_rate(c):
    return lambda k: c / k

# 设置初值和最大迭代次数
initial_w = np.array([50.0, 50.0])
max_iter = 100
c1,c2=1,2
# 定义不同的学习率策略
lr1 = constant_learning_rate(0.05)
lr2 = decreasing_learning_rate(c1)
lr3 = decreasing_learning_rate(c2)

# 应用三种学习率策略
trajectory1, errors1 = stochastic_gradient_descent(samples, initial_w.copy(), lr1, max_iter)
trajectory2, errors2 = stochastic_gradient_descent(samples, initial_w.copy(), lr2, max_iter)
trajectory3, errors3 = stochastic_gradient_descent(samples, initial_w.copy(), lr3, max_iter)

# 绘图：样本点与梯度下降路径
# plt.figure(figsize=(12, 8))

# 图 1：收敛轨迹
# plt.subplot(2, 1, 1)
plt.scatter(samples[:, 0], samples[:, 1], s=10, color='gray', label='Samples')
plt.plot(trajectory1[:, 0], trajectory1[:, 1], '-o', label='Constant LR: 0.05')
plt.plot(trajectory2[:, 0], trajectory2[:, 1], '-o', label=f'Decreasing LR: {c1}/k')
plt.plot(trajectory3[:, 0], trajectory3[:, 1], '-o', label=f'Decreasing LR: {c2}/k')
plt.xlim([-20,20])
plt.ylim([-20,20])
plt.scatter(true_mean[0], true_mean[1], c='red', label='True Mean', zorder=5)
plt.title('Figure 1: Convergence Trajectory')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()

# 图 2：误差随迭代次数变化
plt.figure()
plt.plot(errors1, label='Constant LR: 0.005')
plt.plot(errors2, label=f'Decreasing LR: {c1}/k')
plt.plot(errors3, label=f'Decreasing LR: {c2}/k')
plt.title('Figure 2: Error Convergence')
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.legend()

plt.tight_layout()
plt.show()
