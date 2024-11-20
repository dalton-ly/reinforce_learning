import matplotlib.pyplot as plt

# 读取 s_value 数据
with open('s_value.txt', 'r') as file:
    s_value = list(map(float, file.read().split()))

# 读取 s_policy 数据
with open('s_policy.txt', 'r') as file:
    s_policy = list(map(float, file.read().split()))

# 读取 s_truncated 数据
with open('s_truncated.txt', 'r') as file:
    s_truncated = list(map(float, file.read().split()))

# 确保每个列表至少有40个点，不足的用最后一个值补足
def ensure_length(data, length):
    if len(data) < length:
        data.extend([data[-1]] * (length - len(data)))
    return data[:length]

s_value = ensure_length(s_value, 40)
s_policy = ensure_length(s_policy, 40)
s_truncated = ensure_length(s_truncated, 40)

# 绘制图表
plt.figure(figsize=(10, 6))

plt.plot(s_value, label='Value Iteration', marker='o')
plt.plot(s_policy, label='Policy Iteration', marker='s')
plt.plot(s_truncated, label='Truncated Policy Iteration', marker='^')

plt.xlabel('Iteration')
plt.ylabel('Value')
plt.title('Convergence Speed Comparison')
plt.legend()
plt.grid(True)
plt.show()