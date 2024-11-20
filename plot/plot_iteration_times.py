import matplotlib.pyplot as plt

# 读取 statevalues 数据
statevalues = []
with open('statevalues.txt', 'r') as file:
    for line in file:
        statevalues.append(list(map(float, line.split())))

# 确保每个列表至少有40个点，不足的用最后一个值补足
def ensure_length(data, length):
    if len(data) < length:
        data.extend([data[-1]] * (length - len(data)))
    return data[:length]

# 处理每个列表
times = [1, 5, 9, 56]
optimal = 5.31441  # 最优值

# 计算误差
errors = []
for s_truncated in statevalues:
    s_truncated = ensure_length(s_truncated, 20)
    error = [abs(v - optimal) for v in s_truncated]
    errors.append(error)

# 绘制图表
plt.figure(figsize=(10, 6))

for i, error in enumerate(errors):
    plt.plot(error, label=f'Truncated Policy Iteration (times={times[i]})', marker='o')

plt.axhline(0, color='black', linewidth=0.8)  # 画一条以0为中心的水平线
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title('Error Comparison for Different Iteration Times')
plt.legend()
plt.grid(True)
plt.show()