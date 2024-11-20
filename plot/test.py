import matplotlib.pyplot as plt

# 创建数据
x = [1, 2, 3, 4]
y = [10, 20, 30, 40]

# 创建图形和轴
fig, ax = plt.subplots()

# 绘制数据
ax.plot(y, x)

# 反转 x 轴和 y 轴的方向
ax.invert_xaxis()
ax.invert_yaxis()

# 设置轴标签
ax.set_xlabel('y')
ax.set_ylabel('x')

# 显示图形
plt.show()