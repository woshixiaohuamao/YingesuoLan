import numpy as np

# 空压机数
n = 3

# 每台空压机的产气量
air_production = np.array([10, 20, 15])

# 每台空压机的耗电量
power_consumption = np.array([5, 8, 6])

# 需求总产气量
required_air = 30

# 初始化拉格朗日乘子（罚项），也叫lambda
lagrange_multiplier = 0.5

# 学习率（用于更新乘子）
learning_rate = 0.1

# 最大迭代次数
max_iterations = 50

# 初始解（全关）
x = np.zeros(n)

# 迭代拉格朗日松弛法
for iteration in range(max_iterations):
    costs = power_consumption + lagrange_multiplier * (-air_production)  # 拉格朗日松弛目标函数

    # 贪心选择：如果该机的“拉格朗日成本”<0，则开机（1），否则关（0）
    x = np.where(costs < 0, 1, 0)

    # 计算当前总产气量
    total_air = np.dot(x, air_production)

    # 违反约束程度（需求 - 实际）
    constraint_violation = required_air - total_air

    # 更新拉格朗日乘子（梯度上升）
    lagrange_multiplier += learning_rate * constraint_violation

    # 投影，乘子必须非负
    lagrange_multiplier = max(lagrange_multiplier, 0)

    # 输出每次迭代信息
    print(f"第{iteration+1}次迭代：选中的空压机 {x}，总产气 {total_air}，当前罚项 {lagrange_multiplier:.2f}")

    # 如果满足约束（差值小于1），提前停止
    if abs(constraint_violation) < 1:
        break

# 最终结果
print("\n最终选择的空压机开关方案：", x)
print("最终总产气量：", np.dot(x, air_production))
print("最终总耗电量：", np.dot(x, power_consumption))
