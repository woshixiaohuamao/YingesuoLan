import numpy as np
from scipy.optimize import minimize, LinearConstraint
import matplotlib.pyplot as plt
import time

class Compressor:
    def __init__(self, id, min_flow, max_flow, power_coeffs, start_cost, min_on, min_off):
        self.id = id
        self.min_flow = min_flow
        self.max_flow = max_flow
        self.a, self.b, self.c = power_coeffs
        self.start_cost = start_cost
        self.min_on = min_on
        self.min_off = min_off
        self.status_history = []  # 记录历史状态用于约束检查
    
    def power_consumption(self, flow):
        return self.a * flow**2 + self.b * flow + self.c
    
    def operating_cost(self, flow, status, prev_status, electricity_price):
        energy_cost = self.power_consumption(flow) * electricity_price
        startup_cost = self.start_cost if status == 1 and prev_status == 0 else 0
        return energy_cost + startup_cost
    
    def can_start(self, t):
        """检查是否可以启动（满足最小停机时间）"""
        if t < self.min_off: 
            return False
        # 检查最近min_off小时是否都处于停机状态
        if len(self.status_history) < self.min_off:
            return True
        return all(s == 0 for s in self.status_history[-self.min_off:])
    
    def can_stop(self, t):
        """检查是否可以停机（满足最小运行时间）"""
        if t < self.min_on: 
            return False
        # 检查最近min_on小时是否都处于运行状态
        if len(self.status_history) < self.min_on:
            return True
        return all(s == 1 for s in self.status_history[-self.min_on:])

class CompressorScheduler:
    def __init__(self, compressors, demand_profile, reserve_profile, electricity_price, periods=24):
        self.compressors = compressors
        self.demand = np.array(demand_profile)
        self.reserve = np.array(reserve_profile)
        self.electricity_price = np.array(electricity_price)
        self.periods = periods
        self.num_compressors = len(compressors)
        
        # 算法参数 - 调整为更实用的值
        self.max_iter = 50        # 减少迭代次数
        self.pop_size = 10        # 减小种群大小
        self.F = 0.5              # 减小缩放因子
        self.CR = 0.7             # 降低交叉概率
        self.rho = 1.0            # 增大步长
        self.epsilon = 0.05       # 增大收敛阈值(5%)
        
        # 记录最优解
        self.best_solution = None
        self.best_cost = float('inf')
        self.dual_gap_history = []
        self.iteration_info = []  # 存储迭代信息
        
        # 重置所有压缩机的状态历史
        for comp in self.compressors:
            comp.status_history = []

    def lagrangian_relaxation(self, lambdas):
        total_cost = 0
        all_flows = []
        all_status = []
        
        for comp in self.compressors:
            flows, status, comp_cost = self.solve_single_compressor(comp, lambdas)
            total_cost += comp_cost
            all_flows.append(flows)
            all_status.append(status)
            comp.status_history = list(status)  # 更新状态历史
        
        total_flow = np.sum(all_flows, axis=0)
        penalty = np.sum(lambdas * (total_flow - self.demand))
        L = total_cost - penalty
        
        return L, np.array(all_flows), np.array(all_status), total_flow

    def solve_single_compressor(self, comp, lambdas):
        """改进的单机优化 - 考虑启停约束"""
        flows = np.zeros(self.periods)
        status = np.zeros(self.periods, dtype=int)
        total_cost = 0
        
        # 初始状态设为停机
        prev_status = 0
        
        for t in range(self.periods):
            # 1. 启停决策 - 考虑最小启停时间约束
            if prev_status == 1:  # 当前正在运行
                if comp.can_stop(t) and self.demand[t] < 0.5 * comp.max_flow:
                    status[t] = 0
                else:
                    status[t] = 1
            else:  # 当前处于停机状态
                if comp.can_start(t) and self.demand[t] > 0.6 * comp.max_flow:
                    status[t] = 1
                else:
                    status[t] = 0
            
            # 2. 气量优化
            if status[t] == 1:
                # 考虑拉格朗日乘子的最优气量
                opt_flow = (lambdas[t] - comp.b) / (2 * comp.a)
                flows[t] = max(comp.min_flow, min(comp.max_flow, opt_flow))
            else:
                flows[t] = 0
            
            # 3. 成本计算
            cost = comp.operating_cost(flows[t], status[t], prev_status, self.electricity_price[t])
            total_cost += cost
            prev_status = status[t]
        
        return flows, status, total_cost

    def check_feasibility(self, total_flow):
        """放宽可行性检查条件"""
        demand_error = np.max(np.abs(total_flow - self.demand))
        print(f"需求最大偏差: {demand_error:.2f} m³/min")
        
        # 计算备用能力
        total_capacity = sum(c.max_flow for c in self.compressors)
        reserve_capacity = total_capacity - total_flow
        reserve_error = np.max(self.reserve - reserve_capacity)
        print(f"备用最大缺口: {max(0, reserve_error):.2f} m³/min")
        
        # 放宽条件
        return demand_error < 2.0 and reserve_error < 1.0

    def economic_dispatch(self, status_matrix):
        """改进的经济负荷分配"""
        flows_matrix = np.zeros((self.num_compressors, self.periods))
        total_cost = 0
        
        for t in range(self.periods):
            running_indices = [i for i in range(self.num_compressors) if status_matrix[i, t] == 1]
            n_running = len(running_indices)
            
            if n_running == 0:
                # 没有压缩机运行，但可能有需求 - 强制启动一台
                min_cost_comp = min(self.compressors, key=lambda c: c.operating_cost(c.min_flow, 1, 0, self.electricity_price[t]))
                idx = self.compressors.index(min_cost_comp)
                running_indices = [idx]
                status_matrix[idx, t] = 1
                print(f"时段 {t+1}: 无压缩机运行，强制启动压缩机 {min_cost_comp.id}")
            
            # 边界约束
            bounds = [
                (self.compressors[i].min_flow, self.compressors[i].max_flow)
                for i in running_indices
            ]
            
            # 初始猜测 - 按比例分配
            x0 = [self.demand[t] / n_running] * n_running
            
            # 约束：总流量=需求
            A = np.ones((1, n_running))
            constraints = LinearConstraint(A, lb=self.demand[t], ub=self.demand[t])
            
            # 成本函数
            def cost_func(Q):
                cost = 0
                for i, comp_idx in enumerate(running_indices):
                    comp = self.compressors[comp_idx]
                    # 使用当前状态和前一时段状态（简化处理）
                    prev_status = status_matrix[comp_idx, t-1] if t > 0 else 0
                    cost += comp.operating_cost(Q[i], 1, prev_status, self.electricity_price[t])
                return cost
            
            # 求解
            res = minimize(cost_func, x0, bounds=bounds, constraints=constraints)
            
            if not res.success:
                print(f"时段 {t+1} 经济分配失败: {res.message}")
                # 使用初始分配作为后备方案
                assigned_flow = 0
                for i, comp_idx in enumerate(running_indices[:-1]):
                    comp = self.compressors[comp_idx]
                    flow = min(comp.max_flow, max(comp.min_flow, self.demand[t] - assigned_flow))
                    flows_matrix[comp_idx, t] = flow
                    assigned_flow += flow
                flows_matrix[running_indices[-1], t] = self.demand[t] - assigned_flow
                total_cost += cost_func([flows_matrix[i, t] for i in running_indices])
            else:
                for i, comp_idx in enumerate(running_indices):
                    flows_matrix[comp_idx, t] = res.x[i]
                total_cost += res.fun
        
        return flows_matrix, total_cost

    def solve(self):
        lambdas = np.full(self.periods, 5.0)  # 从更合理的值开始
        maxL = -float('inf')
        minCost = float('inf')
        iteration = 0
        
        print("开始优化...")
        print(f"{'迭代':<5} | {'对偶值':<10} | {'原始值':<10} | {'对偶间隙':<10} | {'可行性':<8}")
        print("-" * 50)
        
        while iteration < self.max_iter:
            iteration += 1
            start_time = time.time()
            
            # 步骤3：求解单机问题
            L, flows_matrix, status_matrix, total_flow = self.lagrangian_relaxation(lambdas)
            
            # 步骤4：更新对偶上界
            if L > maxL:
                maxL = L
            
            # 步骤5：可行性检查
            feasible = self.check_feasibility(total_flow)
            actual_cost = float('inf')
            
            if feasible:
                # 步骤6：经济分配
                _, actual_cost = self.economic_dispatch(status_matrix)
                
                if actual_cost < minCost:
                    minCost = actual_cost
                    self.best_solution = (flows_matrix.copy(), status_matrix.copy())
                    print(f"找到更优解! 成本: {minCost:.2f}元")
            
            # 步骤7：收敛判断
            dual_gap = 0
            if minCost < float('inf') and maxL > -float('inf'):
                dual_gap = (minCost - maxL) / minCost
                self.dual_gap_history.append(dual_gap)
                
                if dual_gap < self.epsilon:
                    print(f"收敛于迭代 {iteration}, 对偶间隙: {dual_gap:.4f}")
                    break
            
            # 记录迭代信息
            iter_time = time.time() - start_time
            self.iteration_info.append({
                'iter': iteration,
                'dual_value': L,
                'primal_value': actual_cost if feasible else None,
                'dual_gap': dual_gap,
                'feasible': feasible,
                'time': iter_time
            })
            
            # 打印迭代信息
            primal_str = f"{actual_cost:.2f}" if feasible else "N/A"
            print(f"{iteration:<5} | {L:<10.2f} | {primal_str:<10} | {dual_gap*100 if dual_gap>0 else 'N/A':<10.2f}% | {'是' if feasible else '否':<8}")
            
            # 更新乘子
            if iteration % 5 == 0:  # 每5次迭代使用DE
                print("使用差分进化优化乘子...")
                lambdas = self.differential_evolution(lambdas, L)
            else:
                lambdas = self.subgradient_update(lambdas, total_flow)
        
        return self.best_solution, minCost


# ===================== 增强的测试案例 =====================
if __name__ == "__main__":
    print("="*50)
    print("空压机智能调度优化系统")
    print("="*50)
    
    # 创建更合理的空压机参数
    compressor1 = Compressor(
        id=1,
        min_flow=10,
        max_flow=30,
        power_coeffs=[0.02, 0.8, 15],  # 功率 = 0.02*Q² + 0.8*Q + 15
        start_cost=80,
        min_on=3,
        min_off=2
    )
    
    compressor2 = Compressor(
        id=2,
        min_flow=15,
        max_flow=40,
        power_coeffs=[0.015, 0.7, 20],
        start_cost=100,
        min_on=4,
        min_off=3
    )
    
    compressor3 = Compressor(
        id=3,
        min_flow=5,
        max_flow=25,
        power_coeffs=[0.025, 0.6, 10],
        start_cost=60,
        min_on=2,
        min_off=1
    )
    
    compressors = [compressor1, compressor2, compressor3]
    
    # 更合理的气量需求曲线 (24小时)
    demand_profile = [
        20, 25, 30, 35, 40, 50, 60, 65, 70, 75, 
        80, 85, 80, 75, 70, 65, 60, 55, 50, 45,
        40, 35, 30, 25
    ]
    
    # 备用要求 (总容量的15-20%)
    max_capacity = sum(c.max_flow for c in compressors)  # 30+40+25=95
    reserve_profile = [0.2 * d for d in demand_profile]
    
    # 更合理的电价曲线 (元/kWh)
    electricity_price = [
        0.4, 0.4, 0.4, 0.4, 0.4, 0.5,
        0.6, 0.7, 0.8, 0.8, 0.8, 0.9,
        1.0, 1.0, 1.0, 1.0, 1.0, 0.9,
        0.8, 0.7, 0.6, 0.5, 0.4, 0.4
    ]
    
    print("\n系统配置:")
    print(f"- 压缩机数量: {len(compressors)}台")
    print(f"- 总最大流量: {max_capacity} m³/min")
    print(f"- 最大需求: {max(demand_profile)} m³/min")
    print(f"- 调度时段: 24小时")
    
    print("\n开始优化计算...")
    scheduler = CompressorScheduler(
        compressors=compressors,
        demand_profile=demand_profile,
        reserve_profile=reserve_profile,
        electricity_price=electricity_price,
        periods=24
    )
    
    solution, min_cost = scheduler.solve()
    
    if solution:
        print(f"\n优化完成! 最低成本: {min_cost:.2f} 元")
        
        # 可视化结果
        scheduler.visualize_schedule()
        
        # 输出对偶间隙收敛过程
        plt.figure(figsize=(10, 5))
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.plot(range(1, len(scheduler.dual_gap_history)+1), 
                [gap*100 for gap in scheduler.dual_gap_history], 
                'o-', linewidth=2)
        plt.title('对偶间隙收敛过程')
        plt.xlabel('迭代次数')
        plt.ylabel('对偶间隙 (%)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        # 输出详细调度方案
        flows, status = solution
        print("\n详细调度方案:")
        print("时间 | 需求 | 总供气 | 压缩机1 | 压缩机2 | 压缩机3 | 状态1 | 状态2 | 状态3")
        for t in range(24):
            total_flow = sum(flows[i][t] for i in range(3))
            print(f"{t+1:2d}h | {demand_profile[t]:3.0f} | {total_flow:6.1f} | "
                f"{flows[0][t]:7.1f} | {flows[1][t]:7.1f} | {flows[2][t]:7.1f} | "
                f"{status[0][t]:^5} | {status[1][t]:^5} | {status[2][t]:^5}")
    else:
        print("\n警告: 未找到可行解，请检查约束条件")
        
        # 输出最后一次尝试的总流量
        _, _, _, total_flow = scheduler.lagrangian_relaxation(np.zeros(24))
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, 25), demand_profile, 'r-', label='需求')
        plt.plot(range(1, 25), total_flow, 'b--', label='实际供气')
        plt.title('需求与实际供气对比')
        plt.xlabel('时间 (小时)')
        plt.ylabel('气量 (m³/min)')
        plt.legend()
        plt.grid(True)
        plt.show()