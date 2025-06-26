import json

class DeviceRecommender:
    def __init__(self, config_path):
        with open(config_path) as f:
            self.config = json.load(f)
        
        # 预处理配置数据：按小时和设备数量建立快速索引
        self.optimized_config = {}
        for hour, data in self.config.items():
            self.optimized_config[hour] = {
                int(count): sorted(combo_list, key=lambda x: x["gas"])
                for count, combo_list in data.items()
            }

    def find_closest(self, sorted_combos, target_gas):
        """二分查找最接近目标流量的配置"""
        left = 0
        right = len(sorted_combos) - 1
        closest = None
        min_diff = float('inf')

        while left <= right:
            mid = (left + right) // 2
            current_gas = sorted_combos[mid]["gas"]
            
            diff = abs(current_gas - target_gas)
            if diff < min_diff:
                min_diff = diff
                closest = sorted_combos[mid]
            
            if current_gas < target_gas:
                left = mid + 1
            else:
                right = mid - 1
        
        return closest

    def recommend(self, hour, target_gas, strategy="efficient"):
        """
        推荐设备组合
        :param hour: 当前小时（0-23）
        :param target_gas: 需要达到的流量值
        :param strategy: 
           - "accurate"：最精确匹配
           - "efficient"：最少设备优先
           - "safe"：不低于目标值的最小配置
        """
        hour_key = str(hour)
        if hour_key not in self.optimized_config:
            return None

        candidates = []
        hour_data = self.optimized_config[hour_key]

        # 获取所有可能的设备数量（升序排列）
        device_counts = sorted(hour_data.keys())

        for count in device_counts:
            # 使用预处理的有序数据进行快速查找
            closest_combo = self.find_closest(hour_data[count], target_gas)
            
            if closest_combo:
                candidates.append({
                    "device_count": count,
                    "gas_diff": abs(closest_combo["gas"] - target_gas),
                    "combo": closest_combo
                })

        if not candidates:
            return None

        # 根据策略选择最佳方案
        if strategy == "accurate":
            # 选择绝对差值最小的（直接找流量差最小的配置）
            # 核心目标：流量值与目标值尽可能接近，不考虑设备数量
            best = min(candidates, key=lambda x: x["gas_diff"])
        elif strategy == "efficient":
            # 在满足差值阈值内选择设备最少的（差值范围内的最小设备数）
            # 核心目标：在流量接近目标值的前提下，尽可能少用设备
            threshold = target_gas * 0.05  # 5%容差
            valid = [c for c in candidates if c["gas_diff"] <= threshold]
            best = min(valid, key=lambda x: x["device_count"]) if valid else min(candidates, key=lambda x: x["gas_diff"])
        elif strategy == "safe":
            # 选择不低于目标值的最小配置
            # 核心目标：流量必须不低于目标值，并在满足条件的前提下尽可能少用设备
            over_candidates = [c for c in candidates if c["combo"]["gas"] >= target_gas]
            if over_candidates:
                best = min(over_candidates, key=lambda x: x["device_count"])
            else:
                best = max(candidates, key=lambda x: x["combo"]["gas"])
        else:
            raise ValueError("Invalid strategy")

        # 提取设备开启列表
        active_devices = [k for k, v in best["combo"]["devices"].items() if v == 1]
        
        return {
            "recommended_count": best["device_count"], # 推荐的设备数量
            "history_gas": best["combo"]["gas"], # 历史的流量值
            "gas_difference": round(best["gas_diff"], 2), # 目标值与历史值的差值
            "active_devices": active_devices, # 开启的设备列表
            "device_config": best["combo"]["devices"] # 完整的设备配置
        }

# 使用示例
if __name__ == "__main__":
    recommender = DeviceRecommender("hourly_config.json")
    
    # 示例1：在18点时需要85流量的推荐（精确模式）
    print("示例1 精确推荐：")
    print(json.dumps(recommender.recommend(18, 85, "accurate"), indent=2))
    
    # 示例2：在9点时需要100流量（高效节能模式）
    print("\n示例2 高效推荐：")
    print(json.dumps(recommender.recommend(9, 100, "efficient"), indent=2))
    
    # 示例3：在14点时需要120流量（安全模式）
    print("\n示例3 安全推荐：")
    print(json.dumps(recommender.recommend(14, 120, "safe"), indent=2))