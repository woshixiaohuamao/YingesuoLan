"""
空压设备智能推荐模块
功能：根据历史配置数据推荐最优设备启停方案
作者：赵泽宸
版本：1.0
"""

import json
from typing import Dict, List, Optional

class DeviceRecommender:
    def __init__(self, config_path: str = "hourly_config.json"):
        try:
            with open(config_path) as f:
                self.config = json.load(f)
            self._preprocess_config()
        except FileNotFoundError:
            raise ValueError(f"配置文件 {config_path} 不存在")
        except json.JSONDecodeError:
            raise ValueError("配置文件格式错误")

    def _preprocess_config(self) -> None:
        self.optimized_config = {}
        for hour, data in self.config.items():
            self.optimized_config[hour] = {
                int(count): sorted(combo_list, key=lambda x: x["gas"])
                for count, combo_list in data.items()
            }

    def _find_closest(self, sorted_combos: List[Dict], target_gas: float) -> Dict:
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

    def recommend(self, hour: int, target_gas: float, strategy: str = "efficient") -> Optional[Dict]:
        if not 0 <= hour <= 23:
            raise ValueError("小时参数必须在0-23之间")
        if target_gas <= 0:
            raise ValueError("目标流量必须大于0")

        hour_key = str(hour)
        if hour_key not in self.optimized_config:
            return None

        hour_data = self.optimized_config[hour_key]
        device_counts = sorted(hour_data.keys())
        candidates = []

        for count in device_counts:
            closest_combo = self._find_closest(hour_data[count], target_gas)
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
            "recommended_count": best["device_count"],
            "history_gas": best["combo"]["gas"],
            "gas_difference": round(best["gas_diff"], 2),
            "active_devices": active_devices,
            "device_config": best["combo"]["devices"]
        }


if __name__ == "__main__":
    # 测试推荐逻辑
    test_cases = [
        (9, 100, "efficient"),
        (14, 120, "safe"),
        (25, 100, "accurate")  # 非法测试
    ]
    
    try:
        recommender = DeviceRecommender()
        for h, g, s in test_cases:
            print(f"\n=== 测试案例：{h}时，目标{g}，策略'{s}' ===")
            result = recommender.recommend(h, g, s)
            print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"模块测试失败：{str(e)}")