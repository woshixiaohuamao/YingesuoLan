import gymnasium as gym  
from gymnasium import spaces
import numpy as np

class CompressorEnv(gym.Env):  
    def __init__(self):
        super(CompressorEnv, self).__init__()
        
        self.num_compressors = 3
        self.air_production = np.array([10, 20, 15])
        self.power_consumption = np.array([5, 8, 6])
        self.required_air = 30
        self.max_steps = 10
        self.current_step = 0

        self.observation_space = spaces.Box(low=0, high=50, shape=(self.num_compressors + 1,), dtype=np.float32)
        self.action_space = spaces.MultiBinary(self.num_compressors)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.required_air = np.random.randint(20, 40)
        state = np.append(self.required_air, [0] * self.num_compressors)
        return state.astype(np.float32), {} 

    def step(self, action):
        self.current_step += 1
        total_air = np.dot(action, self.air_production)
        total_power = np.dot(action, self.power_consumption)
        
        air_error = abs(self.required_air - total_air)
        reward = - (air_error + 0.1 * total_power)

        done = self.current_step >= self.max_steps
        next_state = np.append(self.required_air, action)
        
        return next_state.astype(np.float32), reward, done, False, {}  # 新版step返回5个值：obs, reward, terminated, truncated, info

    def render(self):
        pass
