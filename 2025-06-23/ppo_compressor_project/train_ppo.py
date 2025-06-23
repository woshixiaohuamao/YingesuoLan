# 训练+测试PPO
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from compressor_env import CompressorEnv  


# 创建环境
env = CompressorEnv()

# 检查环境是否合规
check_env(env)

# 创建PPO模型
model = PPO("MlpPolicy", env, verbose=1)

# 训练
model.learn(total_timesteps=5000)

# 保存模型
model.save("ppo_compressor")

# 测试训练好的模型
obs, _ = env.reset()  # 获取obs

for _ in range(10):
    action, _ = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
    print(f"动作: {action} | 奖励: {reward}")
    if done or truncated:  
        obs, _ = env.reset() 

