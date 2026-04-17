import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from environment import DoublePendulumEnv

parser = argparse.ArgumentParser()
parser.add_argument("--timesteps", type=int, default=10000)
parser.add_argument("--reward_type", type=str, default="shaped")
parser.add_argument("--save_path", type=str, default="models/ppo_model.zip")

args = parser.parse_args()

# Create folders
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Create environment
env = DoublePendulumEnv(reward_type=args.reward_type)

# ✅ Wrap with Monitor (IMPORTANT)
log_file = f"logs/{args.reward_type}_monitor.csv"
env = Monitor(env, filename=log_file)

# Create PPO model
model = PPO("MlpPolicy", env, verbose=1)

# Train
model.learn(total_timesteps=args.timesteps)

# Save model
model.save(args.save_path)

print("Training complete. Model saved.")