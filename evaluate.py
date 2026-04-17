import argparse
import imageio
import os

from stable_baselines3 import PPO
from environment import DoublePendulumEnv

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="models/ppo_model.zip")
parser.add_argument("--gif_path", type=str, default="media/output.gif")
parser.add_argument("--steps", type=int, default=300)

args = parser.parse_args()

# Create folders
os.makedirs("media", exist_ok=True)

# Load env + model
env = DoublePendulumEnv()
model = PPO.load(args.model_path, env=env)

obs = env.reset()

frames = []

for _ in range(args.steps):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)

    # Dummy frame (since no pygame yet)
    frame = env._get_obs()  # placeholder visualization
    frames.append(frame)

    if done:
        obs = env.reset()

# Convert frames to image (simple visualization)
frames = [((f - f.min()) / (f.max() - f.min() + 1e-5) * 255).astype("uint8") for f in frames]

imageio.mimsave(args.gif_path, frames, fps=30)

print(f"GIF saved at {args.gif_path}")