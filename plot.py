import pandas as pd
import matplotlib.pyplot as plt

# Load baseline logs
baseline = pd.read_csv("logs/baseline_monitor.csv", comment="#")

# Load shaped logs
shaped = pd.read_csv("logs/shaped_monitor.csv", comment="#")

# Plot
plt.figure(figsize=(8, 5))

plt.plot(baseline["l"], baseline["r"], label="Baseline Reward")
plt.plot(shaped["l"], shaped["r"], label="Shaped Reward")

plt.xlabel("Timesteps")
plt.ylabel("Mean Reward")
plt.title("Reward Comparison")
plt.legend()

# Save plot
plt.savefig("reward_comparison.png")

print("Plot saved as reward_comparison.png")