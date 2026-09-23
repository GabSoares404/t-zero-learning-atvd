import os
import subprocess
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Config
total_timesteps = 100000

runs = {
    "baseline": ["--override", f"total_timesteps={total_timesteps}"],
    "target_1": ["--override", f"total_timesteps={total_timesteps}", "dqn.target_network_frequency=1"],
    "target_5000": ["--override", f"total_timesteps={total_timesteps}", "dqn.target_network_frequency=5000"],
    "buffer_100": ["--override", f"total_timesteps={total_timesteps}", "dqn.buffer_size=100"],
    "buffer_500": ["--override", f"total_timesteps={total_timesteps}", "dqn.buffer_size=500"],
    "lr_1e-2": ["--override", f"total_timesteps={total_timesteps}", "dqn.learning_rate=1e-2"],
    "lr_1e-5": ["--override", f"total_timesteps={total_timesteps}", "dqn.learning_rate=1e-5"],
}

env = os.environ.copy()
env["WANDB_MODE"] = "offline"
env["PYTHONUNBUFFERED"] = "1"

run_history_files = {}

for name, overrides in runs.items():
    print(f"Running {name}...")
    existing_runs = set()
    if Path("wandb").exists():
        existing_runs = set(Path("wandb").glob("offline-run-*"))
    
    cmd = [r".\.venv\Scripts\python.exe", "train.py", "--config", "dqn_cartpole"] + overrides
    subprocess.run(cmd, env=env, check=True)
    
    if Path("wandb").exists():
        current_runs = set(Path("wandb").glob("offline-run-*"))
        new_runs = current_runs - existing_runs
        if new_runs:
            # Sort to get the latest in case multiple were created (should just be one)
            run_dir = sorted(list(new_runs))[-1]
            history_file = run_dir / "files" / "wandb-history.jsonl"
            run_history_files[name] = history_file
            print(f"Captured {name} at {history_file}")

# Now plot
os.makedirs("plots", exist_ok=True)

def read_metrics(filepath, metric_name):
    steps = []
    vals = []
    with open(filepath, 'r') as f:
        for line in f:
            data = json.loads(line)
            if "global_step" in data and metric_name in data:
                steps.append(data["global_step"])
                vals.append(data[metric_name])
    return steps, vals

def plot_group(run_names, metric_name, title, filename):
    plt.figure(figsize=(10, 6))
    for name in run_names:
        if name in run_history_files and run_history_files[name].exists():
            steps, vals = read_metrics(run_history_files[name], metric_name)
            if steps:
                plt.plot(steps, vals, label=name)
    plt.title(title)
    plt.xlabel("Global Step")
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"plots/{filename}.png")
    plt.close()

# Q1
plot_group(["baseline", "target_1", "target_5000"], "charts/episodic_return_mean_last100", "Q1: Return", "q1_return")
plot_group(["baseline", "target_1", "target_5000"], "losses/td_loss", "Q1: TD Loss", "q1_td_loss")
plot_group(["baseline", "target_1", "target_5000"], "losses/q_values", "Q1: Q Values", "q1_q_values")

# Q2
plot_group(["baseline", "buffer_100", "buffer_500"], "charts/episodic_return_mean_last100", "Q2: Return", "q2_return")
plot_group(["baseline", "buffer_100", "buffer_500"], "losses/q_values", "Q2: Q Values", "q2_q_values")

# Q3
plot_group(["baseline", "lr_1e-2", "lr_1e-5"], "charts/episodic_return_mean_last100", "Q3: Return", "q3_return")
plot_group(["baseline", "lr_1e-2", "lr_1e-5"], "losses/td_loss", "Q3: TD Loss", "q3_td_loss")

print("Plots generated in plots/ directory.")
