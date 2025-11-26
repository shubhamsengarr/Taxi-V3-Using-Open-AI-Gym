# taxi_with_dataset.py  (full)
import gymnasium as gym
import numpy as np
import pandas as pd
import random
from time import sleep
import os
import matplotlib.pyplot as plt

OUTPUT_DIR = "taxi_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ENV = "Taxi-v3"
NUM_EPISODES = 160000
MAX_STEPS = 200
ALPHA = 0.1
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.999
EVAL_DEMO_EPISODES = 300
EVAL_RENDER_EPISODES = 3
RENDER_DELAY = 0.12

env = gym.make(ENV)
n_states = env.observation_space.n
n_actions = env.action_space.n
Q = np.zeros((n_states, n_actions), dtype=float)

eps = EPS_START
train_rows = []
episode_rewards = []

print("Training and logging transitions...")
for ep in range(1, NUM_EPISODES + 1):
    state, _ = env.reset()
    total_reward = 0.0
    for step in range(MAX_STEPS):
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            action = int(np.argmax(Q[state]))
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
        best_next = np.max(Q[next_state])
        Q[state, action] += ALPHA * ((reward + (0.0 if done else GAMMA * best_next)) - Q[state, action])
        train_rows.append({
            "episode": ep-1, "step": step, "state": int(state), "action": int(action),
            "reward": float(reward), "next_state": int(next_state),
            "terminated": bool(terminated), "truncated": bool(truncated), "done": done
        })
        total_reward += reward
        state = next_state
        if done:
            break
    episode_rewards.append({"episode": ep-1, "total_reward": total_reward})
    eps = max(EPS_END, eps * EPS_DECAY)
    if ep % 1000 == 0 or ep == 1:
        recent = [r["total_reward"] for r in episode_rewards[-100:]]
        avg_recent = float(np.mean(recent)) if recent else 0.0
        print(f"Ep {ep}/{NUM_EPISODES} eps={eps:.4f} last={total_reward:.2f} avg100={avg_recent:.2f}")

# Save artifacts
np.save(os.path.join(OUTPUT_DIR, "taxi_q_table.npy"), Q)
pd.DataFrame(train_rows).to_csv(os.path.join(OUTPUT_DIR, "taxi_transitions_train.csv"), index=False)
pd.DataFrame(episode_rewards).to_csv(os.path.join(OUTPUT_DIR, "taxi_episode_rewards.csv"), index=False)
print("Saved training artifacts in", OUTPUT_DIR)

# Plot training curve
df = pd.DataFrame(episode_rewards)
df['rolling_mean'] = df['total_reward'].rolling(100, min_periods=1).mean()
plt.figure(figsize=(8,4))
plt.plot(df['episode'], df['total_reward'], alpha=0.3)
plt.plot(df['episode'], df['rolling_mean'])
plt.title("Training Progress")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_curve.png"))
plt.close()
print("Saved training curve.")

# Generate demo dataset by running greedy policy
print("Generating demo dataset using greedy policy...")
demo_rows = []
for ep in range(EVAL_DEMO_EPISODES):
    state, _ = env.reset()
    for step in range(MAX_STEPS):
        action = int(np.argmax(Q[state]))
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
        demo_rows.append({
            "episode": ep, "step": step, "state": int(state), "action": int(action),
            "reward": float(reward), "next_state": int(next_state),
            "terminated": bool(terminated), "truncated": bool(truncated), "done": done
        })
        state = next_state
        if done:
            break
pd.DataFrame(demo_rows).to_csv(os.path.join(OUTPUT_DIR, "taxi_policy_demo.csv"), index=False)
print("Saved demo dataset.")

# Optional: render a few eval episodes (for demo)
print("Rendering a couple of evaluation episodes (greedy policy):")
for e in range(EVAL_RENDER_EPISODES):
    state, _ = env.reset()
    print(f"\n--- Eval {e+1} ---")
    total = 0
    for _ in range(MAX_STEPS):
        action = int(np.argmax(Q[state]))
        state, reward, terminated, truncated, _ = env.step(action)
        total += reward
        env.render()
        sleep(RENDER_DELAY)
        if terminated or truncated:
            break
    print("Total reward:", total)

env.close()
print("All done. Files saved in", OUTPUT_DIR)
